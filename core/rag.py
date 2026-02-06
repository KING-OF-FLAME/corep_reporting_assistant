# core/rag.py
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# -----------------------------
# Data structures
# -----------------------------


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    source_title: str
    source_url: str
    doc_id: str
    doc_type: str  # "pdf"|"txt"|"url"
    page: Optional[int]
    heading_path: str
    paragraph_id: str
    text: str
    text_sha256: str


@dataclass(frozen=True)
class RagSearchResult:
    score: float
    chunk: RagChunk


# -----------------------------
# Core index (TF-IDF)
# -----------------------------


class RagIndex:
    def __init__(self, chunks: List[RagChunk], vectorizer: TfidfVectorizer, matrix: Any):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

    def search(self, query: str, top_k: int = 6) -> List[RagSearchResult]:
        q = (query or "").strip()
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        # Get top_k indices by similarity
        if top_k <= 0:
            top_k = 6
        top_k = min(top_k, len(self.chunks))

        idxs = sims.argsort()[::-1][:top_k]
        results: List[RagSearchResult] = []
        for i in idxs:
            chunk = self.chunks[int(i)]
            score = float(sims[int(i)])
            # Filter out zero matches to reduce noise
            if score <= 0.0:
                continue
            results.append(RagSearchResult(score=score, chunk=chunk))
        return results


# -----------------------------
# Public API
# -----------------------------


def build_or_load_index(
    source_dir: str | Path,
    index_dir: str | Path,
    force_rebuild: bool = False,
) -> RagIndex:
    """
    Builds or loads a local TF-IDF RAG index.

    Data sources (prototype):
    - PDFs and .txt files in source_dir
    - Optional 'sources.json' for URL sources (see _load_url_sources_spec)

    Index artifacts saved in index_dir:
    - chunks.jsonl
    - tfidf.pkl   (vectorizer + sparse matrix)
    """
    source_dir_p = Path(source_dir)
    index_dir_p = Path(index_dir)
    index_dir_p.mkdir(parents=True, exist_ok=True)

    chunks_path = index_dir_p / "chunks.jsonl"
    tfidf_path = index_dir_p / "tfidf.pkl"

    if not force_rebuild and chunks_path.exists() and tfidf_path.exists():
        chunks = _load_chunks_jsonl(chunks_path)
        vectorizer, matrix = _load_tfidf(tfidf_path)
        return RagIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)

    chunks = ingest_sources(source_dir_p)
    if not chunks:
        # Build an empty index (still usable, but search returns nothing)
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([""])
        _save_chunks_jsonl(chunks_path, [])
        _save_tfidf(tfidf_path, vectorizer, matrix)
        return RagIndex(chunks=[], vectorizer=vectorizer, matrix=matrix)

    texts = [c.text for c in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(texts)

    _save_chunks_jsonl(chunks_path, chunks)
    _save_tfidf(tfidf_path, vectorizer, matrix)
    return RagIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def ingest_sources(source_dir: Path) -> List[RagChunk]:
    """
    Ingest local sources for retrieval.

    Supported:
    - *.pdf
    - *.txt, *.md
    - Optional URLs from data/sources.json

    Chunking approach:
    - paragraph-ish chunks using blank lines + sentence boundaries fallback
    - stable paragraph IDs: <doc_id>.p<page?>.par<index>
    """
    chunks: List[RagChunk] = []

    # Local PDFs
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        chunks.extend(_ingest_pdf(pdf_path))

    # Local TXT/MD
    for txt_path in sorted(list(source_dir.glob("*.txt")) + list(source_dir.glob("*.md"))):
        chunks.extend(_ingest_text_file(txt_path))

    # Optional URL sources spec: data/sources.json
    url_spec_path = source_dir / "sources.json"
    if url_spec_path.exists():
        url_specs = _load_url_sources_spec(url_spec_path)
        for spec in url_specs:
            chunks.extend(_ingest_url_source(spec))

    # De-duplicate chunks by text hash + paragraph id (best effort)
    seen = set()
    deduped: List[RagChunk] = []
    for c in chunks:
        key = (c.doc_id, c.paragraph_id, c.text_sha256)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped


def to_retrieved_excerpts(results: List[RagSearchResult]) -> List[Dict[str, Any]]:
    """
    Converts search results into the 'retrieved_excerpts' format expected by prompts.
    """
    out: List[Dict[str, Any]] = []
    for r in results:
        c = r.chunk
        out.append(
            {
                "score": round(r.score, 6),
                "source_title": c.source_title,
                "source_url": c.source_url,
                "paragraph_id": c.paragraph_id,
                "heading_path": c.heading_path,
                "page": c.page,
                "text": c.text,
                "text_sha256": c.text_sha256,
            }
        )
    return out


# -----------------------------
# Ingestion implementations
# -----------------------------


def _ingest_pdf(path: Path) -> List[RagChunk]:
    doc_id = _slugify(path.stem)
    source_title = path.stem
    source_url = str(path.resolve())
    doc_type = "pdf"

    chunks: List[RagChunk] = []
    try:
        reader = PdfReader(str(path))
    except Exception:
        return chunks

    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = _normalize_text(text)
        if not text.strip():
            continue

        paras = _split_into_paragraphs(text)
        for par_idx, par in enumerate(paras, start=1):
            par = par.strip()
            if not par:
                continue

            paragraph_id = f"{doc_id}.p{page_idx+1}.par{par_idx}"
            chunk_id = _make_chunk_id(doc_id, paragraph_id, par)
            chunks.append(
                RagChunk(
                    chunk_id=chunk_id,
                    source_title=source_title,
                    source_url=source_url,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    page=page_idx + 1,
                    heading_path="",
                    paragraph_id=paragraph_id,
                    text=par,
                    text_sha256=_sha256(par),
                )
            )
    return chunks


def _ingest_text_file(path: Path) -> List[RagChunk]:
    doc_id = _slugify(path.stem)
    source_title = path.name
    source_url = str(path.resolve())
    doc_type = "txt"

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    raw = _normalize_text(raw)
    if not raw.strip():
        return []

    # Attempt to preserve headings (Markdown-style) as heading_path context
    # We'll chunk by blank lines, but track the most recent heading encountered.
    chunks: List[RagChunk] = []
    heading = ""

    blocks = re.split(r"\n\s*\n+", raw)
    par_counter = 0
    for b in blocks:
        b = b.strip()
        if not b:
            continue

        # Detect a heading-only block
        if re.match(r"^\s{0,3}#{1,6}\s+\S+", b):
            heading = _strip_md_heading(b)
            continue

        # If block is too large, further split
        for par in _split_large_block(b, max_chars=1200):
            par_counter += 1
            paragraph_id = f"{doc_id}.par{par_counter}"
            chunk_id = _make_chunk_id(doc_id, paragraph_id, par)
            chunks.append(
                RagChunk(
                    chunk_id=chunk_id,
                    source_title=source_title,
                    source_url=source_url,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    page=None,
                    heading_path=heading,
                    paragraph_id=paragraph_id,
                    text=par,
                    text_sha256=_sha256(par),
                )
            )

    return chunks


def _ingest_url_source(spec: Dict[str, Any]) -> List[RagChunk]:
    """
    URL ingestion is optional. Spec format:
    {
      "source_title": "...",
      "source_url": "https://...",
      "doc_id": "optional-stable-id",
      "timeout_s": 20
    }

    Prototype behavior: fetch text-only and chunk.
    - If HTML, we strip tags crudely (no heavy dependencies).
    - For high-fidelity citations, prefer local PDFs placed in ./data.
    """
    source_title = str(spec.get("source_title") or "URL Source")
    source_url = str(spec.get("source_url") or "").strip()
    if not source_url:
        return []

    doc_id = _slugify(str(spec.get("doc_id") or source_title)) or _slugify(source_url)
    timeout_s = int(spec.get("timeout_s") or 20)

    try:
        resp = requests.get(source_url, timeout=timeout_s, headers={"User-Agent": "corep-prototype/1.0"})
        resp.raise_for_status()
        content = resp.text
    except Exception:
        return []

    text = _normalize_text(_strip_html(content))
    if not text.strip():
        return []

    chunks: List[RagChunk] = []
    paras = _split_into_paragraphs(text)
    for i, par in enumerate(paras, start=1):
        par = par.strip()
        if not par:
            continue
        paragraph_id = f"{doc_id}.par{i}"
        chunk_id = _make_chunk_id(doc_id, paragraph_id, par)
        chunks.append(
            RagChunk(
                chunk_id=chunk_id,
                source_title=source_title,
                source_url=source_url,
                doc_id=doc_id,
                doc_type="url",
                page=None,
                heading_path="",
                paragraph_id=paragraph_id,
                text=par,
                text_sha256=_sha256(par),
            )
        )
    return chunks


# -----------------------------
# Storage helpers
# -----------------------------


def _save_chunks_jsonl(path: Path, chunks: List[RagChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(_chunk_to_dict(c), ensure_ascii=False) + "\n")


def _load_chunks_jsonl(path: Path) -> List[RagChunk]:
    chunks: List[RagChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(_dict_to_chunk(d))
    return chunks


def _save_tfidf(path: Path, vectorizer: TfidfVectorizer, matrix: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)


def _load_tfidf(path: Path) -> Tuple[TfidfVectorizer, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return obj["vectorizer"], obj["matrix"]


def _chunk_to_dict(c: RagChunk) -> Dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "source_title": c.source_title,
        "source_url": c.source_url,
        "doc_id": c.doc_id,
        "doc_type": c.doc_type,
        "page": c.page,
        "heading_path": c.heading_path,
        "paragraph_id": c.paragraph_id,
        "text": c.text,
        "text_sha256": c.text_sha256,
    }


def _dict_to_chunk(d: Dict[str, Any]) -> RagChunk:
    return RagChunk(
        chunk_id=str(d.get("chunk_id") or ""),
        source_title=str(d.get("source_title") or ""),
        source_url=str(d.get("source_url") or ""),
        doc_id=str(d.get("doc_id") or ""),
        doc_type=str(d.get("doc_type") or ""),
        page=d.get("page"),
        heading_path=str(d.get("heading_path") or ""),
        paragraph_id=str(d.get("paragraph_id") or ""),
        text=str(d.get("text") or ""),
        text_sha256=str(d.get("text_sha256") or ""),
    )


# -----------------------------
# Chunking / text utilities
# -----------------------------


def _normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Prefer splitting on blank lines; fallback to sentence-ish split for long blocks.
    """
    blocks = re.split(r"\n\s*\n+", text)
    paras: List[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # Further split overly long blocks
        paras.extend(_split_large_block(b, max_chars=1200))
    return [p.strip() for p in paras if p.strip()]


def _split_large_block(block: str, max_chars: int = 1200) -> List[str]:
    if len(block) <= max_chars:
        return [block]

    # Sentence-ish split
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", block)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur).strip())
            cur = [s]
            cur_len = len(s)
        else:
            cur.append(s)
            cur_len += len(s) + 1

    if cur:
        chunks.append(" ".join(cur).strip())

    # If still too large (e.g., no punctuation), hard split
    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                part = c[i : i + max_chars].strip()
                if part:
                    final.append(part)
    return final


def _strip_md_heading(s: str) -> str:
    s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s).strip()
    s = re.sub(r"\s+#+\s*$", "", s).strip()
    return s


def _strip_html(html: str) -> str:
    """
    Minimal HTML stripping without extra dependencies.
    Good enough for prototype RAG; prefer local PDFs for accurate citations.
    """
    # Remove scripts/styles
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    # Replace breaks with newlines
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p\s*>", "\n\n", html)
    # Strip tags
    text = re.sub(r"(?s)<.*?>", " ", html)
    # Decode basic entities
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return text


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "doc"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk_id(doc_id: str, paragraph_id: str, text: str) -> str:
    # Stable enough: doc + paragraph + text hash prefix
    h = _sha256(text)[:12]
    return f"{doc_id}:{paragraph_id}:{h}"


def _load_url_sources_spec(path: Path) -> List[Dict[str, Any]]:
    """
    Optional URL ingestion spec file.

    data/sources.json example:
    [
      {"source_title":"BoE PRA Rulebook page","source_url":"https://...","doc_id":"pra-rulebook"},
      {"source_title":"EBA ITS page","source_url":"https://...","doc_id":"eba-its"}
    ]
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            out = []
            for x in data:
                if isinstance(x, dict) and x.get("source_url"):
                    out.append(x)
            return out
    except Exception:
        return []
    return []


# -----------------------------
# Convenience: query builder
# -----------------------------


def build_default_rag_query(
    user_question: str,
    template_codes: List[str],
) -> str:
    """
    Keeps retrieval simple and focused.
    """
    templates = " ".join(template_codes) if template_codes else ""
    q = f"{user_question} {templates} own funds CET1 Tier 1 Tier 2 capital ratio TREA COREP instructions"
    return re.sub(r"\s+", " ", q).strip()
