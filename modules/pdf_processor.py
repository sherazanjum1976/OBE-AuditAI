"""
pdf_processor.py
----------------
Handles PDF text extraction (PyMuPDF), cleaning, and chunking while
preserving document/page metadata for every chunk. This metadata is
what allows OBE-AuditAI to show real, citable evidence later in the
RAG pipeline (see modules/retrieval.py).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import pymupdf as fitz  # PyMuPDF (modern import name)
except ImportError:
    import fitz  # PyMuPDF (legacy import name, older versions)


@dataclass
class DocChunk:
    """A single chunk of text with full provenance metadata."""
    chunk_id: str
    text: str
    filename: str
    page_number: int
    doc_type: str = "Unknown"


@dataclass
class DocumentResult:
    """Result of processing a single uploaded PDF."""
    filename: str
    total_pages: int
    extracted_chars: int
    chunks: List[DocChunk] = field(default_factory=list)
    has_text: bool = True
    error: Optional[str] = None
    guessed_type: str = "Unknown"


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Light-weight cleaning: normalize whitespace, drop control chars,
    remove page-header/footer noise heuristically."""
    if not text:
        return ""
    # Normalize line breaks & excessive whitespace
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove common isolated page-number lines like "Page 3 of 10" or "3"
    text = re.sub(r"(?im)^\s*page\s+\d+\s*(of\s*\d+)?\s*$", "", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Lightweight document-type guesser (helps the LLM stages & UI labeling)
# --------------------------------------------------------------------------- #
_DOC_TYPE_KEYWORDS = {
    "Course Outline": ["course outline", "course description", "credit hours", "course syllabus"],
    "CLO/PLO Document": ["course learning outcome", "clo", "program learning outcome", "plo"],
    "Assessment Plan": ["assessment plan", "assessment weight", "grading policy", "rubric"],
    "OBE Guidelines": ["obe", "outcome-based education", "outcome based education", "accreditation"],
    "Examination Document": ["examination", "final exam", "midterm", "question paper"],
}


def guess_doc_type(full_text: str, filename: str) -> str:
    lower_text = (full_text or "").lower()
    lower_name = (filename or "").lower()
    scores: Dict[str, int] = {}
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            score += lower_text.count(kw) + (2 if kw in lower_name else 0)
        scores[doc_type] = score
    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] > 0 else "General OBE Document"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(
    text: str,
    filename: str,
    page_number: int,
    doc_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[DocChunk]:
    """Simple sliding-window character chunker that keeps metadata attached."""
    chunks: List[DocChunk] = []
    if not text:
        return chunks

    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # try to break on a sentence/paragraph boundary near the end
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + chunk_size * 0.5:
            end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                DocChunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    text=piece,
                    filename=filename,
                    page_number=page_number,
                    doc_type=doc_type,
                )
            )
        if end >= text_len:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


# --------------------------------------------------------------------------- #
# Main entry point: process a single uploaded PDF (bytes)
# --------------------------------------------------------------------------- #
def process_pdf(file_bytes: bytes, filename: str,
                 chunk_size: int = 1000, chunk_overlap: int = 150) -> DocumentResult:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return DocumentResult(
            filename=filename, total_pages=0, extracted_chars=0,
            chunks=[], has_text=False, error=f"Could not open PDF: {e}"
        )

    all_text_parts = []
    page_texts: List[str] = []
    for page in doc:
        raw = page.get_text("text") or ""
        page_texts.append(raw)
        all_text_parts.append(raw)

    total_pages = doc.page_count
    full_raw_text = "\n".join(all_text_parts)
    extracted_chars = len(full_raw_text.strip())
    has_text = extracted_chars > 30  # heuristic: near-empty => likely scanned/image PDF

    doc_type = guess_doc_type(full_raw_text, filename)

    all_chunks: List[DocChunk] = []
    if has_text:
        for i, raw_page_text in enumerate(page_texts, start=1):
            cleaned = clean_text(raw_page_text)
            if not cleaned:
                continue
            page_chunks = chunk_text(
                cleaned, filename=filename, page_number=i, doc_type=doc_type,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            all_chunks.extend(page_chunks)

    doc.close()

    return DocumentResult(
        filename=filename,
        total_pages=total_pages,
        extracted_chars=extracted_chars,
        chunks=all_chunks,
        has_text=has_text,
        error=None if has_text else "Little or no extractable text found (possibly a scanned/image PDF).",
        guessed_type=doc_type,
    )


def process_multiple_pdfs(files: List[Dict], chunk_size: int = 1000,
                           chunk_overlap: int = 150) -> List[DocumentResult]:
    """
    files: list of dicts like {"name": filename, "bytes": file_bytes}
    """
    results = []
    for f in files:
        result = process_pdf(f["bytes"], f["name"], chunk_size, chunk_overlap)
        results.append(result)
    return results
