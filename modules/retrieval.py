"""
retrieval.py
------------
Semantic retrieval helpers: given a query about a specific OBE audit
concern (e.g. "CLO to PLO mapping evidence"), retrieve the most relevant
chunks from the FAISS store and format them as evidence blocks that can
be (a) injected into an LLM prompt and (b) displayed to the user for
RAG transparency.
"""

from __future__ import annotations

from typing import List, Dict
from dataclasses import dataclass

from modules.vector_store import FaissVectorStore
from modules.embeddings import embed_query
from modules.pdf_processor import DocChunk


@dataclass
class Evidence:
    text: str
    filename: str
    page_number: int
    score: float
    doc_type: str


def retrieve(store: FaissVectorStore, query: str, top_k: int = 5) -> List[Evidence]:
    """Embed a query and retrieve the top_k most relevant chunks."""
    if store.size == 0:
        return []
    qvec = embed_query(query)
    results = store.search(qvec, top_k=top_k)
    evidence = [
        Evidence(
            text=chunk.text,
            filename=chunk.filename,
            page_number=chunk.page_number,
            score=score,
            doc_type=chunk.doc_type,
        )
        for chunk, score in results
    ]
    return evidence


def format_evidence_for_prompt(evidence: List[Evidence], max_chars_per_chunk: int = 900) -> str:
    """Format retrieved evidence into a numbered block suitable for LLM prompts,
    always including filename/page so the LLM can cite it and we can verify it."""
    if not evidence:
        return "No relevant evidence was retrieved from the uploaded documents."

    lines = []
    for i, ev in enumerate(evidence, start=1):
        snippet = ev.text[:max_chars_per_chunk]
        lines.append(
            f"[Evidence {i}] Source: {ev.filename} | Page: {ev.page_number} | "
            f"Type: {ev.doc_type} | Relevance: {ev.score:.2f}\n\"{snippet}\""
        )
    return "\n\n".join(lines)


# A fixed set of retrieval queries used to gather evidence for each OBE
# analysis stage. Keeping these explicit (rather than ad hoc) makes the
# RAG pipeline auditable and reproducible.
STAGE_QUERIES: Dict[str, List[str]] = {
    "document_overview": [
        "course title course code credit hours",
        "course objectives description",
        "teaching activities delivery methods",
    ],
    "clo_extraction": [
        "course learning outcomes CLO statements",
        "students will be able to",
    ],
    "plo_extraction": [
        "program learning outcomes PLO statements",
        "graduate attributes program outcomes",
    ],
    "clo_plo_mapping": [
        "CLO PLO mapping matrix alignment",
        "course learning outcomes mapped to program outcomes",
    ],
    "assessment_extraction": [
        "assessment plan quizzes assignments exams weight",
        "grading breakdown marks distribution",
        "project rubric evaluation criteria",
    ],
    "assessment_alignment": [
        "assessment mapped to CLO outcome",
        "exam question mapped to learning outcome",
    ],
    "obe_rules": [
        "OBE outcome based education requirements policy",
        "accreditation criteria guidelines",
    ],
}


def gather_stage_evidence(store: FaissVectorStore, stage_key: str, top_k: int = 4) -> List[Evidence]:
    """Run all queries configured for a stage and merge/deduplicate results."""
    queries = STAGE_QUERIES.get(stage_key, [])
    seen_ids = set()
    merged: List[Evidence] = []
    for q in queries:
        for ev in retrieve(store, q, top_k=top_k):
            key = (ev.filename, ev.page_number, ev.text[:50])
            if key not in seen_ids:
                seen_ids.add(key)
                merged.append(ev)
    # sort by relevance score desc, keep a reasonable cap
    merged.sort(key=lambda e: e.score, reverse=True)
    return merged[: max(top_k * 2, 6)]
