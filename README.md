# 🎯 OBE-AuditAI

### Intelligent Outcome-Based Education Course Auditor

**OBE-AuditAI** is a Generative AI + RAG (Retrieval-Augmented Generation) application that
audits university course documents for Outcome-Based Education (OBE) quality — CLO wording,
CLO–PLO alignment, assessment coverage, Bloom's Taxonomy balance, documentation completeness —
and produces an explainable score, prioritized gaps, and evidence-grounded recommendations.

> OBE-AuditAI is different from a conventional PDF chatbot because it transforms retrieved
> academic evidence into structured OBE auditing, scoring, alignment analysis, gap detection,
> and actionable recommendations — not free-form Q&A.

---

## 1. Problem Statement

Academic course documents — course outlines, CLOs, PLOs, assessment plans, syllabi, OBE and
accreditation guidelines — are typically spread across several separate PDFs. Manually cross-
checking whether every CLO maps to a PLO, whether every CLO is actually assessed, and whether
the course's cognitive demand (Bloom's Taxonomy) is well-balanced is slow, inconsistent between
reviewers, and easy to get wrong at scale (multiple courses, multiple programs, accreditation
cycles).

## 2. Motivation

OBE and accreditation frameworks (e.g. outcome-based accreditation bodies) require demonstrable,
evidence-based alignment between outcomes, teaching, and assessment. Reviewers currently do this
by hand. An AI system that can read the actual uploaded evidence, ground every finding in a
specific page/document, and produce a transparent, reproducible score turns a multi-hour manual
audit into a few minutes of AI-assisted review — while keeping a human reviewer firmly in the
loop for final judgment.

## 3. Proposed Solution

Upload every relevant course/OBE document. OBE-AuditAI builds a temporary, session-only
knowledge base (FAISS + Sentence Transformer embeddings), retrieves the evidence relevant to
each OBE audit question, and asks an LLM (Groq or Gemini, user's own API key) to reason over
that evidence stage by stage. A **separate, transparent rule-based scoring engine** — not the
LLM — computes the final 0–100 quality score, so the number is always explainable and
reproducible. The system finishes with prioritized gaps and Problem → Evidence → Impact →
Recommendation entries, all traceable back to a specific filename and page.

## 4. Key Features

- 📥 Multi-PDF upload with expected-vs-actual document count check
- 🔎 Real Retrieval-Augmented Generation (not "stuff everything into one prompt")
- 🤖 Provider choice: Groq or Gemini, with your own API key (never stored)
- 🧭 Dynamic, live model discovery per provider (no hardcoded obsolete models)
- 🧩 5-stage OBE analysis workflow (Document Overview → CLO Audit → CLO–PLO Alignment →
  Assessment Alignment → Bloom's Taxonomy)
- 📊 Transparent, rule-based 0–100 quality score across 7 weighted dimensions
- 🚩 Severity-classified gap detection (Critical / High / Medium / Low)
- 💡 Evidence-grounded, Problem→Evidence→Impact→Recommendation generative suggestions
- 📖 Full RAG evidence transparency — every finding links back to filename + page
- 📥 Downloadable reports: Markdown, TXT, JSON, and best-effort PDF
- 🎨 Modern dark/navy Streamlit UI, deployable free on Streamlit Cloud

## 5. RAG Architecture

```text
Multiple PDF Documents
        ↓
PDF Text Extraction (PyMuPDF)
        ↓
Text Cleaning
        ↓
Chunking (≈1000 chars, 150 overlap, metadata-preserving)
        ↓
Sentence Transformer Embeddings (all-MiniLM-L6-v2, local, free)
        ↓
FAISS Vector Database (in-memory, per-session)
        ↓
Semantic Retrieval (stage-specific queries)
        ↓
Relevant Evidence (filename + page + snippet)
        ↓
LLM Reasoning (Groq / Gemini, JSON-structured output)
        ↓
OBE Domain Rules (prompt-encoded auditing logic)
        ↓
Quantitative Scoring (rule-based, NOT LLM-invented)
        ↓
Gap Detection (severity-classified)
        ↓
Generative Recommendations (Problem→Evidence→Impact→Recommendation)
        ↓
Final OBE Audit Report (MD / TXT / JSON / PDF)
```

Every chunk keeps its source filename, page number, and a heuristically-guessed document type
(Course Outline, CLO/PLO Document, Assessment Plan, OBE Guidelines, Examination Document) so
retrieved evidence is always attributable.

## 6. AI Workflow (Analysis Stages)

| Stage | What it does |
|---|---|
| 1. Document Analyzer | Extracts course title, code, credit hours, objectives, CLOs, PLOs, teaching activities, assessments, and flags missing info |
| 2. CLO Auditor | Per-CLO: action verb, measurability, clarity, relevance, Bloom level, strengths/weaknesses |
| 3. CLO–PLO Alignment Analyzer | Builds a mapping matrix; flags strong / weak / missing / unjustified mappings |
| 4. Assessment Alignment Analyzer | CLO↔Assessment matrix; over-/under-assessed CLOs; coverage % |
| 5. Bloom's Taxonomy Analyzer | Distribution across 6 cognitive levels; flags lower-order concentration |
| Gap Detection | Aggregates all stages into severity-classified issues |
| Recommendations | Problem → Evidence → Impact → Recommendation, with improved CLO wording where useful |

## 7. OBE Scoring Methodology

The LLM never invents the final score. `modules/scoring.py` computes each dimension from
structured stage outputs using explicit, documented formulas:

| Dimension | Weight | How it's computed |
|---|---:|---|
| CLO Quality | 20% | Average of clarity & relevance scores per CLO, penalized for flagged-vague CLOs |
| CLO–PLO Alignment | 20% | Share of CLOs with a mapping, weighted by mapping strength, penalized for unjustified/missing mappings |
| Assessment Alignment | 20% | 100 minus penalties per alignment problem, over-/under-assessed CLO, unlinked assessment |
| CLO Assessment Coverage | 15% | % of CLOs with at least one linked assessment |
| Bloom's Distribution | 10% | Penalizes >50% concentration in Remember/Understand; rewards Apply & higher-order share |
| CLO Measurability | 10% | % of CLOs judged measurable from their action verb |
| Documentation Completeness | 5% | 100 minus 12 points per missing expected documentation item |

Overall score = weighted sum of the seven dimensions, rounded to the nearest whole number.

## 8. Technology Stack

- **Python 3.10+**
- **Streamlit** — UI
- **PyMuPDF (`fitz`/`pymupdf`)** — PDF text extraction
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — local, free embeddings
- **FAISS (`faiss-cpu`)** — in-memory vector search
- **Groq API / Google Gemini API** — LLM reasoning (bring your own key)
- **pandas** — tabular displays
- **reportlab** — best-effort PDF report export

## 9. Project Structure

```text
OBE-AuditAI/
├── app.py                     # Streamlit UI & orchestration
├── requirements.txt
├── README.md
├── .gitignore
└── modules/
    ├── __init__.py
    ├── pdf_processor.py       # Extraction, cleaning, chunking
    ├── embeddings.py          # Sentence-Transformer wrapper
    ├── vector_store.py        # FAISS index wrapper
    ├── retrieval.py           # Semantic retrieval + evidence formatting
    ├── llm_provider.py        # Groq/Gemini abstraction, model discovery
    ├── obe_analyzer.py        # 5-stage OBE AI workflow + gaps + recommendations
    ├── scoring.py             # Transparent rule-based scoring engine
    └── report_generator.py    # MD / TXT / JSON / PDF report builders
```

## 10. Installation & Local Execution

```bash
git clone https://github.com/<your-username>/OBE-AuditAI.git
cd OBE-AuditAI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 11. API Key Configuration

1. Get a free API key:
   - Groq: https://console.groq.com
   - Gemini: https://aistudio.google.com/app/apikey
2. Paste it into the sidebar "Enter API Key" field (masked input).
3. The key is held only in Streamlit's in-memory session state for that session — it is
   **never written to disk, never logged, and never included in any exported report.**
4. Closing the browser tab / restarting the app clears the key.

## 12. GitHub Deployment

1. Create a new GitHub repository (e.g. `OBE-AuditAI`).
2. Copy all project files into the repo folder.
3. `git add . && git commit -m "Initial commit: OBE-AuditAI"`
4. `git push origin main`
5. Verify the repository contains `app.py`, `requirements.txt`, `modules/`, `README.md`,
   `.gitignore` — and **no** API keys or `.env` files.

## 13. Streamlit Cloud Deployment

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app", select your `OBE-AuditAI` repository and branch.
3. Set the main file path to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
5. No secrets/config files are required — users supply their own API key at runtime through
   the UI.

The app has no dependency on local file paths, Colab paths, local FAISS files, or any paid
service — everything needed for deployment is `app.py` + `requirements.txt` + `modules/`.

## 14. Example Use Case

A program coordinator uploads: `Course_Outline.pdf`, `PLOs.pdf`, `Assessment_Plan.pdf`, and
`University_OBE_Policy.pdf`. OBE-AuditAI builds a knowledge base from all four, runs the audit,
and reports (for example) an overall score of 78/100, flags that one CLO uses a non-measurable
verb ("understand"), shows that CLO has no linked assessment, and recommends a specific
measurable rewording — with the exact page of the course outline the CLO came from.

## 15. Limitations

- Analysis quality depends on the completeness/clarity of the uploaded documents and on the
  chosen LLM's reasoning quality.
- Bloom's Taxonomy classification is an AI-supported interpretation, not an authoritative
  academic judgment — it should be reviewed by qualified faculty.
- Scanned/image-only PDFs without a text layer will not be usable unless OCR'd first (this is
  clearly flagged in the UI).
- Free-tier LLM APIs are subject to their own rate limits and quotas, which can slow or
  interrupt an audit on very large document sets.
- This is a decision-support tool, not a substitute for human academic and accreditation
  judgment.

## 16. Future Improvements

- Program-level (multi-course) OBE auditing and roll-up dashboards
- Automatic accreditation report generation in accreditation-body-specific templates
- Historical course-to-course comparison and trend tracking
- Integration with actual student performance/grade data for CLO attainment analysis
- Predictive CLO attainment modeling
- University-wide OBE quality dashboard across departments

---

*OBE-AuditAI was built as a hackathon MVP prioritizing a working end-to-end AI workflow over
unnecessary technical complexity.*
