"""
OBE-AuditAI — Intelligent Outcome-Based Education Course Auditor
==================================================================
Main Streamlit application. See README.md for architecture details.

This file wires together the RAG + AI workflow:
  Documents -> Extraction -> Chunking -> Embeddings -> FAISS ->
  Retrieval -> LLM Reasoning -> OBE Rules -> Scoring -> Gaps ->
  Recommendations -> Report
"""

from __future__ import annotations

import json
import time
import pandas as pd
import streamlit as st

from modules import pdf_processor, embeddings, vector_store, obe_analyzer, scoring, report_generator
from modules.llm_provider import get_model_for_provider, LLMError

# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="OBE-AuditAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ----------------------------------------------------------------- */
    /* Base palette (kept in sync with .streamlit/config.toml)           */
    /* ----------------------------------------------------------------- */
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #101a30 100%);
        color: #e8ecf5;
    }
    section[data-testid="stSidebar"] {
        background: #0c1526;
        border-right: 1px solid #1e2c4a;
    }
    section[data-testid="stSidebar"] * {
        color: #e8ecf5;
    }
    h1, h2, h3, h4 {
        color: #f2f5fb !important;
    }
    p, li, span, label, .stMarkdown {
        color: #d7deed;
    }
    a, a:visited {
        color: #6fd3ff !important;
    }
    hr, [data-testid="stDivider"] {
        border-color: #223458 !important;
    }

    /* ----------------------------------------------------------------- */
    /* Captions / secondary text — must never fall back to a color close */
    /* to the dark background.                                          */
    /* ----------------------------------------------------------------- */
    [data-testid="stCaptionContainer"], .stCaption {
        color: #a9b8d6 !important;
    }

    /* ----------------------------------------------------------------- */
    /* Progress bars ("sliders") — sidebar upload progress AND main-pane */
    /* build/audit progress were using Streamlit's default light-theme   */
    /* track/fill, which was nearly invisible against the dark page.     */
    /* ----------------------------------------------------------------- */
    div[data-testid="stProgress"] > div {
        background-color: #16233f !important;
        border-radius: 999px;
        height: 12px;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #2f6fed, #37e6c1) !important;
        border-radius: 999px;
    }

    /* st.slider, if used anywhere (track + thumb + selected range) */
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
        background: #1b2740 !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #2f6fed, #37e6c1) !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 3px solid #37e6c1 !important;
        box-shadow: 0 0 0 3px rgba(55,230,193,0.25) !important;
    }

    /* ----------------------------------------------------------------- */
    /* Inputs, selects, and their (portal-rendered) popovers/dropdowns   */
    /* ----------------------------------------------------------------- */
    div[data-baseweb="select"] > div,
    input, textarea {
        background-color: #101c34 !important;
        color: #f2f5fb !important;
        border: 1px solid #2b3f6b !important;
    }
    div[data-baseweb="popover"], ul[data-baseweb="menu"], div[role="listbox"] {
        background-color: #101c34 !important;
        border: 1px solid #2b3f6b !important;
    }
    li[role="option"], div[role="option"] {
        color: #f2f5fb !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #1b2c50 !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #16233f !important;
        color: #f2f5fb !important;
        border: 1px solid #2b3f6b !important;
    }

    /* ----------------------------------------------------------------- */
    /* File uploader dropzone                                            */
    /* ----------------------------------------------------------------- */
    section[data-testid="stFileUploaderDropzone"] {
        background: #101c34 !important;
        border: 1.5px dashed #3a5590 !important;
        border-radius: 12px;
    }
    section[data-testid="stFileUploaderDropzone"] * {
        color: #dbe3f2 !important;
    }
    div[data-testid="stFileUploaderFile"] {
        background: #131f38 !important;
        color: #e8ecf5 !important;
        border-radius: 8px;
    }

    /* ----------------------------------------------------------------- */
    /* Alerts (info / success / warning / error) — force high-contrast   */
    /* text regardless of Streamlit's default alert background.         */
    /* ----------------------------------------------------------------- */
    div[data-testid="stAlertContainer"] {
        background: #131f38 !important;
        border: 1px solid #2b3f6b !important;
        border-radius: 10px;
    }
    div[data-testid="stAlertContainer"] * {
        color: #f2f5fb !important;
    }

    /* ----------------------------------------------------------------- */
    /* Metrics, tabs, expanders                                          */
    /* ----------------------------------------------------------------- */
    div[data-testid="stMetric"] {
        background: #131f38;
        border: 1px solid #24365e;
        border-radius: 10px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricLabel"] * { color: #a9b8d6 !important; }
    div[data-testid="stMetricValue"] * { color: #f2f5fb !important; }

    button[data-baseweb="tab"] { color: #a9b8d6 !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #59d0ff !important;
        border-bottom-color: #59d0ff !important;
    }

    details, div[data-testid="stExpander"] {
        background: #101c34 !important;
        border: 1px solid #24365e !important;
        border-radius: 10px;
    }
    details summary, div[data-testid="stExpander"] summary * {
        color: #f2f5fb !important;
    }

    /* ----------------------------------------------------------------- */
    /* Cards / hero / badges                                             */
    /* ----------------------------------------------------------------- */
    .obe-card {
        background: linear-gradient(145deg, #131f38, #0f1a30);
        border: 1px solid #24365e;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        color: #e8ecf5;
    }
    .obe-card, .obe-card * {
        color: #e8ecf5;
    }
    .obe-card h4 { color: #f2f5fb !important; margin-top: 0; }
    .obe-card b, .obe-card strong { color: #ffffff !important; }
    .obe-upload-guide {
        border-left: 4px solid #37e6c1;
    }
    .obe-score-hero {
        text-align: center;
        padding: 28px 10px;
        border-radius: 18px;
        background: radial-gradient(circle at top, #16264a 0%, #0c1526 80%);
        border: 1px solid #2b3f6b;
        margin-bottom: 20px;
    }
    .obe-score-number {
        font-size: 64px;
        font-weight: 800;
        background: linear-gradient(90deg, #59d0ff, #8b7bff, #37e6c1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-critical { background:#4a1620; color:#ff9baa; border:1px solid #6b2233;}
    .badge-high { background:#4a2e14; color:#ffc37a; border:1px solid #6b4420;}
    .badge-medium { background:#4a4414; color:#f7ea86; border:1px solid #6b6220;}
    .badge-low { background:#123d2c; color:#8bf0c0; border:1px solid #1f5c41;}

    .stButton>button {
        background: linear-gradient(90deg, #2f6fed, #7b5cff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }
    .stButton>button:disabled {
        background: #1b2740 !important;
        color: #6b7ba0 !important;
    }
    .stDownloadButton>button {
        background: #131f38;
        color: #f2f5fb;
        border: 1px solid #2b3f6b;
        border-radius: 10px;
        font-weight: 600;
    }

    .workflow-step {
        text-align:center;
        padding: 10px 4px;
        border-radius: 10px;
        background: #101c34;
        border: 1px solid #223458;
        font-size: 13px;
        color: #e8ecf5 !important;
    }

    /* Dataframes render in an iframe on most Streamlit versions and pick */
    /* up the .streamlit/config.toml theme automatically; this just keeps */
    /* the surrounding chrome consistent.                                */
    div[data-testid="stDataFrame"] {
        border: 1px solid #24365e;
        border-radius: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
def init_state():
    defaults = {
        "provider": "Groq",
        "api_key": "",
        "selected_model": None,
        "expected_docs": 3,
        "uploaded_files_meta": [],
        "doc_results": None,
        "vector_store": None,
        "kb_built": False,
        "audit_results": None,
        "audit_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown("# 🎯 OBE-AuditAI")
st.markdown("#### AI-Powered Outcome-Based Education Course Auditor")
st.markdown(
    "> Upload your course and OBE documents. OBE-AuditAI retrieves relevant evidence, "
    "audits CLO–PLO–assessment alignment, calculates an explainable quality score, "
    "detects gaps, and generates actionable recommendations."
)

with st.expander("🔎 How the AI Workflow Works (click to expand)", expanded=False):
    steps = ["📄 Documents", "🔤 Extraction", "✂️ Chunking", "🧠 Embeddings", "🗄️ FAISS",
              "🔎 Retrieval", "🤖 LLM Reasoning", "🎓 OBE Rules", "📊 Scoring", "💡 Recommendations"]
    cols = st.columns(len(steps))
    for c, s in zip(cols, steps):
        c.markdown(f"<div class='workflow-step'>{s}</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class='obe-card obe-upload-guide'>
<h4>📂 What should you upload?</h4>
<p><b>Accepted format:</b> PDF only. Upload the actual course/program documents you want audited — not summaries or screenshots.</p>
<p><b>What kind of documents work best</b> (upload whichever of these you have; you don't need all of them, but more coverage = a more complete audit):</p>
<ul>
<li>🧾 <b>Course outline / syllabus</b> — course title, code, credit hours, objectives, CLOs</li>
<li>🎯 <b>CLO–PLO document</b> — Course Learning Outcomes and Program Learning Outcomes, and any mapping between them</li>
<li>📋 <b>Assessment plan / rubric</b> — assignments, quizzes, exams, projects and their weightings</li>
<li>🎓 <b>OBE / accreditation guidelines</b> — the policy document your program is being audited against</li>
<li>📝 <b>Exam or question papers</b> (optional) — helps validate Bloom's Taxonomy coverage</li>
</ul>
<p><b>Uploading more than one file?</b> Each PDF is read and chunked <b>one file at a time, in the order you upload them</b> — this is quick sequential text extraction, not a bottleneck. Once every file has been processed, all of their chunks are combined into <b>one unified knowledge base</b> (a single FAISS index). From that point on, every analysis stage searches <b>across all your uploaded documents together</b> — e.g. it can cross-reference a CLO from your course outline against your OBE guidelines and your assessment plan in the same step, rather than analyzing each document in isolation.</p>
<p>👉 Upload your files and configure Step 1 in the sidebar on the left, then continue with Steps 2–4 below.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------------------------- #
# SIDEBAR — Step 1: Configure AI
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚙️ Step 1 — Configure AI")

    provider = st.selectbox("LLM Provider", ["Groq", "Gemini"], key="provider")

    api_key = st.text_input(
        f"Enter {provider} API Key",
        type="password",
        key="api_key",
        help="Your key is used only for this session and is never stored.",
    )
    st.caption("🔐 Your API key is used only for this session and is not stored by OBE-AuditAI.")

    # A single, fixed, curated text model is used per provider (no picker).
    # This avoids two real issues seen with an open model dropdown: some
    # provider-hosted "models" are non-chat (audio/TTS) models that reject
    # text requests, and many chat models have free-tier quotas too tight
    # for the ~8 sequential LLM calls a full audit makes.
    fixed_model = get_model_for_provider(provider)
    st.session_state["selected_model"] = fixed_model
    st.caption(f"🤖 Model: **{fixed_model}**")
    if not api_key:
        st.info("Enter an API key above to run the audit.")

    st.divider()
    st.markdown("## 📥 Step 2 — Upload Knowledge")
    st.caption("📖 See the main panel for what to upload and how multiple files are handled.")
    expected_docs = st.number_input("How many documents will you upload?", min_value=1, max_value=20,
                                     value=st.session_state["expected_docs"], step=1, key="expected_docs")
    uploaded_files = st.file_uploader(
        "Upload course & OBE PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")
        if len(uploaded_files) != expected_docs:
            st.warning(
                f"⚠️ You specified {expected_docs} expected document(s) but uploaded "
                f"{len(uploaded_files)}. Proceeding is fine, but double-check nothing was missed."
            )
        for f in uploaded_files:
            st.caption(f"📄 {f.name}")

# --------------------------------------------------------------------------- #
# Step 3 — Build Knowledge Base
# --------------------------------------------------------------------------- #
st.markdown("## 🏗️ Step 3 — Build OBE Knowledge Base")

build_disabled = not uploaded_files if "uploaded_files" in dir() else True
build_col1, build_col2 = st.columns([1, 3])
with build_col1:
    build_clicked = st.button("🏗️ Build OBE Knowledge Base", disabled=not uploaded_files)

if build_clicked and uploaded_files:
    progress = st.progress(0, text="Reading PDFs...")
    files_payload = [{"name": f.name, "bytes": f.read()} for f in uploaded_files]
    doc_results = pdf_processor.process_multiple_pdfs(files_payload)
    progress.progress(35, text="Cleaning & chunking text...")

    all_chunks = []
    for dr in doc_results:
        all_chunks.extend(dr.chunks)

    if not all_chunks:
        st.error("No extractable text was found in any uploaded document. Please check your PDFs "
                  "(they may be scanned images without OCR).")
        st.session_state["kb_built"] = False
    else:
        progress.progress(60, text=f"Embedding {len(all_chunks)} chunks with Sentence Transformers...")
        texts = [c.text for c in all_chunks]
        vectors = embeddings.embed_texts(texts)

        progress.progress(85, text="Building FAISS index...")
        store = vector_store.build_vector_store(all_chunks, vectors)

        progress.progress(100, text="Knowledge base ready ✅")
        time.sleep(0.3)

        st.session_state["doc_results"] = doc_results
        st.session_state["vector_store"] = store
        st.session_state["kb_built"] = True
        st.session_state["audit_results"] = None  # invalidate old audit

if st.session_state.get("kb_built"):
    doc_results = st.session_state["doc_results"]
    store = st.session_state["vector_store"]

    st.markdown("### 📚 Knowledge Base Summary")
    summary_rows = []
    for dr in doc_results:
        status = "✅ Processed" if dr.has_text else "⚠️ No extractable text"
        summary_rows.append({
            "Filename": dr.filename,
            "Detected Type": dr.guessed_type,
            "Pages": dr.total_pages,
            "Chunks": len(dr.chunks),
            "Status": status,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    total_pages = sum(dr.total_pages for dr in doc_results)
    total_chunks = sum(len(dr.chunks) for dr in doc_results)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents Uploaded", len(doc_results))
    m2.metric("Documents Expected", expected_docs)
    m3.metric("Total Pages", total_pages)
    m4.metric("Total Chunks Indexed", total_chunks)

    for dr in doc_results:
        if not dr.has_text:
            st.warning(f"⚠️ '{dr.filename}' had little or no extractable text — it may be a scanned/image PDF.")

st.divider()

# --------------------------------------------------------------------------- #
# Step 4 — Run OBE Audit
# --------------------------------------------------------------------------- #
st.markdown("## 🚀 Step 4 — Run OBE Audit")

run_disabled = not (st.session_state.get("kb_built") and st.session_state.get("api_key") and st.session_state.get("selected_model"))
if run_disabled:
    st.info("Complete Steps 1–3 (API key, model, and knowledge base) before running the audit.")

run_clicked = st.button("🚀 Run OBE Audit", disabled=run_disabled)

def run_full_audit(provider: str, api_key: str, model: str, store):
    # Groq's free-tier keys enforce a per-minute request/token budget. This
    # audit makes 8 back-to-back LLM calls, so we space them out a little
    # even when each individual call succeeds — this avoids tripping the
    # per-minute cap partway through the run. llm_provider's own retry/
    # backoff logic still handles any 429 that slips through on top of this.
    INTER_STAGE_DELAY_S = 3 if provider == "Groq" else 0

    results = {}
    prog = st.progress(0, text="Stage 1/7 — Analyzing document overview...")
    try:
        stage1, ev1 = obe_analyzer.stage1_document_overview(provider, api_key, model, store)
        results["stage1"], results["evidence1"] = stage1, ev1
        prog.progress(14, text="Stage 2/7 — Auditing CLOs...")
        time.sleep(INTER_STAGE_DELAY_S)

        stage2, ev2 = obe_analyzer.stage2_clo_audit(provider, api_key, model, store)
        results["stage2"], results["evidence2"] = stage2, ev2
        clo_ids = [c.get("clo_id") for c in stage2.get("clos", []) if c.get("clo_id")]
        prog.progress(28, text="Stage 3/7 — Analyzing CLO–PLO alignment...")
        time.sleep(INTER_STAGE_DELAY_S)

        stage3, ev3 = obe_analyzer.stage3_clo_plo_alignment(provider, api_key, model, store, clo_ids)
        results["stage3"], results["evidence3"] = stage3, ev3
        prog.progress(42, text="Stage 4/7 — Analyzing assessment alignment...")
        time.sleep(INTER_STAGE_DELAY_S)

        stage4, ev4 = obe_analyzer.stage4_assessment_alignment(provider, api_key, model, store, clo_ids)
        results["stage4"], results["evidence4"] = stage4, ev4
        prog.progress(56, text="Stage 5/7 — Analyzing Bloom's Taxonomy distribution...")
        time.sleep(INTER_STAGE_DELAY_S)

        bloom_hint = [{"clo_id": c.get("clo_id"), "bloom_level": c.get("bloom_level")} for c in stage2.get("clos", [])]
        stage5, ev5 = obe_analyzer.stage5_bloom_analysis(provider, api_key, model, store, bloom_hint)
        results["stage5"], results["evidence5"] = stage5, ev5
        prog.progress(68, text="Computing transparent quantitative score...")

        score_result = scoring.compute_overall_score(stage1, stage2, stage3, stage4, stage5)
        results["score_result"] = score_result
        prog.progress(78, text="Stage 6/7 — Detecting gaps...")
        time.sleep(INTER_STAGE_DELAY_S)

        gaps, ev_gaps = obe_analyzer.detect_gaps(provider, api_key, model, store, stage1, stage2, stage3, stage4, stage5)
        results["gaps"], results["evidence_gaps"] = gaps, ev_gaps
        prog.progress(88, text="Stage 7/7 — Generating recommendations...")
        time.sleep(INTER_STAGE_DELAY_S)

        recommendations, ev_rec = obe_analyzer.generate_recommendations(provider, api_key, model, store, gaps)
        results["recommendations"], results["evidence_rec"] = recommendations, ev_rec
        prog.progress(96, text="Writing executive summary...")
        time.sleep(INTER_STAGE_DELAY_S)

        exec_summary = obe_analyzer.generate_executive_summary(provider, api_key, model, stage1, score_result, gaps)
        results["exec_summary"] = exec_summary

        prog.progress(100, text="Audit complete ✅")
        time.sleep(0.3)
        return results, None
    except LLMError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error during audit: {e}"


if run_clicked and not run_disabled:
    audit_results, error = run_full_audit(
        st.session_state["provider"], st.session_state["api_key"],
        st.session_state["selected_model"], st.session_state["vector_store"]
    )
    if error:
        st.error(f"❌ {error}")
    else:
        st.session_state["audit_results"] = audit_results
        st.success("Audit completed successfully!")

st.divider()

# --------------------------------------------------------------------------- #
# Step 5 — Results
# --------------------------------------------------------------------------- #
if st.session_state.get("audit_results"):
    r = st.session_state["audit_results"]
    stage1, stage2, stage3, stage4, stage5 = r["stage1"], r["stage2"], r["stage3"], r["stage4"], r["stage5"]
    score_result = r["score_result"]
    gaps = r["gaps"]
    recommendations = r["recommendations"]
    exec_summary = r["exec_summary"]

    st.markdown("## 📊 Executive Dashboard")
    st.markdown(
        f"<div class='obe-score-hero'><div>OBE QUALITY SCORE</div>"
        f"<div class='obe-score-number'>{score_result['overall_score']}/100</div></div>",
        unsafe_allow_html=True,
    )

    cat_scores = score_result["category_scores"]
    cols = st.columns(4)
    for i, (cat, sc) in enumerate(cat_scores.items()):
        with cols[i % 4]:
            st.markdown(f"**{cat}**")
            st.progress(min(int(sc), 100) / 100)
            st.caption(f"{sc:.0f}%")

    with st.expander("ℹ️ Why each score was assigned"):
        for cat, expl in score_result["category_explanations"].items():
            st.markdown(f"**{cat}** ({cat_scores[cat]:.0f}%): {expl}")

    st.markdown("### 📝 Executive Summary")
    st.markdown(f"<div class='obe-card'>{exec_summary}</div>", unsafe_allow_html=True)

    tabs = st.tabs([
        "🧩 CLO Analysis", "🔗 CLO–PLO Matrix", "📋 Assessment Alignment",
        "🌸 Bloom's Taxonomy", "🚩 Gaps", "💡 Recommendations", "📖 Evidence"
    ])

    # --- CLO Analysis ---
    with tabs[0]:
        clos = stage2.get("clos", [])
        if not clos:
            st.info("No CLOs were detected in the uploaded documents.")
        for clo in clos:
            flag = " 🔴 Flagged as vague" if clo.get("flagged_vague") else ""
            with st.expander(f"{clo.get('clo_id', 'CLO')} — {clo.get('statement', '')[:80]}{flag}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Clarity", f"{clo.get('clarity_score', '-')}/5")
                c2.metric("Relevance", f"{clo.get('relevance_score', '-')}/5")
                c3.metric("Bloom Level", clo.get("bloom_level", "-"))
                st.write(f"**Action verb:** {clo.get('action_verb', '-')}")
                st.write(f"**Measurable:** {'Yes' if clo.get('measurable') else 'No'}")
                st.write(f"**Potential assessment method:** {clo.get('potential_assessment_method', '-')}")
                if clo.get("strengths"):
                    st.write("**Strengths:** " + ", ".join(clo["strengths"]))
                if clo.get("weaknesses"):
                    st.write("**Weaknesses:** " + ", ".join(clo["weaknesses"]))

    # --- CLO-PLO Matrix ---
    with tabs[1]:
        plos = stage3.get("plos", [])
        mappings = stage3.get("mappings", [])
        if plos and mappings:
            clo_ids_sorted = sorted({m.get("clo_id") for m in mappings})
            matrix = pd.DataFrame(index=clo_ids_sorted, columns=plos)
            for m in mappings:
                symbol = "✓✓" if m.get("strength") == "strong" else "✓"
                matrix.loc[m.get("clo_id"), m.get("plo_id")] = symbol
            matrix = matrix.fillna("")
            st.dataframe(matrix, use_container_width=True)
            st.caption("✓✓ = strong mapping (explicit evidence)  |  ✓ = weak/implied mapping")
        else:
            st.info("No CLO–PLO mapping evidence was found in the uploaded documents.")

        if stage3.get("clos_without_plo_mapping"):
            st.warning("CLOs without PLO mapping: " + ", ".join(stage3["clos_without_plo_mapping"]))
        if stage3.get("unjustified_mappings"):
            st.warning("Potentially unjustified mappings: " + "; ".join(stage3["unjustified_mappings"]))

    # --- Assessment Alignment ---
    with tabs[2]:
        assess_matrix = stage4.get("assessment_clo_matrix", [])
        if assess_matrix:
            df = pd.DataFrame([
                {
                    "Assessment": a.get("assessment_name"),
                    "Weight %": a.get("weight_percent"),
                    "Linked CLOs": ", ".join(a.get("linked_clos", []) or []),
                } for a in assess_matrix
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No assessment-to-CLO mapping evidence found.")

        c1, c2 = st.columns(2)
        c1.metric("CLO Assessment Coverage", f"{stage4.get('coverage_percent', 'N/A')}%")
        if stage4.get("clos_without_assessment"):
            c2.warning("No assessment: " + ", ".join(stage4["clos_without_assessment"]))
        if stage4.get("over_assessed_clos"):
            st.write("⚠️ Over-assessed CLOs: " + ", ".join(stage4["over_assessed_clos"]))
        if stage4.get("under_assessed_clos"):
            st.write("⚠️ Under-assessed CLOs: " + ", ".join(stage4["under_assessed_clos"]))
        if stage4.get("alignment_problems"):
            st.write("**Alignment problems:**")
            for p in stage4["alignment_problems"]:
                st.write(f"- {p}")

    # --- Bloom's Taxonomy ---
    with tabs[3]:
        dist = stage5.get("clo_distribution_percent", {})
        if dist:
            bloom_df = pd.DataFrame({"Level": list(dist.keys()), "Percent": list(dist.values())})
            st.bar_chart(bloom_df.set_index("Level"))
        else:
            st.info("Bloom's distribution could not be determined.")
        default_disclaimer = "Bloom's classification is an AI-supported interpretation."
        st.caption(f"ℹ️ {stage5.get('disclaimer', default_disclaimer)}")
        if stage5.get("lower_order_concentration_flag"):
            st.warning("⚠️ Excessive concentration at lower-order cognitive levels detected.")
        if stage5.get("recommendations"):
            st.write("**Recommendations:**")
            for rec in stage5["recommendations"]:
                st.write(f"- {rec}")

    # --- Gaps ---
    with tabs[4]:
        severity_class = {"Critical": "badge-critical", "High": "badge-high",
                           "Medium": "badge-medium", "Low": "badge-low"}
        severity_icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
        issues = gaps.get("issues", [])
        if not issues:
            st.success("No significant gaps detected.")
        for issue in sorted(issues, key=lambda i: ["Critical", "High", "Medium", "Low"].index(i.get("severity", "Low"))
                             if i.get("severity") in ["Critical", "High", "Medium", "Low"] else 4):
            sev = issue.get("severity", "Low")
            badge = severity_class.get(sev, "badge-low")
            icon = severity_icon.get(sev, "⚪")
            st.markdown(
                f"<div class='obe-card'><span class='badge {badge}'>{icon} {sev}</span>"
                f"<b>{issue.get('title')}</b><br>{issue.get('description')}"
                f"<br><i>Category: {issue.get('category', '-')}</i></div>",
                unsafe_allow_html=True,
            )

    # --- Recommendations ---
    with tabs[5]:
        recs = recommendations.get("recommendations", [])
        if not recs:
            st.info("No specific recommendations were generated.")
        for rec in recs:
            with st.expander(f"💡 {rec.get('problem', 'Recommendation')}"):
                st.write(f"**Evidence:** {rec.get('evidence')}")
                st.write(f"**Impact:** {rec.get('impact')}")
                st.write(f"**Recommendation:** {rec.get('recommendation')}")
                if rec.get("improved_clo_wording"):
                    st.success(f"✏️ Suggested rewording: {rec['improved_clo_wording']}")

    # --- Evidence Transparency ---
    with tabs[6]:
        st.markdown("Evidence retrieved and used across the audit stages, for transparency.")
        evidence_map = {
            "Document Overview": r.get("evidence1", []),
            "CLO Audit": r.get("evidence2", []),
            "CLO–PLO Alignment": r.get("evidence3", []),
            "Assessment Alignment": r.get("evidence4", []),
            "Bloom's Analysis": r.get("evidence5", []),
            "Gap Detection": r.get("evidence_gaps", []),
            "Recommendations": r.get("evidence_rec", []),
        }
        for stage_name, ev_list in evidence_map.items():
            with st.expander(f"📖 Evidence used for: {stage_name} ({len(ev_list)} chunk(s))"):
                if not ev_list:
                    st.write("No evidence retrieved for this stage.")
                for ev in ev_list:
                    st.markdown(
                        f"**{ev.filename}** — page {ev.page_number} ({ev.doc_type}), "
                        f"relevance {ev.score:.2f}"
                    )
                    st.caption(ev.text[:500] + ("..." if len(ev.text) > 500 else ""))
                    st.markdown("---")

    # --- Downloads ---
    st.markdown("## 📥 Download Full Report")
    doc_summary = [
        {"filename": dr.filename, "doc_type": dr.guessed_type, "pages": dr.total_pages,
         "status": "Processed" if dr.has_text else "No extractable text"}
        for dr in st.session_state["doc_results"]
    ]
    course_meta = {
        "course_title": stage1.get("course_title"),
        "course_code": stage1.get("course_code"),
        "credit_hours": stage1.get("credit_hours"),
    }
    report_data = report_generator.build_report_data(
        course_meta, score_result, exec_summary, stage1, stage2, stage3, stage4, stage5,
        gaps, recommendations, doc_summary,
    )

    dl1, dl2, dl3, dl4 = st.columns(4)
    md_report = report_generator.to_markdown_report(report_data)
    txt_report = report_generator.to_txt_report(report_data)
    json_report = report_generator.to_json_report(report_data)
    pdf_report = report_generator.to_pdf_report(report_data)

    dl1.download_button("⬇️ Markdown Report", md_report, file_name="OBE_Audit_Report.md", mime="text/markdown")
    dl2.download_button("⬇️ TXT Report", txt_report, file_name="OBE_Audit_Report.txt", mime="text/plain")
    dl3.download_button("⬇️ JSON Report", json_report, file_name="OBE_Audit_Report.json", mime="application/json")
    if pdf_report:
        dl4.download_button("⬇️ PDF Report", pdf_report, file_name="OBE_Audit_Report.pdf", mime="application/pdf")
    else:
        dl4.caption("PDF generation unavailable this run.")

st.divider()
st.caption(
    "OBE-AuditAI — a RAG + Generative AI decision-support tool for OBE quality assurance. "
    "AI-generated findings should be reviewed by qualified academic staff before formal use."
)
