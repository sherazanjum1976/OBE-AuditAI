# 🎯 OBE-AuditAI — Hackathon Presentation Package

Target length: **4–5 minutes**, **9 slides**. Everything below — slide content, speaker notes,
AI voice-over script, timing, visual instructions, video assembly workflow, and the live demo
script — is ready to use directly.

---

## Slide-by-Slide Content & Voice-Over

### Slide 1 — The Problem
**On-screen title:** OBE-AuditAI
**On-screen body:**
- Course documents (CLOs, PLOs, assessments, OBE rules) are scattered across many PDFs
- Manually checking CLO–PLO–assessment alignment is slow and inconsistent
- Reviewers do this by hand, every course, every accreditation cycle

**Voice-over (≈25 sec):**
"Every accredited program depends on Outcome-Based Education — CLOs mapped to PLOs, backed by
real assessments. But that evidence usually lives across five or six separate PDFs: a course
outline, a PLO list, an assessment plan, an OBE policy document. Checking that everything lines
up is done by hand today — and it's slow, and it's inconsistent from one reviewer to the next."

---

### Slide 2 — Our Solution
**On-screen visual:** `Multiple Documents → AI/RAG → OBE Audit → Score → Recommendations`
**On-screen body:**
- OBE-AuditAI: an AI quality-assurance workflow, not a PDF chatbot
- Reads all your course documents together
- Produces a scored, evidence-backed OBE audit in minutes

**Voice-over (≈28 sec):**
"That's the problem OBE-AuditAI solves. It's not a chatbot you ask questions to — it's a
structured quality-assurance workflow. You upload every relevant document, and the system reads
them together, retrieves the evidence that actually matters, and turns it into a scored,
evidence-backed OBE audit in minutes instead of hours."

---

### Slide 3 — How It Works
**On-screen diagram:**
```text
Upload PDFs → Extract → Chunk → Embed → FAISS →
Retrieve Evidence → LLM Reasoning → OBE Rules → Score + Recommendations
```

**Voice-over (≈32 sec):**
"Under the hood, every PDF is text-extracted, cleaned, and split into small overlapping chunks
that keep their source page number attached. Those chunks are embedded and stored in a FAISS
vector index built fresh for that session. When the audit runs, the system retrieves only the
chunks relevant to each specific OBE question — like CLO–PLO mapping — and sends just that
evidence to the language model for reasoning."

---

### Slide 4 — AI / RAG Technology
**On-screen body (logos/labels):**
- PyMuPDF — PDF extraction
- Sentence-Transformers (all-MiniLM-L6-v2) — free, local embeddings
- FAISS — in-memory vector database
- Groq / Gemini — LLM reasoning (your own API key)
- Python + Streamlit — the application layer

**Voice-over (≈28 sec):**
"The stack is deliberately lightweight and free-tier friendly. PyMuPDF handles extraction,
Sentence-Transformers generates embeddings locally with no paid API needed, and FAISS gives us
a real vector database without any cloud dependency. For reasoning, you plug in your own Groq
or Gemini key — the whole thing runs on Python and Streamlit, so it deploys anywhere for free."

---

### Slide 5 — OBE Intelligence
**On-screen body:**
- CLO quality (clarity, measurability, action verbs)
- CLO–PLO alignment matrix
- Assessment alignment & coverage
- Bloom's Taxonomy distribution
- Documentation completeness & gaps

**Voice-over (≈30 sec):**
"This is where the domain intelligence lives. The system audits every CLO for clarity and
measurability, builds a CLO-to-PLO alignment matrix straight from your documents' evidence,
checks whether every learning outcome is actually assessed, and classifies the course's
cognitive demand using Bloom's Taxonomy — flagging if too much of the course sits at the
'remember and understand' level."

---

### Slide 6 — Live Demonstration
**On-screen body:** Upload → Build Knowledge Base → Run Audit → View Results

**Voice-over (≈55–60 sec, spoken over the live demo):**
"Let's see it live. I'll upload our course outline, PLO list, and assessment plan... the system
tells me how many pages and chunks it processed... one click to build the knowledge base — text
extraction, chunking, embedding, and FAISS indexing all happen right here. Now I run the audit.
In the background it's retrieving evidence and reasoning through five OBE analysis stages. And
here's the result: an overall OBE quality score, broken down by category, with a full CLO–PLO
matrix and an assessment alignment view — all generated from the documents I just uploaded."

---

### Slide 7 — AI Recommendations
**On-screen body (example):**
- **Detected Gap:** CLO3 is difficult to measure
- **Evidence:** Page 2, Course_Outline.pdf
- **AI Recommendation:** Rewrite CLO3 using a measurable, higher-order action verb

**Voice-over (≈28 sec):**
"Every recommendation follows the same structure: the problem, the exact evidence it came from,
the impact if it's left unaddressed, and a concrete recommendation — including, where useful, a
rewritten CLO with a stronger, measurable action verb. Nothing is invented; everything traces
back to a real page in a real uploaded document."

---

### Slide 8 — Impact
**On-screen body:**
- University teachers & program coordinators
- OBE coordinators & curriculum developers
- Accreditation teams & QA offices
- **Impact:** Less manual auditing time, better OBE alignment, continuous course improvement

**Voice-over (≈25 sec):**
"This is built for the people who currently do this by hand — teachers, program coordinators,
OBE and quality-assurance offices, accreditation teams. The impact is straightforward: far less
manual auditing effort, stronger CLO–PLO–assessment alignment, and a repeatable way to keep
improving a course every semester."

---

### Slide 9 — Future Vision
**On-screen body:**
- Program-level OBE auditing
- Automatic accreditation report generation
- Historical course comparison
- Student performance integration & CLO attainment prediction
- University-wide quality dashboard

**Closing line (on screen):** "OBE-AuditAI — Turning Academic Documents into Actionable Quality
Insights."

**Voice-over (≈20 sec):**
"Looking ahead, this scales from a single course to an entire program — automatic accreditation
reports, historical comparisons across offerings, and eventually tying in real student
performance data to predict CLO attainment. OBE-AuditAI: turning academic documents into
actionable quality insights."

---

## Timing Table

| Slide | Duration | Content |
|---|---:|---|
| 1 | 25 sec | Problem |
| 2 | 30 sec | Solution |
| 3 | 35 sec | Workflow |
| 4 | 30 sec | Technology |
| 5 | 35 sec | OBE intelligence |
| 6 | 60 sec | Live demo |
| 7 | 30 sec | Recommendations |
| 8 | 25 sec | Impact |
| 9 | 20 sec | Future |
| **Total** | **≈4 min 50 sec** | |

---

## Visual / Animation Instructions Per Slide

| Slide | On-screen | Animation/Transition | Screen Capture Needed? | Emphasize |
|---|---|---|---|---|
| 1 | Title + 3 problem bullets, muted red/orange accent | Fade in bullets one at a time | No | "scattered across many PDFs" |
| 2 | Horizontal flow diagram (5 boxes with arrows) | Boxes light up left→right in sequence | No | "not a PDF chatbot" |
| 3 | Vertical pipeline diagram (9 steps) | Steps highlight top→bottom as narrated | No | "keep their source page number" |
| 4 | Tech logos/labels in a grid | Simple fade-in grid | No | "free-tier friendly" |
| 5 | 5 icons (CLO, PLO matrix, assessment, Bloom, docs) | Icons pop in with score-card style | Optional: matrix screenshot | "Bloom's Taxonomy" |
| 6 | **Live screen recording** of the actual app | Screen recording; cursor highlight on each step | **Yes — required** | Score dashboard + evidence expander |
| 7 | Problem→Evidence→Impact→Recommendation card | Card slides in top-to-bottom, line by line | Optional: recommendation screenshot | "traces back to a real page" |
| 8 | Persona icons + 2-line impact statement | Simple fade | No | Bold the impact statement |
| 9 | Bullet roadmap + closing tagline | Bullets fade in; tagline scales up at the end | No | Closing tagline |

---

## AI Video Creation Package

### A. Slide Content
See the slide-by-slide section above — use it directly as your slide deck text (or convert
into a `.pptx` using the same content).

### B. Voice-Over
Use the voice-over scripts above verbatim, or paste each into an AI voice-over/TTS tool.

### C. Timing
See the timing table above (target total: **4 min 50 sec**, fits the 4–5 minute requirement).

### D. Visual Instructions
See the table above.

### E. Video Assembly Workflow

```text
Generate Slides (Google Slides / PowerPoint / Canva — free tiers)
       ↓
Generate AI Voice-over (see tool options below)
       ↓
Capture Application Demo (OBS Studio — free, open source screen recorder)
       ↓
Synchronize Slide + Voice (drop narration under each slide in your editor's timeline)
       ↓
Add Transitions (simple fade/cut — avoid distracting effects)
       ↓
Add Background Music (optional, low volume — YouTube Audio Library, royalty-free)
       ↓
Add Subtitles (auto-caption feature in CapCut / Descript free tier, or manual .srt)
       ↓
Export MP4 (1080p, H.264)
```

**Free/low-cost tool recommendations:**

| Need | Free-tier option | Note |
|---|---|---|
| Slides | Google Slides, Canva (free tier) | Export as images or record directly |
| Screen recording | OBS Studio (fully free) | Record the live Streamlit demo at 1080p |
| AI voice-over | Any TTS tool's free tier (e.g. a browser-based TTS extension) | Free tiers usually cap total minutes/characters — this script fits comfortably under most free character limits per slide |
| Video editing | CapCut (free), DaVinci Resolve (free) | Both support auto-captions and free export |
| Royalty-free music | YouTube Audio Library | No subscription needed |

If a chosen AI voice-over tool's free tier runs out of characters/minutes, the simple fallback
is: record the scripts above with your own voice using any phone or laptop microphone in a
quiet room — the scripts are already written in natural spoken language, so no extra editing
of phrasing is needed.

### F. Final Export Instructions
- Export at 1080p (1920×1080), H.264 MP4, 30fps.
- Keep the file under typical hackathon submission size limits (compress with HandBrake, free,
  if needed).
- Double check total runtime lands between 4:00 and 5:00.

---

## Live Hackathon Demo Script (4–5 minutes)

```text
0:00–0:30   Problem
            "Course documents are scattered, manual OBE auditing is slow and inconsistent."

0:30–1:00   Solution
            "OBE-AuditAI: upload everything, get a scored, evidence-backed audit in minutes."

1:00–1:30   Architecture
            Point to the workflow diagram in the app: Documents → RAG → LLM → Score → Recs.

1:30–3:30   Live application demonstration
            - Select provider (Groq/Gemini) and paste API key (model is pre-selected — one fixed, curated model per provider)
            - Upload 3–4 sample course/OBE PDFs
            - Click "Build OBE Knowledge Base" — narrate extraction/chunking/embedding/FAISS
            - Click "Run OBE Audit" — narrate the 5 stages running
            - Land on the Executive Dashboard: call out the overall score and category bars

3:30–4:15   Results and recommendations
            - Open the CLO–PLO matrix tab
            - Open Gaps tab, point to a Critical/High issue
            - Open Recommendations tab, show one Problem→Evidence→Impact→Recommendation card
            - Open the Evidence tab briefly to prove it's grounded in real pages

4:15–5:00   Impact + future vision
            "This is for teachers, OBE coordinators, and accreditation teams — less manual
            effort, stronger alignment, and a clear path to program-level auditing next."
```

**Presenter tips:**
- Do not read code or explain module internals — evaluators care about the transformation:
  **Documents → Evidence → Intelligence → Score → Action.**
- Pre-load sample PDFs before presenting so the upload step is fast.
- If a live API call is slow on stage, have a previously-completed audit's results ready as a
  backup screenshot/tab.
