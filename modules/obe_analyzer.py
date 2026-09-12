"""
obe_analyzer.py
---------------
The heart of OBE-AuditAI's AI workflow. Each stage:
  1. Retrieves relevant evidence from the FAISS store (RAG)
  2. Sends that evidence + a stage-specific instruction to the LLM
  3. Parses the LLM's structured JSON response
  4. Returns (parsed_result, evidence_used) so the UI can show both the
     analysis AND the exact source evidence behind it (RAG transparency)

The LLM is always instructed to ground its answers in the retrieved
evidence and to explicitly flag anything not supported by evidence,
rather than inventing facts.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Tuple, Optional

from modules.vector_store import FaissVectorStore
from modules.retrieval import gather_stage_evidence, format_evidence_for_prompt, Evidence
from modules.llm_provider import generate, LLMError

BASE_SYSTEM_PROMPT = (
    "You are an expert Outcome-Based Education (OBE) auditor and curriculum "
    "quality-assurance specialist. You analyze retrieved evidence from real "
    "course documents (course outlines, CLOs, PLOs, assessment plans, OBE "
    "guidelines). You must: "
    "1) Base every finding strictly on the provided evidence. "
    "2) If evidence is insufficient for a claim, say so explicitly instead "
    "of inventing information. "
    "3) Always respond with valid JSON ONLY, matching the requested schema, "
    "with no markdown code fences, no preamble, and no explanation text "
    "outside the JSON object."
)


def _extract_balanced_json(text: str) -> Optional[str]:
    """Find the first complete, brace-balanced {...} object in text.

    More robust than a greedy regex when a reasoning model has leaked
    chain-of-thought text before/after the JSON (which happens
    intermittently with Groq's gpt-oss models) — this walks the string
    tracking nesting depth and string/escape state, so it isn't confused
    by braces that happen to appear inside the leaked commentary, and it
    correctly reports "no complete object" when the JSON was truncated
    mid-object rather than silently grabbing a broken fragment.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # unbalanced — likely truncated mid-object


def _safe_json_parse(raw_text: str) -> Dict[str, Any]:
    """Robustly parse JSON out of an LLM response, tolerating stray code
    fences, leaked reasoning/chain-of-thought text before or after the
    JSON object, or minor leading/trailing commentary."""
    text = raw_text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text.strip()).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate = _extract_balanced_json(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise LLMError(
        "The AI model returned a response that could not be parsed as JSON. "
        "This can happen occasionally with some models — please try running "
        "the audit again, or switch to a different model."
    )


def _generate_json(provider: str, api_key: str, model: str, system_prompt: str,
                    user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """generate() + _safe_json_parse(), with one automatic retry on parse
    failure. Reasoning models (e.g. Groq's gpt-oss-120b/20b) intermittently
    leak chain-of-thought into the answer or run out of token budget before
    finishing the JSON — a single retry with a sharper instruction resolves
    most of those cases without surfacing an error to the user."""
    raw = generate(provider, api_key, model, system_prompt, user_prompt,
                    temperature=temperature, max_tokens=max_tokens)
    try:
        return _safe_json_parse(raw)
    except LLMError:
        retry_prompt = (
            user_prompt
            + "\n\nIMPORTANT: Your previous response could not be parsed as JSON. "
              "Respond with ONLY a single valid JSON object matching the schema above — "
              "no reasoning, no commentary, no markdown code fences, nothing before or "
              "after the JSON."
        )
        raw_retry = generate(provider, api_key, model, system_prompt, retry_prompt,
                              temperature=min(temperature, 0.1), max_tokens=max_tokens)
        return _safe_json_parse(raw_retry)


def _run_stage(provider: str, api_key: str, model: str, store: FaissVectorStore,
               stage_key: str, instruction: str, schema_hint: str,
               top_k: int = 5) -> Tuple[Dict[str, Any], List[Evidence]]:
    evidence = gather_stage_evidence(store, stage_key, top_k=top_k)
    evidence_block = format_evidence_for_prompt(evidence)

    user_prompt = (
        f"TASK:\n{instruction}\n\n"
        f"RETRIEVED EVIDENCE FROM UPLOADED DOCUMENTS:\n{evidence_block}\n\n"
        f"REQUIRED JSON SCHEMA:\n{schema_hint}\n\n"
        "Respond with ONLY the JSON object."
    )

    parsed = _generate_json(provider, api_key, model, BASE_SYSTEM_PROMPT, user_prompt,
                             temperature=0.15, max_tokens=4000)
    return parsed, evidence


# --------------------------------------------------------------------------- #
# Stage 1 — Document Analyzer
# --------------------------------------------------------------------------- #
def stage1_document_overview(provider, api_key, model, store) -> Tuple[Dict, List[Evidence]]:
    instruction = (
        "Extract a structured overview of the course from the evidence: course title, "
        "course code, credit hours, course objectives, a list of CLOs found (verbatim or "
        "close paraphrase), a list of PLOs found, teaching activities, assessments found "
        "(with weights if stated), projects, exams, assignments, and any OBE/accreditation "
        "rules mentioned. Clearly list any of these fields that could NOT be found in the evidence."
    )
    schema = """{
  "course_title": "string or null",
  "course_code": "string or null",
  "credit_hours": "string or null",
  "course_objectives": ["string", "..."],
  "clos_found": ["string", "..."],
  "plos_found": ["string", "..."],
  "teaching_activities": ["string", "..."],
  "assessments_found": [{"name": "string", "weight_percent": "number or null"}],
  "obe_rules_mentioned": ["string", "..."],
  "missing_information": ["string", "..."]
}"""
    return _run_stage(provider, api_key, model, store, "document_overview", instruction, schema, top_k=6)


# --------------------------------------------------------------------------- #
# Stage 2 — CLO Auditor
# --------------------------------------------------------------------------- #
def stage2_clo_audit(provider, api_key, model, store) -> Tuple[Dict, List[Evidence]]:
    instruction = (
        "For every CLO you can find in the evidence, analyze: the CLO statement, its main "
        "action verb, whether it is measurable (true/false), a clarity score 1-5, a "
        "relevance score 1-5 (relevance to a university-level course), its Bloom's Taxonomy "
        "cognitive level (one of Remember, Understand, Apply, Analyze, Evaluate, Create), a "
        "plausible assessment method, strengths, and weaknesses. Flag CLOs that are vague or "
        "hard to measure. If no CLOs are found in evidence, return an empty list and say so."
    )
    schema = """{
  "clos": [
    {
      "clo_id": "string e.g. CLO1",
      "statement": "string",
      "action_verb": "string",
      "measurable": true,
      "clarity_score": 4,
      "relevance_score": 5,
      "bloom_level": "Apply",
      "potential_assessment_method": "string",
      "strengths": ["string"],
      "weaknesses": ["string"],
      "flagged_vague": false
    }
  ],
  "notes": "string"
}"""
    return _run_stage(provider, api_key, model, store, "clo_extraction", instruction, schema, top_k=6)


# --------------------------------------------------------------------------- #
# Stage 3 — CLO–PLO Alignment Analyzer
# --------------------------------------------------------------------------- #
def stage3_clo_plo_alignment(provider, api_key, model, store, clo_ids: List[str]) -> Tuple[Dict, List[Evidence]]:
    clo_hint = ", ".join(clo_ids) if clo_ids else "unknown — infer from evidence"
    instruction = (
        f"Build a CLO-to-PLO alignment matrix using ONLY mappings supported by the evidence "
        f"(explicit mapping tables, or explicit statements linking a CLO to a PLO). Known CLO "
        f"IDs so far: {clo_hint}. For each PLO identified, list which CLOs map to it and mark "
        "the mapping strength as 'strong' (explicitly stated), 'weak' (implied/partial), or "
        "'unjustified' (claimed in a document but with unclear evidence). Separately, list any "
        "CLOs that appear to have NO PLO mapping at all in the evidence."
    )
    schema = """{
  "plos": ["PLO1", "PLO2"],
  "mappings": [
    {"clo_id": "CLO1", "plo_id": "PLO1", "strength": "strong"}
  ],
  "clos_without_plo_mapping": ["CLO3"],
  "unjustified_mappings": ["CLO2->PLO4: no supporting evidence found"],
  "notes": "string"
}"""
    return _run_stage(provider, api_key, model, store, "clo_plo_mapping", instruction, schema, top_k=6)


# --------------------------------------------------------------------------- #
# Stage 4 — Assessment Alignment Analyzer
# --------------------------------------------------------------------------- #
def stage4_assessment_alignment(provider, api_key, model, store, clo_ids: List[str]) -> Tuple[Dict, List[Evidence]]:
    clo_hint = ", ".join(clo_ids) if clo_ids else "unknown — infer from evidence"
    instruction = (
        f"Using the evidence, analyze how assessments map to CLOs. Known CLO IDs so far: "
        f"{clo_hint}. Identify: CLOs with no assessment support, assessments with no clear "
        "CLO linkage, CLOs that appear over-assessed (covered by many assessments relative to "
        "others) or under-assessed (covered by very few or none), overall assessment coverage "
        "as a percentage of CLOs that have at least one linked assessment, and the assessment "
        "weight distribution if weights are stated. List clear alignment problems."
    )
    schema = """{
  "assessment_clo_matrix": [
    {"assessment_name": "string", "weight_percent": "number or null", "linked_clos": ["CLO1"]}
  ],
  "clos_without_assessment": ["CLO4"],
  "assessments_without_clo": ["string"],
  "over_assessed_clos": ["CLO1"],
  "under_assessed_clos": ["CLO4"],
  "coverage_percent": 75,
  "alignment_problems": ["string"],
  "notes": "string"
}"""
    return _run_stage(provider, api_key, model, store, "assessment_alignment", instruction, schema, top_k=6)


# --------------------------------------------------------------------------- #
# Stage 5 — Bloom's Taxonomy Analyzer
# --------------------------------------------------------------------------- #
def stage5_bloom_analysis(provider, api_key, model, store, clo_bloom_levels: List[Dict]) -> Tuple[Dict, List[Evidence]]:
    levels_hint = json.dumps(clo_bloom_levels) if clo_bloom_levels else "none extracted yet — infer from evidence"
    instruction = (
        "Using the evidence and the already-classified CLO Bloom levels provided below, produce "
        "an overall Bloom's Taxonomy distribution across the six levels (Remember, Understand, "
        "Apply, Analyze, Evaluate, Create) for both CLOs and, where evidence allows, assessments. "
        "Identify if there is excessive concentration at lower-order levels (Remember/Understand) "
        "and recommend improvements. Explicitly note that this classification is an AI-supported "
        f"interpretation, not an absolute academic judgment.\n\nCLO Bloom levels already classified: {levels_hint}"
    )
    schema = """{
  "clo_distribution_percent": {"Remember": 0, "Understand": 20, "Apply": 40, "Analyze": 20, "Evaluate": 10, "Create": 10},
  "assessment_distribution_percent": {"Remember": 0, "Understand": 10, "Apply": 40, "Analyze": 30, "Evaluate": 10, "Create": 10},
  "lower_order_concentration_flag": false,
  "recommendations": ["string"],
  "disclaimer": "Bloom's classification is an AI-supported interpretation, not an absolute academic judgment."
}"""
    return _run_stage(provider, api_key, model, store, "assessment_extraction", instruction, schema, top_k=5)


# --------------------------------------------------------------------------- #
# Gap Detection (aggregation stage — mostly rule + LLM synthesis)
# --------------------------------------------------------------------------- #
def detect_gaps(provider, api_key, model, store,
                 stage1: Dict, stage2: Dict, stage3: Dict, stage4: Dict, stage5: Dict) -> Tuple[Dict, List[Evidence]]:
    instruction = (
        "Given the structured findings below from prior OBE analysis stages, detect and "
        "prioritize concrete gaps and issues. Classify each issue's severity as one of: "
        "'Critical', 'High', 'Medium', 'Low'. Cover: missing CLOs, weak CLO statements, CLOs "
        "without assessment evidence, assessments without CLO support, weak CLO-PLO alignment, "
        "missing PLO support, excessive lower-order cognitive levels, documentation "
        "inconsistencies, and missing OBE evidence.\n\n"
        f"STAGE 1 (Document Overview): {json.dumps(stage1)[:3000]}\n\n"
        f"STAGE 2 (CLO Audit): {json.dumps(stage2)[:3000]}\n\n"
        f"STAGE 3 (CLO-PLO Alignment): {json.dumps(stage3)[:3000]}\n\n"
        f"STAGE 4 (Assessment Alignment): {json.dumps(stage4)[:3000]}\n\n"
        f"STAGE 5 (Bloom's Analysis): {json.dumps(stage5)[:2000]}\n"
    )
    schema = """{
  "issues": [
    {
      "title": "string",
      "severity": "Critical",
      "category": "string e.g. CLO Quality / Alignment / Assessment / Documentation",
      "description": "string",
      "affected_items": ["CLO3"]
    }
  ]
}"""
    evidence = gather_stage_evidence(store, "obe_rules", top_k=4)
    evidence_block = format_evidence_for_prompt(evidence)
    user_prompt = (
        f"TASK:\n{instruction}\n\nRELEVANT OBE POLICY EVIDENCE:\n{evidence_block}\n\n"
        f"REQUIRED JSON SCHEMA:\n{schema}\n\nRespond with ONLY the JSON object."
    )
    parsed = _generate_json(provider, api_key, model, BASE_SYSTEM_PROMPT, user_prompt,
                             temperature=0.2, max_tokens=3000)
    return parsed, evidence


# --------------------------------------------------------------------------- #
# Generative Recommendations
# --------------------------------------------------------------------------- #
def generate_recommendations(provider, api_key, model, store, gaps: Dict) -> Tuple[Dict, List[Evidence]]:
    issues = gaps.get("issues", [])
    instruction = (
        "For each of the following detected issues, generate a structured recommendation "
        "using the Problem -> Evidence -> Impact -> Recommendation format. Where the issue "
        "concerns a specific CLO, propose an improved, measurable rewording of that CLO using "
        "a strong, higher-order action verb where appropriate. Ground every recommendation in "
        "OBE principles and, where possible, the retrieved evidence. Do not present unsupported "
        "claims as facts — if evidence is thin, say so in the 'evidence' field.\n\n"
        f"ISSUES:\n{json.dumps(issues)[:4000]}"
    )
    schema = """{
  "recommendations": [
    {
      "problem": "string",
      "evidence": "string",
      "impact": "string",
      "recommendation": "string",
      "improved_clo_wording": "string or null",
      "severity": "Critical"
    }
  ]
}"""
    evidence = gather_stage_evidence(store, "clo_extraction", top_k=4)
    evidence_block = format_evidence_for_prompt(evidence)
    user_prompt = (
        f"TASK:\n{instruction}\n\nSUPPORTING EVIDENCE:\n{evidence_block}\n\n"
        f"REQUIRED JSON SCHEMA:\n{schema}\n\nRespond with ONLY the JSON object."
    )
    parsed = _generate_json(provider, api_key, model, BASE_SYSTEM_PROMPT, user_prompt,
                             temperature=0.3, max_tokens=4000)
    return parsed, evidence


# --------------------------------------------------------------------------- #
# Executive summary (short narrative synthesis)
# --------------------------------------------------------------------------- #
def generate_executive_summary(provider, api_key, model, stage1: Dict, score_result: Dict, gaps: Dict) -> str:
    instruction = (
        "Write a concise (120-180 words) executive summary of this OBE audit for a program "
        "coordinator, in plain prose (not JSON, not bullet points). Mention the overall score, "
        "the strongest area, the weakest area, and the single most important next step. Be "
        "specific but avoid inventing numbers not provided below."
    )
    user_prompt = (
        f"{instruction}\n\n"
        f"Course: {stage1.get('course_title', 'Unknown')}\n"
        f"Overall Score: {score_result.get('overall_score')}/100\n"
        f"Category Scores: {json.dumps(score_result.get('category_scores', {}))}\n"
        f"Top Issues: {json.dumps([i.get('title') for i in gaps.get('issues', [])[:5]])}\n"
    )
    try:
        return generate(provider, api_key, model,
                         "You are an OBE quality-assurance report writer. Respond with plain prose only.",
                         user_prompt, temperature=0.4, max_tokens=800)
    except LLMError as e:
        return f"(Executive summary unavailable: {e})"
