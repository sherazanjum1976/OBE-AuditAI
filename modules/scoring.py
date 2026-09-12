"""
scoring.py
----------
A transparent, rule-based scoring engine. The LLM is NEVER allowed to
invent the final numeric score — every sub-score below is computed here
from structured stage outputs using explicit, documented formulas, so
the final "Overall OBE Quality Score" is always explainable and
reproducible.

Weights (sum to 100%):
  CLO Quality                 20%
  CLO-PLO Alignment           20%
  Assessment Alignment        20%
  CLO Assessment Coverage     15%
  Bloom's Distribution        10%
  CLO Measurability           10%
  Documentation Completeness   5%
"""

from __future__ import annotations

from typing import Dict, Any, List

WEIGHTS = {
    "CLO Quality": 0.20,
    "CLO-PLO Alignment": 0.20,
    "Assessment Alignment": 0.20,
    "CLO Assessment Coverage": 0.15,
    "Bloom's Distribution": 0.10,
    "CLO Measurability": 0.10,
    "Documentation Completeness": 0.05,
}


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def score_clo_quality(stage2: Dict[str, Any]) -> Dict[str, Any]:
    clos = stage2.get("clos", []) or []
    if not clos:
        return {"score": 0.0, "explanation": "No CLOs were found in the evidence, so CLO quality cannot be assessed."}
    total = 0.0
    for clo in clos:
        clarity = clo.get("clarity_score", 3) or 3
        relevance = clo.get("relevance_score", 3) or 3
        avg = (clarity + relevance) / 2.0  # 1-5 scale
        pct = (avg / 5.0) * 100
        if clo.get("flagged_vague"):
            pct -= 15
        total += _clamp(pct)
    score = total / len(clos)
    n_flagged = sum(1 for c in clos if c.get("flagged_vague"))
    explanation = (
        f"Averaged clarity & relevance scores across {len(clos)} CLO(s); "
        f"{n_flagged} CLO(s) flagged as vague/hard-to-measure (penalized)."
    )
    return {"score": round(score, 1), "explanation": explanation}


def score_clo_plo_alignment(stage3: Dict[str, Any], total_clos: int) -> Dict[str, Any]:
    mappings = stage3.get("mappings", []) or []
    unjustified = stage3.get("unjustified_mappings", []) or []
    missing = stage3.get("clos_without_plo_mapping", []) or []

    if total_clos == 0:
        return {"score": 0.0, "explanation": "No CLOs available to evaluate alignment."}

    strong = sum(1 for m in mappings if m.get("strength") == "strong")
    weak = sum(1 for m in mappings if m.get("strength") == "weak")

    # Base: reward CLOs that have at least one strong mapping
    mapped_clo_ids = {m.get("clo_id") for m in mappings if m.get("strength") in ("strong", "weak")}
    coverage_ratio = len(mapped_clo_ids) / total_clos if total_clos else 0
    strength_bonus = (strong / max(1, len(mappings))) * 20 if mappings else 0

    score = coverage_ratio * 80 + strength_bonus
    score -= len(unjustified) * 5
    score -= len(missing) * 3
    score = _clamp(score)

    explanation = (
        f"{len(mapped_clo_ids)}/{total_clos} CLOs have some PLO mapping "
        f"({strong} strong, {weak} weak); {len(missing)} CLO(s) unmapped; "
        f"{len(unjustified)} mapping(s) flagged as potentially unjustified."
    )
    return {"score": round(score, 1), "explanation": explanation}


def score_assessment_alignment(stage4: Dict[str, Any]) -> Dict[str, Any]:
    problems = stage4.get("alignment_problems", []) or []
    over = stage4.get("over_assessed_clos", []) or []
    under = stage4.get("under_assessed_clos", []) or []
    no_clo_assessments = stage4.get("assessments_without_clo", []) or []

    score = 100.0
    score -= len(problems) * 6
    score -= len(over) * 4
    score -= len(under) * 5
    score -= len(no_clo_assessments) * 5
    score = _clamp(score)

    explanation = (
        f"Deducted for {len(problems)} noted alignment problem(s), {len(over)} over-assessed "
        f"CLO(s), {len(under)} under-assessed CLO(s), and {len(no_clo_assessments)} "
        f"assessment(s) without clear CLO linkage."
    )
    return {"score": round(score, 1), "explanation": explanation}


def score_clo_coverage(stage4: Dict[str, Any]) -> Dict[str, Any]:
    coverage = stage4.get("coverage_percent")
    if coverage is None:
        return {"score": 50.0, "explanation": "Coverage percentage not determinable from evidence; using neutral default."}
    coverage = _clamp(float(coverage))
    return {"score": round(coverage, 1), "explanation": f"{coverage:.0f}% of CLOs have at least one linked assessment."}


def score_bloom_distribution(stage5: Dict[str, Any]) -> Dict[str, Any]:
    dist = stage5.get("clo_distribution_percent", {}) or {}
    if not dist:
        return {"score": 50.0, "explanation": "Bloom's distribution not determinable from evidence; using neutral default."}

    lower = dist.get("Remember", 0) + dist.get("Understand", 0)
    higher = dist.get("Analyze", 0) + dist.get("Evaluate", 0) + dist.get("Create", 0)
    mid = dist.get("Apply", 0)

    # Ideal-ish profile for many university courses: modest lower-order share,
    # solid Apply, and meaningful higher-order share. Penalize heavy skew to
    # lower-order levels; reward presence across mid/higher levels.
    score = 100.0
    if lower > 50:
        score -= (lower - 50) * 1.2
    score += min(higher, 30) * 0.5
    score += min(mid, 40) * 0.25
    score = _clamp(score)

    flag = stage5.get("lower_order_concentration_flag", False)
    explanation = (
        f"Lower-order (Remember+Understand) share: {lower:.0f}%, higher-order "
        f"(Analyze+Evaluate+Create) share: {higher:.0f}%. "
        + ("Flagged for excessive lower-order concentration." if flag else "No excessive lower-order concentration flagged.")
    )
    return {"score": round(score, 1), "explanation": explanation}


def score_clo_measurability(stage2: Dict[str, Any]) -> Dict[str, Any]:
    clos = stage2.get("clos", []) or []
    if not clos:
        return {"score": 0.0, "explanation": "No CLOs were found, so measurability cannot be assessed."}
    measurable = sum(1 for c in clos if c.get("measurable"))
    pct = (measurable / len(clos)) * 100
    return {"score": round(pct, 1), "explanation": f"{measurable}/{len(clos)} CLOs judged measurable based on their stated action verbs."}


def score_documentation_completeness(stage1: Dict[str, Any]) -> Dict[str, Any]:
    missing = stage1.get("missing_information", []) or []
    # Each missing core field costs 12 points, floor at 0
    score = _clamp(100 - len(missing) * 12)
    explanation = f"{len(missing)} expected item(s) of course documentation could not be found in the uploaded evidence."
    return {"score": round(score, 1), "explanation": explanation}


def compute_overall_score(stage1: Dict[str, Any], stage2: Dict[str, Any], stage3: Dict[str, Any],
                           stage4: Dict[str, Any], stage5: Dict[str, Any]) -> Dict[str, Any]:
    total_clos = len(stage2.get("clos", []) or [])

    components = {
        "CLO Quality": score_clo_quality(stage2),
        "CLO-PLO Alignment": score_clo_plo_alignment(stage3, total_clos),
        "Assessment Alignment": score_assessment_alignment(stage4),
        "CLO Assessment Coverage": score_clo_coverage(stage4),
        "Bloom's Distribution": score_bloom_distribution(stage5),
        "CLO Measurability": score_clo_measurability(stage2),
        "Documentation Completeness": score_documentation_completeness(stage1),
    }

    overall = 0.0
    for name, weight in WEIGHTS.items():
        overall += components[name]["score"] * weight

    return {
        "overall_score": round(_clamp(overall)),
        "category_scores": {name: components[name]["score"] for name in WEIGHTS},
        "category_explanations": {name: components[name]["explanation"] for name in WEIGHTS},
        "weights": WEIGHTS,
    }
