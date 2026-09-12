"""
report_generator.py
--------------------
Builds the final OBE Audit Report in multiple downloadable formats:
Markdown, plain text, JSON, and (best-effort) PDF via reportlab.

PDF generation failures never crash the app — they are caught and the
UI simply omits the PDF download option for that run.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, List, Optional


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "N/A"


def build_report_data(course_meta: Dict[str, Any], score_result: Dict[str, Any],
                       exec_summary: str, stage1: Dict, stage2: Dict, stage3: Dict,
                       stage4: Dict, stage5: Dict, gaps: Dict, recommendations: Dict,
                       doc_summary: List[Dict]) -> Dict[str, Any]:
    """Assemble a single unified dict representing the whole audit — used as
    the source of truth for every export format."""
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "course_meta": course_meta,
        "documents_processed": doc_summary,
        "overall_score": score_result.get("overall_score"),
        "category_scores": score_result.get("category_scores"),
        "category_explanations": score_result.get("category_explanations"),
        "weights": score_result.get("weights"),
        "executive_summary": exec_summary,
        "document_overview": stage1,
        "clo_audit": stage2,
        "clo_plo_alignment": stage3,
        "assessment_alignment": stage4,
        "bloom_analysis": stage5,
        "gaps": gaps,
        "recommendations": recommendations,
    }


def to_json_report(report_data: Dict[str, Any]) -> str:
    return json.dumps(report_data, indent=2, ensure_ascii=False)


def to_markdown_report(report_data: Dict[str, Any]) -> str:
    md = []
    md.append("# 🎯 OBE-AuditAI — Course Audit Report\n")
    md.append(f"*Generated: {report_data.get('generated_at')}*\n")

    cm = report_data.get("course_meta", {})
    md.append(f"**Course:** {cm.get('course_title', 'N/A')}  ")
    md.append(f"**Code:** {cm.get('course_code', 'N/A')}  ")
    md.append(f"**Credit Hours:** {cm.get('credit_hours', 'N/A')}\n")

    md.append("## 📊 Overall OBE Quality Score\n")
    md.append(f"### {report_data.get('overall_score')} / 100\n")
    md.append("| Category | Score | Weight | Explanation |")
    md.append("|---|---|---|---|")
    cat_scores = report_data.get("category_scores", {})
    cat_expl = report_data.get("category_explanations", {})
    weights = report_data.get("weights", {})
    for cat, score in cat_scores.items():
        w = weights.get(cat, 0) * 100
        expl = cat_expl.get(cat, "")
        md.append(f"| {cat} | {_fmt_pct(score)}% | {w:.0f}% | {expl} |")
    md.append("")

    md.append("## 📝 Executive Summary\n")
    md.append(report_data.get("executive_summary", "N/A") + "\n")

    md.append("## 📄 Documents Processed\n")
    md.append("| Filename | Type | Pages | Status |")
    md.append("|---|---|---|---|")
    for d in report_data.get("documents_processed", []):
        md.append(f"| {d.get('filename')} | {d.get('doc_type')} | {d.get('pages')} | {d.get('status')} |")
    md.append("")

    md.append("## 🎓 Course Overview (Stage 1)\n")
    overview = report_data.get("document_overview", {})
    md.append(f"- **Objectives:** {', '.join(overview.get('course_objectives', []) or ['N/A'])}")
    md.append(f"- **Teaching Activities:** {', '.join(overview.get('teaching_activities', []) or ['N/A'])}")
    missing = overview.get("missing_information", [])
    if missing:
        md.append(f"- **Missing Information:** {', '.join(missing)}")
    md.append("")

    md.append("## 🧩 CLO Audit (Stage 2)\n")
    for clo in report_data.get("clo_audit", {}).get("clos", []):
        flag = " 🔴 *Flagged vague*" if clo.get("flagged_vague") else ""
        md.append(f"**{clo.get('clo_id')}**: {clo.get('statement')}{flag}")
        md.append(f"- Action verb: {clo.get('action_verb')} | Bloom level: {clo.get('bloom_level')} | "
                   f"Measurable: {clo.get('measurable')} | Clarity: {clo.get('clarity_score')}/5 | "
                   f"Relevance: {clo.get('relevance_score')}/5")
        if clo.get("weaknesses"):
            md.append(f"- Weaknesses: {', '.join(clo.get('weaknesses'))}")
        md.append("")

    md.append("## 🔗 CLO–PLO Alignment (Stage 3)\n")
    align = report_data.get("clo_plo_alignment", {})
    plos = align.get("plos", [])
    mappings = align.get("mappings", [])
    if plos and mappings:
        clo_ids = sorted({m.get("clo_id") for m in mappings})
        header = "| CLO | " + " | ".join(plos) + " |"
        sep = "|---|" + "---|" * len(plos)
        md.append(header)
        md.append(sep)
        for clo_id in clo_ids:
            row = [clo_id]
            for plo in plos:
                match = next((m for m in mappings if m.get("clo_id") == clo_id and m.get("plo_id") == plo), None)
                if match:
                    symbol = "✓✓" if match.get("strength") == "strong" else "✓"
                else:
                    symbol = ""
                row.append(symbol)
            md.append("| " + " | ".join(row) + " |")
    md.append("")
    if align.get("clos_without_plo_mapping"):
        md.append(f"- **CLOs without PLO mapping:** {', '.join(align['clos_without_plo_mapping'])}")
    if align.get("unjustified_mappings"):
        md.append(f"- **Potentially unjustified mappings:** {', '.join(align['unjustified_mappings'])}")
    md.append("")

    md.append("## 📋 Assessment Alignment (Stage 4)\n")
    assess = report_data.get("assessment_alignment", {})
    md.append(f"- **Coverage:** {_fmt_pct(assess.get('coverage_percent'))}%")
    if assess.get("clos_without_assessment"):
        md.append(f"- **CLOs without assessment:** {', '.join(assess['clos_without_assessment'])}")
    if assess.get("assessments_without_clo"):
        md.append(f"- **Assessments without CLO linkage:** {', '.join(assess['assessments_without_clo'])}")
    if assess.get("alignment_problems"):
        md.append(f"- **Alignment problems:** {', '.join(assess['alignment_problems'])}")
    md.append("")

    md.append("## 🌸 Bloom's Taxonomy Distribution (Stage 5)\n")
    bloom = report_data.get("bloom_analysis", {})
    dist = bloom.get("clo_distribution_percent", {})
    for level, pct in dist.items():
        md.append(f"- {level}: {_fmt_pct(pct)}%")
    md.append(f"\n*{bloom.get('disclaimer', '')}*\n")

    md.append("## 🚩 Detected Gaps\n")
    severity_icons = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    for issue in report_data.get("gaps", {}).get("issues", []):
        icon = severity_icons.get(issue.get("severity"), "⚪")
        md.append(f"{icon} **[{issue.get('severity')}] {issue.get('title')}** — {issue.get('description')}")
    md.append("")

    md.append("## 💡 Recommendations\n")
    for rec in report_data.get("recommendations", {}).get("recommendations", []):
        md.append(f"### {rec.get('problem')}")
        md.append(f"- **Evidence:** {rec.get('evidence')}")
        md.append(f"- **Impact:** {rec.get('impact')}")
        md.append(f"- **Recommendation:** {rec.get('recommendation')}")
        if rec.get("improved_clo_wording"):
            md.append(f"- **Suggested Rewording:** _{rec.get('improved_clo_wording')}_")
        md.append("")

    md.append("---")
    md.append("*Report generated by OBE-AuditAI — an AI-powered RAG-based OBE course auditor. "
               "AI-generated content should be reviewed by a qualified academic before formal use.*")

    return "\n".join(md)


def to_txt_report(report_data: Dict[str, Any]) -> str:
    """Plain-text version derived from the markdown by stripping markdown syntax."""
    md = to_markdown_report(report_data)
    txt_lines = []
    for line in md.split("\n"):
        line = line.replace("**", "").replace("*", "").replace("###", "").replace("##", "").replace("#", "")
        line = line.replace("|", " ").replace("---", "")
        txt_lines.append(line.rstrip())
    return "\n".join(txt_lines)


def to_pdf_report(report_data: Dict[str, Any]) -> Optional[bytes]:
    """Best-effort PDF generation using reportlab. Returns None on failure
    rather than crashing the app."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        import io

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                 topMargin=0.6 * inch, bottomMargin=0.6 * inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], textColor=colors.HexColor("#12203c"))
        h2_style = ParagraphStyle("H2Custom", parent=styles["Heading2"], textColor=colors.HexColor("#1b3a6b"))
        normal = styles["Normal"]

        story = []
        story.append(Paragraph("OBE-AuditAI — Course Audit Report", title_style))
        story.append(Spacer(1, 10))
        cm = report_data.get("course_meta", {})
        story.append(Paragraph(f"Course: {cm.get('course_title', 'N/A')} | Code: {cm.get('course_code', 'N/A')}", normal))
        story.append(Spacer(1, 14))

        story.append(Paragraph(f"Overall OBE Quality Score: {report_data.get('overall_score')} / 100", h2_style))
        story.append(Spacer(1, 6))

        table_data = [["Category", "Score", "Weight"]]
        weights = report_data.get("weights", {})
        for cat, score in report_data.get("category_scores", {}).items():
            table_data.append([cat, f"{_fmt_pct(score)}%", f"{weights.get(cat, 0) * 100:.0f}%"])
        t = Table(table_data, colWidths=[3 * inch, 1.2 * inch, 1.2 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b3a6b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        story.append(Paragraph("Executive Summary", h2_style))
        story.append(Paragraph(report_data.get("executive_summary", "N/A"), normal))
        story.append(Spacer(1, 14))

        story.append(Paragraph("Detected Gaps", h2_style))
        for issue in report_data.get("gaps", {}).get("issues", [])[:15]:
            story.append(Paragraph(f"[{issue.get('severity')}] {issue.get('title')}: {issue.get('description')}", normal))
        story.append(Spacer(1, 14))

        story.append(Paragraph("Recommendations", h2_style))
        for rec in report_data.get("recommendations", {}).get("recommendations", [])[:15]:
            story.append(Paragraph(f"<b>Problem:</b> {rec.get('problem')}", normal))
            story.append(Paragraph(f"<b>Recommendation:</b> {rec.get('recommendation')}", normal))
            story.append(Spacer(1, 6))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None
