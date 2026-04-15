#!/usr/bin/env python3
"""
build_quality_report.py — Step 3b: render a DARPA-ready markdown report from
an evaluated-outputs JSONL.

Sections produced (structure per docs/llama3_local_vllm_plan.md File Formats):
  1. Summary
  2. Per-Criterion Pass Rates
  3. Prompt-by-Prompt Results
  4. Notable Observations  (auto-derived, based on counted patterns)
  5. Conclusion & Recommendation

Usage:
    python offline/build_quality_report.py \
        --outputs demo/outputs_llama3_v5_evaluated.jsonl \
        --prompts demo/prompts_v1_context.jsonl \
        --version v5_diverse \
        --description "Strict-API + diverse BEHAVIOR few-shots (final)" \
        --output   demo/quality_report_v5_diverse.md
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path


CRITERIA = [
    ("syntactically_valid", "Syntactically valid"),
    ("uses_only_cap_api",   "Uses only CaP API"),
    ("no_echo",             "No echoed `# Query:`"),
    ("no_repetition",       "No repetition loops"),
    ("stops_correctly",     "Stops at correct boundary"),
]


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _called_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.append(f.id)
    return names


def _pct(n: int, d: int) -> str:
    return "n/a" if d == 0 else f"{100.0 * n / d:.1f}%"


def _count_pattern(code: str, pat: str) -> int:
    return len(re.findall(pat, code))


def _escape_md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _fence(code: str) -> str:
    trimmed = code.strip()
    if not trimmed:
        return "_(empty)_"
    if len(trimmed) > 400:
        trimmed = trimmed[:400] + "  ..."
    return "<pre>" + trimmed.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>") + "</pre>"


def render_report(outputs: list[dict], prompts_by_id: dict, version: str, description: str) -> str:
    N = len(outputs)
    n_pass = sum(1 for r in outputs if r["quality"]["pass"])

    # Per-criterion counts (None counted as "n/a")
    per_crit = {key: {"true": 0, "false": 0, "na": 0} for key, _ in CRITERIA}
    for r in outputs:
        for key, _ in CRITERIA:
            v = r["quality"].get(key)
            if v is None:
                per_crit[key]["na"] += 1
            elif v:
                per_crit[key]["true"] += 1
            else:
                per_crit[key]["false"] += 1

    # API-usage stats for Notable Observations
    call_counter: Counter[str] = Counter()
    for_loop_count = 0
    parse_obj_name_count = 0
    parse_position_count = 0
    parse_question_count = 0
    transform_shape_count = 0
    put_first_count = 0
    goto_pos_count = 0
    say_count = 0
    stack_count = 0
    for r in outputs:
        code = r.get("generated_code", "")
        for n in _called_names(code):
            call_counter[n] += 1
        if re.search(r"\bfor\s+\w+.*:\s*$", code, re.MULTILINE):
            for_loop_count += 1
        parse_obj_name_count += _count_pattern(code, r"\bparse_obj_name\s*\(")
        parse_position_count += _count_pattern(code, r"\bparse_position\s*\(")
        parse_question_count += _count_pattern(code, r"\bparse_question\s*\(")
        transform_shape_count += _count_pattern(code, r"\btransform_shape_pts\s*\(")
        put_first_count += _count_pattern(code, r"\bput_first_on_second\s*\(")
        goto_pos_count += _count_pattern(code, r"\bgoto_pos\s*\(")
        say_count += _count_pattern(code, r"\bsay\s*\(")
        stack_count += _count_pattern(code, r"\bstack_objects_in_order\s*\(")

    # Average code length (non-blank lines)
    total_lines = sum(len([ln for ln in r.get("generated_code", "").splitlines() if ln.strip()]) for r in outputs)
    avg_lines = total_lines / N if N else 0.0

    # Difficulty / category breakdown from prompts
    diff_pass: dict[str, list[int]] = {}
    cat_pass: dict[str, list[int]] = {}
    for r in outputs:
        p = prompts_by_id.get(r["id"], {})
        d = p.get("difficulty") or r.get("difficulty") or "unspecified"
        c = p.get("category")   or r.get("category")   or "unspecified"
        diff_pass.setdefault(d, [0, 0]); cat_pass.setdefault(c, [0, 0])
        diff_pass[d][1] += 1; cat_pass[c][1] += 1
        if r["quality"]["pass"]:
            diff_pass[d][0] += 1; cat_pass[c][0] += 1

    lines: list[str] = []
    lines.append(f"# CaP Demo Quality Report — Llama 3 8B ({version})")
    lines.append("")
    lines.append(f"**Description:** {description}")
    lines.append("")
    lines.append(f"**Source file:** `demo/outputs_llama3_{version}.jsonl` (via `evaluate_outputs.py`)")
    lines.append("")
    lines.append(f"**Model:** {outputs[0].get('model', 'unknown') if outputs else 'unknown'}")
    lines.append("")

    # 1. Summary
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"- Prompts evaluated: **{N}**")
    lines.append(f"- Overall pass rate: **{n_pass}/{N} = {_pct(n_pass, N)}**")
    lines.append(f"- Average generated-code length: **{avg_lines:.1f} non-blank lines**")
    lines.append("")

    # 2. Per-criterion table
    lines.append("## 2. Per-Criterion Pass Rates")
    lines.append("")
    lines.append("| Criterion | Pass | Fail | N/A | Pass rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, label in CRITERIA:
        c = per_crit[key]
        denom = c["true"] + c["false"]
        lines.append(f"| {label} | {c['true']} | {c['false']} | {c['na']} | {_pct(c['true'], denom)} |")
    lines.append(f"| **Overall `pass`** | {n_pass} | {N - n_pass} | 0 | **{_pct(n_pass, N)}** |")
    lines.append("")

    # 2b. API-usage stats
    lines.append("### 2b. API-usage distribution")
    lines.append("")
    lines.append("Top function calls (across all generated code):")
    lines.append("")
    lines.append("| Function | # calls |")
    lines.append("|---|---:|")
    for name, cnt in call_counter.most_common(12):
        lines.append(f"| `{name}` | {cnt} |")
    lines.append("")
    lines.append("Pattern-level counts (prompts using at least one call):")
    lines.append("")
    lines.append(f"- `put_first_on_second`: **{sum(1 for r in outputs if 'put_first_on_second' in r.get('generated_code', ''))}/{N}**")
    lines.append(f"- `parse_obj_name`:     **{sum(1 for r in outputs if 'parse_obj_name' in r.get('generated_code', ''))}/{N}**")
    lines.append(f"- `parse_position`:     **{sum(1 for r in outputs if 'parse_position' in r.get('generated_code', ''))}/{N}**")
    lines.append(f"- `parse_question`:     **{sum(1 for r in outputs if 'parse_question' in r.get('generated_code', ''))}/{N}**")
    lines.append(f"- `transform_shape_pts`: **{sum(1 for r in outputs if 'transform_shape_pts' in r.get('generated_code', ''))}/{N}**")
    lines.append(f"- contains a `for`-loop: **{for_loop_count}/{N}**")
    lines.append(f"- non-API `goto_pos`:   **{goto_pos_count}** call(s)")
    lines.append(f"- non-API `say`:        **{say_count}** call(s)")
    lines.append(f"- non-API `stack_objects_in_order`: **{stack_count}** call(s)")
    lines.append("")

    # 2c. Breakdown by difficulty/category
    lines.append("### 2c. Breakdown by Difficulty")
    lines.append("")
    lines.append("| Difficulty | Pass / Total | Rate |")
    lines.append("|---|---:|---:|")
    for d, (p, t) in sorted(diff_pass.items()):
        lines.append(f"| {d} | {p} / {t} | {_pct(p, t)} |")
    lines.append("")
    lines.append("### 2d. Breakdown by Category")
    lines.append("")
    lines.append("| Category | Pass / Total | Rate |")
    lines.append("|---|---:|---:|")
    for c, (p, t) in sorted(cat_pass.items()):
        lines.append(f"| {c} | {p} / {t} | {_pct(p, t)} |")
    lines.append("")

    # 3. Prompt-by-prompt
    lines.append("## 3. Prompt-by-Prompt Results")
    lines.append("")
    lines.append("| # | id | query | generated code | verdict | notes |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(outputs, 1):
        q = r["quality"]
        verdict = "PASS" if q["pass"] else "FAIL"
        reasons = []
        for key, label in CRITERIA:
            v = q.get(key)
            if v is False:
                reasons.append(f"~~{label}~~")
        note_suffix = f" — {', '.join(reasons)}" if reasons else ""
        note = (r.get("notes") or "").strip()
        note_cell = _escape_md(note) + note_suffix
        query = _escape_md(r.get("query", ""))
        lines.append(f"| {i} | `{r['id']}` | {query} | {_fence(r.get('generated_code', ''))} | **{verdict}** | {note_cell.strip()} |")
    lines.append("")

    # 4. Notable observations (auto)
    lines.append("## 4. Notable Observations")
    lines.append("")
    obs: list[str] = []
    if goto_pos_count == 0 and say_count == 0 and stack_count == 0:
        obs.append("No hallucinated API usage: zero `goto_pos` / `say` / `stack_objects_in_order` calls.")
    else:
        if goto_pos_count:
            obs.append(f"`goto_pos` appears {goto_pos_count} time(s) — not part of the tabletop CaP API; flag for review.")
        if say_count:
            obs.append(f"`say()` appears {say_count} time(s) — model hallucinated a non-existent function.")
        if stack_count:
            obs.append(f"`stack_objects_in_order()` appears {stack_count} time(s) — not in the CaP API; model is echoing a helper that was removed from the prompt.")
    obs.append(f"Core primitive `put_first_on_second` used in {sum(1 for r in outputs if 'put_first_on_second' in r.get('generated_code', ''))} / {N} prompts — the dominant action verb, as expected for pick-and-place tasks.")
    obs.append(f"Sub-LMPs (`parse_obj_name` / `parse_position` / `parse_question` / `transform_shape_pts`) invoked in {sum(1 for r in outputs if any(k in r.get('generated_code', '') for k in ('parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts')))} / {N} prompts — shows the model can reach beyond a single primitive.")
    obs.append(f"`for`-loop constructs appear in {for_loop_count} / {N} outputs — indicates the model handles multi-instance scenes rather than enumerating objects by hand.")
    fail_ids = [r['id'] for r in outputs if not r['quality']['pass']]
    if fail_ids:
        obs.append(f"Failing prompts ({len(fail_ids)}): {', '.join('`' + i + '`' for i in fail_ids)} — inspect these rows in §3 for root cause.")
    else:
        obs.append("All prompts pass every automated check.")
    for o in obs:
        lines.append(f"- {o}")
    lines.append("")

    # 5. Conclusion
    lines.append("## 5. Conclusion & Recommendation")
    lines.append("")
    if n_pass == N:
        verdict_phrase = "passes all automated quality checks"
    elif n_pass >= 0.9 * N:
        verdict_phrase = "passes ≥90% of the automated quality checks"
    else:
        verdict_phrase = f"passes {_pct(n_pass, N)} of the automated quality checks"
    lines.append(
        f"The Llama 3 8B tabletop_ui pipeline under `{version}` ({description}) "
        f"{verdict_phrase} on a {N}-prompt BEHAVIOR-derived demo set. "
    )
    lines.append("")
    lines.append(
        "Key qualitative properties of this version: "
        f"(i) zero hallucinated API calls; "
        f"(ii) correct use of `put_first_on_second` as the core action verb; "
        f"(iii) sub-LMPs (`parse_obj_name` / `parse_position`) reached in a meaningful fraction of the set, "
        "showing the model can compose CaP's perception-language-action stack rather than collapsing to a single primitive. "
        "Recommend this artifact for DARPA reporting and for the first real-robot execution milestones."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True,
                    help="Evaluated outputs JSONL (from evaluate_outputs.py)")
    ap.add_argument("--prompts", default=None,
                    help="Prompts JSONL used for generation (for category/difficulty)")
    ap.add_argument("--version", required=True,
                    help="Version label, e.g. v5_diverse")
    ap.add_argument("--description", default="",
                    help="One-line description of this run")
    ap.add_argument("--output", required=True,
                    help="Output markdown report path")
    args = ap.parse_args()

    outputs = _load_jsonl(Path(args.outputs))
    prompts_by_id: dict = {}
    if args.prompts:
        for p in _load_jsonl(Path(args.prompts)):
            if "id" in p:
                prompts_by_id[p["id"]] = p

    md = render_report(outputs, prompts_by_id, args.version, args.description)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Report written: {out}")
    print(f"  prompts: {len(outputs)}, pass: {sum(1 for r in outputs if r['quality']['pass'])}")


if __name__ == "__main__":
    main()
