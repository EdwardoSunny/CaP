"""
JSON plan dispatcher for VirtualHome-style action sequences.

Parses VH AS output of the form
    {"WALK": ["washing_machine", "1001"], "GRAB": ["soap", "1002"], ...}
preserving duplicate keys (which VH emits routinely), and dispatches each
entry through an action map onto an adapter instance.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _extract_first_json_blob(text: str) -> str:
    """Return the substring starting at the first '{' or '[' and ending at the
    matching close bracket. Tolerates leading language tags, markdown fences,
    and trailing prose.
    """
    s = text.strip()
    # Strip markdown fences
    if s.startswith("```"):
        s = s[3:]
        # drop optional language tag on the first line
        if "\n" in s:
            first_line, rest = s.split("\n", 1)
            if first_line.strip().lower() in ("json", "python", ""):
                s = rest
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    # Skip leading language tag without fences (e.g. "json\n{...}")
    if s.lower().startswith("json") and len(s) > 4 and not s[4].isalnum():
        s = s[4:].lstrip()

    # Find first '{' or '['
    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start == -1:
        raise ValueError("No JSON object/array found in LMP output")

    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    raise ValueError("Unbalanced JSON in LMP output")


def parse_vh_plan(plan_str: str) -> List[Tuple[str, list]]:
    """Parse a VH-style plan string, preserving entry order and duplicates.

    Accepts either:
      * `{"WALK": [...], "GRAB": [...], ...}` — VH AS schema (duplicate keys ok)
      * `{"output": [...]}` — VH SD schema; list items are passed through as-is
      * `[{"WALK": [...]}, {"GRAB": [...]}, ...]` — list-of-singletons variant

    Returns a list of `(action_name, args_list)` tuples.
    """
    blob = _extract_first_json_blob(plan_str)

    # object_pairs_hook fires bottom-up: the LAST call is the outermost dict,
    # so its captured items are the top-level entries (preserving duplicates).
    all_hook_calls: List[list] = []

    def _hook(items):
        all_hook_calls.append(list(items))
        return dict(items)

    parsed = json.loads(blob, object_pairs_hook=_hook)

    if isinstance(parsed, dict) and "output" in parsed and len(parsed) == 1:
        return [("_predicate", [p]) for p in parsed["output"]]

    if isinstance(parsed, list):
        flat: List[Tuple[str, list]] = []
        for item in parsed:
            if isinstance(item, dict):
                for k, v in item.items():
                    flat.append((k, v))
        return flat

    if isinstance(parsed, dict):
        top_level = all_hook_calls[-1] if all_hook_calls else []
        return [(k, v) for k, v in top_level]

    return []


def run_json_plan(
    plan_str: str,
    adapter: Any,
    action_map: Dict[str, Dict[str, Any]],
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Dispatch a parsed VH plan through the action map onto `adapter`.

    Args:
        plan_str: Raw JSON text emitted by the LMP.
        adapter: Instance exposing methods named in `action_map[*].method`.
        action_map: Mapping of VH action name → {method, args}.
        strict: If True, raise on unknown actions. Otherwise log+skip.

    Returns one result record per entry with fields:
      {action, method, args, ok, error?}
    """
    entries = parse_vh_plan(plan_str)
    results: List[Dict[str, Any]] = []

    for action, raw_args in entries:
        spec = action_map.get(action)
        if spec is None:
            msg = f"Unknown VH action: {action}"
            if strict:
                raise KeyError(msg)
            logger.warning(msg + " (skipping)")
            results.append({"action": action, "ok": False, "error": msg})
            continue

        method_name = spec["method"]
        method = getattr(adapter, method_name, None)
        if method is None:
            msg = f"Adapter has no method '{method_name}' for action '{action}'"
            if strict:
                raise AttributeError(msg)
            logger.warning(msg + " (skipping)")
            results.append({"action": action, "method": method_name,
                            "ok": False, "error": msg})
            continue

        args = list(raw_args) if isinstance(raw_args, (list, tuple)) else [raw_args]
        try:
            method(*args)
            results.append({"action": action, "method": method_name,
                            "args": args, "ok": True})
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            if strict:
                raise
            results.append({"action": action, "method": method_name,
                            "args": args, "ok": False, "error": str(e)})

    return results
