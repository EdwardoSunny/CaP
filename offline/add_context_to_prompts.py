#!/usr/bin/env python3
"""
add_context_to_prompts.py — Enrich prompts_candidates.jsonl with a `context` field.

The `context` field is a Python-formatted string:
    objects = ['notebook', 'hardback', 'carton']

This matches the format used by every few-shot example in tabletop_ui_prompt.txt,
so when batch_lmp_codegen.py appends it just before the query, the LLM sees:

    ...few-shot examples...
    objects = ['notebook', 'hardback', 'carton']
    # Query: put all the books inside the carton.

instead of the generic "box, battery, plastic, tin foil" default.

Source of the object list:
  1. Unique entries from the `objects` field (deduped, order-preserved).
  2. Any destination object extracted from the query text (if not already present).

Usage:
    python offline/add_context_to_prompts.py \
        --input demo/prompts_candidates.jsonl \
        --output demo/prompts_v1_context.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Destination extraction from query text
# ---------------------------------------------------------------------------

# Ordered so more specific patterns are tried first.
_DEST_PATTERNS = [
    # "put all X inside/into the Y"
    r'\b(?:into|inside)\s+the\s+([a-z][a-z\s]+?)(?:\s*[\.,]|$)',
    # "fill the Y with ..."
    r'\bfill\s+the\s+([a-z][a-z\s]+?)\s+with\b',
    # "place X on top of the Y"
    r'\bon\s+top\s+of\s+the\s+([a-z][a-z\s]+?)(?:\s*[\.,]|$)',
    # "arrange X next to the Y"
    r'\bnext\s+to\s+the\s+([a-z][a-z\s]+?)(?:\s*[\.,]|$)',
    # "put them in the Y" / "put everything in the Y"
    r'\bput\s+(?:them|everything|it)\s+in\s+(?:the\s+)?([a-z][a-z\s]+?)(?:\s*[\.,]|$)',
    # generic "on the Y" at end
    r'\bon\s+the\s+([a-z][a-z\s]+?)(?:\s*[\.,]|$)',
    # "to the Y" at end
    r'\bto\s+the\s+([a-z][a-z\s]+?)(?:\s*$)',
]


def extract_destination(query: str) -> str | None:
    """
    Extract the destination object name from the query text.
    Returns None if no clear destination is found.
    """
    q = query.lower().strip().rstrip('.')
    for pattern in _DEST_PATTERNS:
        m = re.search(pattern, q)
        if m:
            dest = m.group(1).strip()
            # Filter out non-object phrases
            if dest and len(dest.split()) <= 5 and dest not in {
                'designated area', 'table', 'here', 'there', 'it', 'them'
            }:
                return dest
    return None


def build_context_field(record: dict, keep_duplicates: bool = False) -> str:
    """
    Build the `context` string from the record's `objects` field, supplemented
    by any destination object extracted from the `query` field.

    Returns a Python-syntax string ready to be appended to the LMP prompt, e.g.:
        objects = ['notebook', 'hardback', 'carton']

    When keep_duplicates=True, the full `objects` list is preserved (minus
    redundant duplicates of the destination). This signals to the LLM that
    there are multiple instances of the same type, encouraging use of
    parse_obj_name('the books', ...) + for-loop patterns instead of
    enumerating unique names by hand.
    """
    raw_objects: list[str] = record.get("objects", [])

    if keep_duplicates:
        # Preserve every instance; only deduplicate if the destination duplicates one.
        objs = [o.strip() for o in raw_objects]
        seen_lower = {o.lower() for o in objs}
    else:
        # Deduplicate while preserving order.
        seen_lower: set[str] = set()
        objs: list[str] = []
        for obj in raw_objects:
            key = obj.lower().strip()
            if key not in seen_lower:
                seen_lower.add(key)
                objs.append(obj.strip())

    # Try to add destination if not already covered.
    query = record.get("query", "")
    dest = extract_destination(query)
    if dest and dest.lower() not in seen_lower:
        objs.append(dest)

    items_str = ", ".join(f"'{o}'" for o in objs)
    return f"objects = [{items_str}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add per-prompt `context` field derived from the `objects` field.",
    )
    parser.add_argument("--input",  required=True, help="Input JSONL (prompts_candidates.jsonl)")
    parser.add_argument("--output", required=True, help="Output JSONL (prompts_v1_context.jsonl)")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Keep every instance in the objects list (signal multi-instance "
                             "scenes — encourages parse_obj_name + for-loop usage in generation)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"[error] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    for line in in_path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    print(f"Processing {len(records)} records from {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            context = build_context_field(rec, keep_duplicates=args.keep_duplicates)
            rec["context"] = context
            if args.verbose:
                print(f"  {rec['id']}: {context}")
            f.write(json.dumps(rec) + "\n")

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
