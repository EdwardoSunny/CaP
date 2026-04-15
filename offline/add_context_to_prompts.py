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


def build_context_field(record: dict) -> str:
    """
    Build the `context` string from the record's `objects` field, supplemented
    by any destination object extracted from the `query` field.

    Returns a Python-syntax string ready to be appended to the LMP prompt, e.g.:
        objects = ['notebook', 'hardback', 'carton']
    """
    raw_objects: list[str] = record.get("objects", [])

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_objs: list[str] = []
    for obj in raw_objects:
        key = obj.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_objs.append(obj.strip())

    # Try to add destination if not already covered.
    query = record.get("query", "")
    dest = extract_destination(query)
    if dest and dest.lower() not in seen:
        unique_objs.append(dest)

    # Format as Python list literal.
    items_str = ", ".join(f"'{o}'" for o in unique_objs)
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
            context = build_context_field(rec)
            rec["context"] = context
            if args.verbose:
                print(f"  {rec['id']}: {context}")
            f.write(json.dumps(rec) + "\n")

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
