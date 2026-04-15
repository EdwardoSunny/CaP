#!/usr/bin/env python3
"""
extract_cap_prompts.py — Extractor Agent for CaP demo prompt set.

Scans EAI benchmark prompt files (BEHAVIOR action_sequencing) and filters to tasks
whose target states are expressible through pick-and-place / spatial operations only —
i.e., compatible with CaP's tabletop API.

Outputs: demo/prompts_candidates.jsonl
    One JSON object per line, each representing a candidate demo prompt.

Usage:
    python offline/extract_cap_prompts.py \\
        --behavior-dir ../eai_ctrlg/standalone_evaluation/data/prompts/behavior/action_sequencing \\
        --output demo/prompts_candidates.jsonl \\
        --verbose

Design notes:
    - BEHAVIOR task identifiers encode the task name: "boxing_books_up_for_storage_0_..."
      → task_name = "boxing books up for storage"
    - The target environment state determines CaP compatibility:
        ALLOWED predicates: ontop, inside, nextto, onfloor, under, forall+inside, forall+ontop
        BLOCKED predicates: soaked, cooked, sliced, stained, frozen, open, toggled, etc.
    - VirtualHome AS prompts use numeric-only identifiers (e.g. "27_2") with no task-name
      signal, so they are not extracted here.
    - Natural-language queries are generated from (task_name, objects, target_predicate)
      using a small rule-based template set.
"""

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# CaP API compatibility filter
# ---------------------------------------------------------------------------

# Target state predicates that are purely spatial and achievable by a robot arm.
ALLOWED_PREDICATES = {
    "ontop", "inside", "nextto", "onfloor", "under",
    "forall", "forpairs",  # quantifiers — check the inner predicate separately
    "exists",              # existential — allowed if paired with spatial inner predicate
}

# Any occurrence of these tokens in the target state signals a state-change action
# that CaP cannot perform → block the task.
BLOCKED_TOKENS = {
    "soaked", "cooked", "sliced", "stained", "frozen", "unfreeze", "freeze",
    "cleaned", "dusty", "toggled", "open", "closed", "cook", "soak",
    "dry", "saturated", "dirty",
}

# Object categories that are not tabletop-scale or not graspable.
BLOCKED_CATEGORIES = {
    "floor.n.01", "room_floor", "agent.n.01", "wall.n.01", "ceiling.n.01",
    "door.n.01", "window.n.01",
}

# Map (task_name_keywords, target_predicate) → natural-language query template.
# {objects} expands to a comma-joined list of cleaned object names.
# {dest} expands to the destination object/surface.
QUERY_TEMPLATES = [
    # inside tasks
    ({"box", "book", "storage", "boxing"},
     "inside",
     "put all the {category_plural} inside the {dest}"),
    ({"collect", "can", "aluminum"},
     "inside",
     "collect all the {category_plural} and put them in the {dest}"),
    ({"fill", "basket", "easter"},
     "inside",
     "fill the {dest} with all the items on the table"),
    ({"fill", "stocking", "christmas"},
     "inside",
     "fill the {dest} with the objects on the table"),
    ({"assemble", "gift", "basket"},
     "inside",
     "place the items into the gift baskets"),
    ({"bottle", "fruit"},
     "inside",
     "put all the fruit into the bottles"),
    ({"clear", "table", "dinner"},
     "inside",
     "clear the table — put everything in the {dest}"),
    # ontop tasks
    ({"bring", "wood"},
     "ontop",
     "move all the wooden planks to the {dest}"),
    ({"misplaced", "collect", "item"},
     "ontop",
     "place all the misplaced items on the table"),
    ({"set", "table"},
     "ontop",
     "set the table — place the plates, cups, and utensils in position"),
    # nextto / sorting
    ({"sort", "arrange"},
     "nextto",
     "arrange the objects in a row next to each other"),
    # generic fallbacks keyed only on predicate
    ({},
     "inside",
     "put all the {category_plural} into the {dest}"),
    ({},
     "ontop",
     "place the {category_plural} on top of the {dest}"),
    ({},
     "nextto",
     "arrange the objects next to the {dest}"),
    ({},
     "onfloor",
     "move the {category_plural} to the designated area"),
]


def _task_name_from_identifier(identifier: str) -> str:
    """Extract human-readable task name from a BEHAVIOR identifier string."""
    parts = identifier.split("_")
    task_words = []
    for p in parts:
        if p.isdigit():
            break
        task_words.append(p)
    return " ".join(task_words)


def _clean_obj_name(raw: str) -> str:
    """
    Strip trailing numeric index from an object name.
    'notebook_59' → 'notebook'
    'bottom_cabinet_no_top_16' → 'bottom cabinet no top'
    """
    parts = raw.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].replace("_", " ")
    return raw.replace("_", " ")


def _clean_category(raw: str) -> str:
    """'book.n.02' → 'book'"""
    return raw.split(".")[0].replace("_", " ")


def _plural(word: str) -> str:
    """Naive English pluralization for common object names."""
    if word.endswith("s") or word.endswith("x") or word.endswith("ch"):
        return word + "es"
    if word.endswith("y") and not word[-2] in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _find_target_predicate(target_state: str) -> str:
    """Return the dominant spatial predicate in the target state string."""
    for pred in ("inside", "ontop", "nextto", "onfloor", "under"):
        if pred in target_state:
            return pred
    return "ontop"  # fallback


def _is_blocked(target_state: str) -> bool:
    """Return True if the target state contains any non-spatial (state-change) tokens."""
    target_lower = target_state.lower()
    return any(token in target_lower for token in BLOCKED_TOKENS)


def _extract_task_section(prompt_text: str) -> str:
    """Return the task-specific Input section (everything after the last 'Input:' marker)."""
    idx = prompt_text.rfind("Input:")
    return prompt_text[idx:] if idx >= 0 else ""


def _parse_objects(section: str) -> tuple[list[str], list[str]]:
    """
    Parse the 'interactable objects:' block and return (names, categories).
    Only returns objects whose categories are NOT in BLOCKED_CATEGORIES.
    """
    obj_match = re.search(r"interactable objects:(.*?)Please output", section, re.DOTALL)
    if not obj_match:
        return [], []
    block = obj_match.group(1)
    raw_names = re.findall(r"'name':\s*'([^']+)'", block)
    raw_cats = re.findall(r"'category':\s*'([^']+)'", block)
    names, cats = [], []
    for n, c in zip(raw_names, raw_cats):
        if not any(bc in c for bc in BLOCKED_CATEGORIES):
            names.append(n)
            cats.append(c)
    return names, cats


def _parse_target_state(section: str) -> str:
    """Return the target environment state block as a plain string."""
    m = re.search(r"target environment state:(.*?)interactable objects:", section, re.DOTALL)
    return m.group(1).strip() if m else ""


def _find_destination(names: list[str], cats: list[str], predicate: str) -> str:
    """
    Heuristically find the destination object for the target predicate.
    Destinations are containers (box, basket, bucket, bottle) for 'inside';
    surfaces (table, countertop, shelf) for 'ontop'; etc.
    """
    CONTAINER_KEYWORDS = {"box", "carton", "basket", "bucket", "bottle", "bag", "bin", "stocking"}
    SURFACE_KEYWORDS = {"table", "countertop", "shelf", "floor", "tray", "plate"}
    NEXTTO_KEYWORDS = {"sink", "table", "counter"}

    candidates = {
        "inside": CONTAINER_KEYWORDS,
        "ontop": SURFACE_KEYWORDS,
        "nextto": NEXTTO_KEYWORDS,
        "onfloor": {"floor"},
    }.get(predicate, SURFACE_KEYWORDS)

    for name, cat in zip(names, cats):
        clean = _clean_obj_name(name)
        cat_clean = _clean_category(cat)
        if any(kw in clean or kw in cat_clean for kw in candidates):
            return clean
    # fallback: last object
    return _clean_obj_name(names[-1]) if names else "table"


def _generate_query(task_name: str, objects: list[str], cats: list[str], predicate: str) -> str:
    """
    Generate a natural-language task query from task metadata using template matching.
    """
    task_words = set(task_name.lower().split())
    dest = _find_destination(objects, cats, predicate)
    # pick dominant non-destination category
    for name, cat in zip(objects, cats):
        clean_cat = _clean_category(cat)
        if clean_cat not in ("floor", "ceiling", "wall", "door", "agent"):
            category = clean_cat
            break
    else:
        category = "object"
    category_plural = _plural(category)

    for keywords, pred, template in QUERY_TEMPLATES:
        if pred != predicate:
            continue
        if not keywords or keywords & task_words:
            query = template.format(
                category=category,
                category_plural=category_plural,
                dest=dest,
            )
            return query

    # Absolute fallback
    return f"move the {category_plural} to the {dest}"


def _assign_difficulty(objects: list[str], task_name: str) -> str:
    """Assign difficulty based on object count and task complexity keywords."""
    n = len(objects)
    complex_kw = {"all", "every", "sort", "arrange", "assemble", "fill", "pair"}
    task_words = set(task_name.lower().split())
    if n <= 2:
        return "easy"
    if n <= 4 or not (complex_kw & task_words):
        return "medium"
    return "hard"


def _infer_cap_operations(predicate: str, objects: list[str]) -> list[str]:
    """List the CaP API calls likely needed for this task."""
    ops = ["put_first_on_second"]
    if len(objects) > 3:
        ops.append("parse_obj_name")
    if predicate == "ontop" and len(objects) > 2:
        ops.append("stack_objects_in_order")
    if predicate in ("nextto",):
        ops.append("parse_position")
    return ops


def extract_behavior_candidates(data_dir: Path, verbose: bool = False) -> list[dict]:
    """
    Parse behavior/action_sequencing prompt file and return CaP-compatible candidates.
    """
    prompt_file = data_dir / "action_sequence_prompts.json"
    if not prompt_file.exists():
        print(f"[extractor] WARNING: {prompt_file} not found — skipping BEHAVIOR AS")
        return []

    data = json.loads(prompt_file.read_text())
    candidates = []
    blocked_count = 0

    for item in data:
        identifier = item["identifier"]
        task_name = _task_name_from_identifier(identifier)
        if not task_name:
            continue

        section = _extract_task_section(item["llm_prompt"])
        if not section:
            continue

        target_state = _parse_target_state(section)
        if _is_blocked(target_state):
            blocked_count += 1
            if verbose:
                print(f"  [BLOCKED] {task_name}")
            continue

        names, cats = _parse_objects(section)
        if not names:
            continue

        predicate = _find_target_predicate(target_state)
        query = _generate_query(task_name, names, cats, predicate)

        candidates.append({
            "source": "behavior/action_sequencing",
            "source_identifier": identifier,
            "task_name": task_name,
            "objects": [_clean_obj_name(n) for n in names[:6]],
            "object_categories": [_clean_category(c) for c in cats[:6]],
            "target_predicate": predicate,
            "query": query,
            "cap_operations": _infer_cap_operations(predicate, names),
            "difficulty": _assign_difficulty(names, task_name),
        })

        if verbose:
            print(f"  [OK] {task_name} → \"{query}\"")

    print(f"[extractor] BEHAVIOR AS: {len(candidates)} compatible / "
          f"{len(data)} total ({blocked_count} blocked by state-change predicates)")
    return candidates


def deduplicate(candidates: list[dict]) -> list[dict]:
    """Remove near-duplicate task names (keep first occurrence per task_name)."""
    seen_tasks: set[str] = set()
    out = []
    for c in candidates:
        key = c["task_name"]
        if key not in seen_tasks:
            seen_tasks.add(key)
            out.append(c)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract CaP-compatible tasks from EAI benchmark prompt files."
    )
    parser.add_argument(
        "--behavior-dir",
        default="../eai_ctrlg/standalone_evaluation/data/prompts/behavior/action_sequencing",
        help="Path to behavior/action_sequencing prompt directory.",
    )
    parser.add_argument(
        "--output",
        default="demo/prompts_candidates.jsonl",
        help="Output path for candidate prompts (JSONL format).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-task accept/reject decisions.",
    )
    args = parser.parse_args()

    behavior_dir = Path(args.behavior_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Extract from BEHAVIOR action_sequencing ---
    candidates = extract_behavior_candidates(behavior_dir, verbose=args.verbose)

    # Deduplicate by task name
    candidates = deduplicate(candidates)

    # Assign stable IDs
    for i, c in enumerate(candidates):
        c["id"] = f"c{i+1:03d}"

    # Write JSONL
    with out_path.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"[extractor] Wrote {len(candidates)} candidates → {out_path}")
    print()
    print("Next step: run verify_prompts.py to validate and curate demo/prompts.jsonl")
    print(f"  python offline/verify_prompts.py --input {out_path} --output demo/prompts.jsonl \\")
    print(f"    --vllm-host localhost:8000 --model dganochenko/llama-3-8b-chat")


if __name__ == "__main__":
    main()
