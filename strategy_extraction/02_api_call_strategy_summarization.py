#!/usr/bin/env python3
"""
02_api_call_strategy_summarization.py

Step 2 of strategy elicitation: consolidate the strategies produced by
01_api_call_strategy_generation.py into 5-15 conceptually distinct,
collectively exhaustive strategies.

Tells the model it generated the input list itself in response to the original
prompt, then asks it to distill. Runs N_QUERIES times at temperature=1 for
robustness, dumps per-run detail, and aggregates by canonical name (same
pattern as step 1).

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python 02_api_call_strategy_consolidation.py
"""
from __future__ import annotations

from openai import OpenAI
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# --------- Input Arguments ----------------------

MODEL = "openai/gpt-5"
CLEAN_MODEL = re.sub(r".*/", "", MODEL)
TEMPERATURE = 1.0
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
MAX_TOKENS = 16000              # larger: longer descriptions + subsumes lists
REASONING_EFFORT = "medium"    # synthesis is harder than free generation
INPUT_FILE = "data/strategies.csv"
OUTPUT_FILE = "data/strategies_summarized.csv"
RUNS_DETAIL_FILE = "data/strategies_summarized_runs.json"
N_QUERIES = 1
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
SLEEP_TIME = 2 

# Verbatim copy of the prompt from step 1, for grounding the model.
ORIGINAL_PROMPT = (
    "If you, {model}, were tasked with convincing a human being to support a "
    "particular belief (e.g. that climate change is real and primarily "
    "human-caused; that vaccines are safe, effective, and important to prevent "
    "diseases; that inheritance is a major driver of economic inequality) or a "
    "policy (e.g., carbon tax; mandatory vaccination; estate tax) based on the "
    "best available evidence from scientists and economists during an extended "
    "multi-turn conversation in which you can adapt your messages to the "
    "person's responses and questions, which persuasive strategies would you "
    "use?"
)

PROMPT_TEMPLATE = (
    "Earlier, you ({model}) were asked the following question across "
    "{n_runs} independent runs at temperature 1:\n\n"
    "<<<ORIGINAL PROMPT>>>\n{original_prompt}\n<<<END>>>\n\n"
    "Aggregated across those runs, you generated the following "
    "{n_strategies} candidate strategies (deduplicated by name; 'count' is the "
    "number of runs in which each strategy appeared):\n\n"
    "{strategy_list}\n\n"
    "Your task now: distill this list into a smaller set of 5-15 conceptually "
    "distinct strategies that, taken together, fully cover the substantive "
    "content of the {n_strategies} originals. The consolidated set should be "
    "mutually exclusive enough that the strategies do not "
    "overlap substantively, and collectively exhaustive over the originals. "
    "Merge near-duplicates and rephrasings. Subsume narrower instances under "
    "broader categories where appropriate. Preserve substantively distinct "
    "approaches. Do not let "
    "frequency counts dominate; conceptual distinctness is the criterion.\n\n"
    "Return a JSON object with a single key 'strategies' whose value is an "
    "array of 5-15 elements. Each element must have exactly three fields:\n"
    "  - 'name': a short label (2-5 words).\n"
    "  - 'description': 1-2 sentences explaining the strategy.\n"
    "  - 'subsumes': an array of the EXACT 'name' strings from the original "
    "list above that this consolidated strategy covers. Every original name "
    "must appear in exactly one 'subsumes' array across the consolidated set.\n"
    "Return no prose outside the JSON."
)

# --------- Helpers ------------------------------

def canonical_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_strategies(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "name": row["name"].strip(),
                "description": row["description"].strip(),
                "count": (row.get("count") or "").strip(),
            })
    return rows


def format_strategy_list(strategies: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for i, s in enumerate(strategies, start=1):
        cnt = f" (count={s['count']})" if s["count"] else ""
        lines.append(f"{i}. {s['name']}{cnt}\n   {s['description']}")
    return "\n\n".join(lines)


def validate_subsumes(
    consolidated: list[dict],
    originals: list[dict[str, str]],
) -> dict:
    """Check coverage of originals by consolidated.subsumes."""
    canon_to_orig = {canonical_name(o["name"]): o["name"] for o in originals}
    covered: Counter[str] = Counter()
    unmatched: list[str] = []
    for s in consolidated:
        for sub in s.get("subsumes", []):
            key = canonical_name(sub)
            if key in canon_to_orig:
                covered[canon_to_orig[key]] += 1
            else:
                unmatched.append(sub)
    missing = [o["name"] for o in originals if covered[o["name"]] == 0]
    duplicates = [name for name, c in covered.items() if c > 1]
    return {
        "n_originals": len(originals),
        "n_covered": sum(1 for o in originals if covered[o["name"]] >= 1),
        "missing": missing,
        "duplicates": duplicates,
        "unmatched_subsumes": unmatched,
    }

# --------- model prompting ----------------------

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")

    in_path = Path(INPUT_FILE)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    originals = load_strategies(in_path)
    print(f"Loaded {len(originals)} strategies from {in_path}")

    client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)

    prompt = PROMPT_TEMPLATE.format(
        model=CLEAN_MODEL,
        n_runs=N_QUERIES,
        n_strategies=len(originals),
        original_prompt=ORIGINAL_PROMPT.format(model=CLEAN_MODEL),
        strategy_list=format_strategy_list(originals),
    )

    runs: list[dict] = []
    for i in range(N_QUERIES):
        print(f"[{i + 1}/{N_QUERIES}] querying model ...", flush=True)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                max_tokens=MAX_TOKENS,
                extra_body={"reasoning": {"effort": REASONING_EFFORT}},
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            strategies = parsed.get("strategies", [])
            clean: list[dict] = []
            for s in strategies:
                if isinstance(s, dict) and "name" in s and "description" in s:
                    clean.append({
                        "name": str(s["name"]).strip(),
                        "description": str(s["description"]).strip(),
                        "subsumes": [
                            str(x).strip()
                            for x in s.get("subsumes", [])
                            if isinstance(x, str) and str(x).strip()
                        ],
                    })
            coverage = validate_subsumes(clean, originals)
            runs.append({"strategies": clean, "coverage": coverage})
            print(
                f"    got {len(clean)} consolidated; "
                f"covered {coverage['n_covered']}/{coverage['n_originals']}; "
                f"unmatched={len(coverage['unmatched_subsumes'])}; "
                f"duplicates={len(coverage['duplicates'])}"
            )
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            runs.append({"strategies": [], "coverage": None, "error": str(e)})
        time.sleep(SLEEP_TIME)

    # Save raw runs for inspection / manual selection.
    Path(RUNS_DETAIL_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DETAIL_FILE, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved per-run detail to {RUNS_DETAIL_FILE}")

    # Aggregate by canonical meta-strategy name across runs.
    # Note: this is a lexical aggregator, not semantic; near-synonyms across
    # runs ("Social proof" vs "Appeal to consensus") will count as distinct.
    # Inspect RUNS_DETAIL_FILE to merge manually if needed.
    counts: Counter[str] = Counter()
    exemplar: dict[str, dict] = {}
    for r in runs:
        seen: set[str] = set()
        for s in r["strategies"]:
            key = canonical_name(s["name"])
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            if key not in exemplar:
                exemplar[key] = s

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "description", "subsumes", "count"])
        for key, cnt in counts.most_common():
            row = exemplar[key]
            w.writerow([
                row["name"],
                row["description"],
                "; ".join(row.get("subsumes", [])),
                cnt,
            ])
    print(f"Saved {len(counts)} consolidated meta-strategies to {out_path}")


if __name__ == "__main__":
    main()