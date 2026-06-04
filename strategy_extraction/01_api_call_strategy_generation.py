#!/usr/bin/env python3
"""
01_api_call_strategy_generation.py

Replicates Costello et al. (2024), "Durably reducing conspiracy beliefs through
dialogues with AI," Science, SI Appendix 6.1 strategy-elicitation step for AI Dialogue Study.

Queries the model N_QUERIES times at temperature=1 to elicit candidate persuasion
strategies, then collects all in a dataset for us to decide which strategies to include in the classification prompt.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python 05_api_call_strategy_generation.py
"""
# --------- packages ----------------------------
from __future__ import annotations

from openai import OpenAI # SDK (software developement kit, wraps json api request into easier format)
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import csv

# --------- Input Arguments ----------------------

MODEL= "openai/gpt-5"
CLEAN_MODEL = re.sub(r".*/", "", MODEL)
TEMPERATURE = 1.0
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
MAX_TOKENS = 3000
REASONING_EFFORT = "minimal"
OUTPUT_FILE = "data/strategies.csv"
N_QUERIES = 10
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
SLEEP_TIME = 2

PROMPT_TEMPLATE = (
    """If you, {model}, were tasked with convincing a human being to support a particular belief (e.g. that climate change is real and primarily human-caused; that vaccines are safe, effective, and important to prevent diseases; that inheritance is a major driver of economic inequality) or a policy (e.g., carbon tax; mandatory vaccination; estate tax) based on the best available evidence from scientists and economists during an extended multi-turn conversation in which you can adapt your messages to the person's responses and questions, which persuasive strategies would you use?\n
    The conversational format permits both (a) strategies you would plan in advance and (b) strategies that exploit the back-and-forth of a conversation. 
    Include strategies of both types.\n
    Please return 5-10 CONCEPTUALLY DISTINCT strategies.

Return a JSON object with a single key 'strategies' whose value is an array. Each element must have exactly two fields: 'name' (a short label, 2-5 words) and 'description' (1-2 sentences explaining the strategy). Return no prose outside the JSON.
"""
)

# --------- Helpers ------------------------------


def canonical_name(name: str) -> str:
    """
    Make strategy names easier and consistent
    """
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------- model prompting ----------------------

def main():
   
    api_key = os.environ.get("OPENROUTER_API_KEY") # having done: export OPENROUTER_API_KEY="key" in terminal, of the correct env before
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")

    client = OpenAI(
        base_url=OPENROUTER_BASE,
        api_key=api_key
    )

    prompt = PROMPT_TEMPLATE.format(
        model=CLEAN_MODEL
    )

    runs: list[list[dict[str, str]]] = []
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
            # Normalize shape
            clean: list[dict[str, str]] = []
            for s in strategies:
                if isinstance(s, dict) and "name" in s and "description" in s:
                    clean.append({
                        "name": str(s["name"]).strip(),
                        "description": str(s["description"]).strip(),
                    })
            runs.append(clean)
            print(f"    got {len(clean)} strategies")
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            runs.append([])
        time.sleep(SLEEP_TIME)

    # count each canonical strategy once per run it appears in.
    counts: Counter[str] = Counter()
    exemplar: dict[str, dict[str, str]] = {}
    for run in runs:
        seen: set[str] = set()
        for s in run:
            key = canonical_name(s["name"])
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            if key not in exemplar:
                exemplar[key] = s

    # Write csv with name, description, count (sorted by count desc).
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "description", "count"])
        for key, cnt in counts.most_common():
            w.writerow([exemplar[key]["name"], exemplar[key]["description"], cnt])
    print(f"\nSaved {len(counts)} strategies to {out_path}")


if __name__ == "__main__":
    main()


