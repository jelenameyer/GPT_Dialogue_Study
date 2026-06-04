#!/usr/bin/env python3
"""
06_api_call_strategy_classification_all_chats_together.py

Replicates Costello et al. (2024), "Durably reducing conspiracy beliefs through
dialogues with AI," Science, SI Appendix 6.1 classification step for AI Dialogue Study.

For each conversation, queries the model at temperature=0 to rate the presence
and ubiquity (None / Low / Moderate / High) of each strategy, in each round,
of the AI persuader's turns.

Reads:
  - data/strategies_summarized.csv    (curated: columns name, description, count)
  - ../orig_data/data_analysis_AI_dialogue_study.csv (columns: conv_id, round, role, text)
Writes:
  - data/ratings_long.csv  (columns: conv_id, strategy, rating)

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python 06_api_call_conversation_classification.py
"""
# --------- packages ----------------------------
from __future__ import annotations

import pandas as pd
from openai import OpenAI
import json
import csv
import os
import sys
import time
from pathlib import Path

# --------- Input Arguments ----------------------

MODEL = "openai/gpt-5"
TEMPERATURE = 0.0
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
MAX_TOKENS = 6000
STRATEGIES_FILE = "data/strategies_summarized.csv"
CONVERSATIONS_FILE = "../orig_data/data_analysis_AI_dialogue_study.csv" 
OUTPUT_FILE = "data/ratings.csv"
N_ROUNDS_DEFAULT = 3
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
SLEEP_TIME = 1
SUMMARY_EVERY = 100  # print a progress summary every N conversations
# (conversation_column, rating_scale_column_name, round_number)
ROUND_COLS = [
    ("GPTResponse",  "userFirstReply",  1),
    ("GPTResponse2", "userSecondReply", 2),
    ("GPTResponse3", "userThirdReply",  3),
]

SYSTEM_TEMPLATE = """You are about to be shown the text of a written conversation about a societal
policy topic (climate change, vaccination, or economic inequality).
The two people in this conversation are a Persuader (AI) and a Participant. It is the role of
the Persuader to convince the Participant of a particular belief or policy. Each conversation
has {n_rounds} rounds.

Your job is to classify the PERSUADER'S overall persuasive technique across the whole
conversation. For each strategy below, give a single rating reflecting how much that strategy
characterizes the Persuader's contributions taken as a whole (across all {n_rounds} persuader
turns combined), not turn by turn.

** Strategy List **
{strategy_list}

Please be sure not to classify the responses of the Participant. Use the Participant's
responses only for context, so that you can understand the responses of the Persuader.

Return JSON of the form:
{{
  "ratings": {{
    "<strategy_name>": "<rating>",
    ...
  }}
}}
Use exactly the strategy names provided above as JSON keys. Each value must be a single
rating string.

** Response Scale **
Use the following response scale for each rating:
None: Strategy not used.
Low: Strategy used rarely, in a limited fashion.
Moderate: Strategy used repeatedly or with clear emphasis.
High: Strategy used extensively and/or centrally throughout the response.

Return only the JSON object. No commentary."""


# --------- Helpers ------------------------------

def load_strategies(path: str) -> list[dict[str, str]]:
    """Read curated strategies.csv. Ignores the 'count' column."""
    df = pd.read_csv(path)
    return [
        {"name": str(r["name"]).strip(), "description": str(r["description"]).strip()}
        for _, r in df.iterrows()
    ]


def format_strategy_list(strategies: list[dict[str, str]]) -> str:
    return "\n".join(f"- {s['name']}: {s['description']}" for s in strategies)


def load_conversations(path: str) -> dict[str, list[dict]]:
    """
    Wide -> long. Each input row is one conversation with up to 3 rounds.
    Columns: X, GPTResponse, userFirstReply, GPTResponse2, userSecondReply,
             GPTResponse3, userThirdReply, self_classified, ...
    """
    df = pd.read_csv(path, sep=",")

    convs: dict[str, list[dict]] = {}
    for _, r in df.iterrows():
        cid = str(r["X"])
        turns: list[dict] = []
        for pers_col, part_col, rnd in ROUND_COLS:
            pers = r.get(pers_col)
            part = r.get(part_col)
            if isinstance(pers, str) and pers.strip():
                turns.append({"round": rnd, "role": "persuader",   "text": pers.strip()})
            if isinstance(part, str) and part.strip():
                turns.append({"round": rnd, "role": "participant", "text": part.strip()})
        if turns:
            convs[cid] = turns
    return convs


def format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        label = "PERSUADER" if t["role"] == "persuader" else "PARTICIPANT"
        lines.append(f"[Round {t['round']}] {label}: {t['text']}")
    return "\n\n".join(lines)


# --------- model prompting ----------------------

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")

    client = OpenAI(
        base_url=OPENROUTER_BASE, 
        api_key=api_key
        )

    strategies = load_strategies(STRATEGIES_FILE)
    if not strategies:
        sys.exit(f"No strategies in {STRATEGIES_FILE}")
    strategy_names = {s["name"] for s in strategies}
    print(f"Loaded {len(strategies)} strategies from {STRATEGIES_FILE}")

    conversations = load_conversations(CONVERSATIONS_FILE)
    print(f"Loaded {len(conversations)} conversations from {CONVERSATIONS_FILE}")

    strategy_list_str = format_strategy_list(strategies)

    # Open output CSV and stream rows (flushed per conversation so a mid-run
    # crash still leaves completed conversations on disk; re-running truncates).
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["conv_id", "strategy", "rating"])

    start_time = time.time()
    ok_count = 0
    err_count = 0
    total = len(conversations)

    for i, (cid, turns) in enumerate(conversations.items(), start=1):
        n_rounds = max((t["round"] for t in turns), default=N_ROUNDS_DEFAULT)
        print(f"[{i}/{total}] classifying {cid}", flush=True)

        system = SYSTEM_TEMPLATE.format(
            n_rounds=n_rounds,
            strategy_list=strategy_list_str,
        )
        user_msg = format_conversation(turns)

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"effort": "minimal"}},
            )
            content = resp.choices[0].message.content
            if content is None:
                fr = resp.choices[0].finish_reason
                raise RuntimeError(f"empty content; finish_reason={fr}; usage={resp.usage}")
            parsed = json.loads(content)
            ratings = parsed.get("ratings", {})

            for sname, rating in ratings.items():
                if sname not in strategy_names:
                    print(f"    WARN: unexpected strategy in output: {sname!r}")
                    continue
                if not isinstance(rating, str):
                    print(f"    WARN: non-string rating for {sname!r}: {type(rating).__name__}={rating!r}")
                    continue
                writer.writerow([cid, sname, rating.strip()])

            # Note any strategies the model silently dropped.
            for s in strategies:
                if s["name"] not in ratings:
                    print(f"    WARN: missing strategy in output: '{s['name']}'")
            f.flush()
            ok_count += 1
            print(f"    ok ({len(ratings)} strategies rated)")
        except Exception as e:
            err_count += 1
            print(f"    ERROR: {e}", flush=True)

        if i % SUMMARY_EVERY == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0.0
            eta_min = ((total - i) / rate / 60) if rate > 0 else float("inf")
            print(
                f"[progress] {i}/{total}  ok={ok_count} err={err_count}  "
                f"elapsed={elapsed/60:.1f}m  eta={eta_min:.1f}m",
                flush=True,
            )

        time.sleep(SLEEP_TIME)

    f.close()
    print(f"\nSaved ratings to {out_path}")


if __name__ == "__main__":
    main()