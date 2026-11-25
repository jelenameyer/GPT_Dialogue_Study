# AI Dialogue Study — Exploratory Analyses

Exploratory analyses supplementing the paper:

> *Conversational AI shifts beliefs and policy support among skeptics across contested societal issues* Kotz et al. (2026)

Two analysis tracks: (1) eliciting persuasion strategies from GPT-5 and
classifying how strongly each appears in every AI–participant conversation, and
(2) the most common words in participants' replies across conditions.

## Data

Download `data_analysis_AI_dialogue_study.csv` from the
[OSF project](https://osf.io/23fny/) and place it in `orig_data/`:

```
orig_data/data_analysis_AI_dialogue_study.csv
```

The code expects this exact path and nothing runs without it. The dataset is
not included in this repository.

## Layout

```
strategy_extraction/    # strategy elicitation, classification, and analysis
  01_api_call_strategy_generation.py        -> data/strategies.csv
  02_api_call_strategy_summarization.py     -> data/strategies_summarized.csv
  03_api_call_strategy_classification_all_chats_together.py  -> data/ratings.csv
  04_test_reproducibility.ipynb             # stability of temperature-0 ratings
  05_investigate_strategy_results.ipynb     # exploratory plots
common_word_analysis/   # word-frequency / word-cloud analysis of user replies
```

## Setup

```bash
pip install openai pandas numpy matplotlib nltk wordcloud networkx
export OPENROUTER_API_KEY=sk-or-...   # only needed for the API scripts (01–03)
```

Run scripts from inside their directory so the relative paths resolve:

```bash
cd strategy_extraction
python 01_api_call_strategy_generation.py
```

HTML exports of the notebooks are included for viewing without running anything.
