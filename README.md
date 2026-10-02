# agent-insights

Coaching tool that shows real estate brokerages how well each agent does for clients: what homes sold for versus what they were worth, what buyers paid, and how often an agent's offers win. The public version runs on **fake data only** (Santa Barbara-style prices). A brokerage runs a private copy on its own data.

Results are framed as coaching, not rankings: every score is a percent versus the average agent, with an honest range. A range that includes 0 means "not clearly different from average yet".

## What it does

1. **Fake data** (`src/generate_fake_data.py`): 150 agents, 5,000 sales, 12,600 offers, with hidden skills so we can grade the model.
2. **Naive ranking** (`src/naive_ranking.py`): average sold price per agent. Shows why the obvious approach misleads (selection bias, small samples).
3. **Model** (`src/agent_model.py`): predicts price from the home, zip, month and trend, then credits each agent for what is left over (seller side and buyer side separately).
4. **Shrinkage** (`src/shrinkage.py`): pulls thin records toward average and adds 95% ranges.
5. **Offer win rates** (`src/offer_win_rates.py`): wins / offers made, shrunk the same way.
6. **Messy data** (`src/make_messy_data.py`, `src/clean_data.py`): a dirty export and the cleaner that handles it.
7. **Coaching app** (`app/coaching_app.py`): Streamlit view per agent.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_fake_data.py        # writes data/fake/ (not committed; seed 42 repeats exactly)
cd src
python naive_ranking.py && python agent_model.py && python shrinkage.py && python offer_win_rates.py
cd ..
streamlit run app/coaching_app.py
pytest                                   # automated tests
```

## Traps this project handles

Selection bias, small samples, market timing, list-price games, hidden home quality, and target leakage. See `docs/data_dictionary.md`.

## Privacy

Never put real client, MLS or brokerage data in this repo. Private runs belong in `data/private/` (git-ignored).
