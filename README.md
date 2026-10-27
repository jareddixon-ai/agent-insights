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
8. **Stress test** (`src/stress_test.py`): rebuilds the fake data with selection bias flipped (better agents get cheaper homes). Average sold price collapses; the model holds.
9. **Price model** (`src/price_model.py`): home-only price, cross-validated (every home predicted by a model that never saw it). Median error about 3%.
10. **Speed and pricing** (`src/speed_and_pricing.py`): days on market vs similar homes, and list price vs model value. Shown as context, never scored, because the agent controls list price.
11. **Bootstrap ranges** (`src/bootstrap_ranges.py`): refits the model on 1,000 resampled histories (2 to 3 minutes) and flags agents with under 5 deals.
12. **Coaching table** (`src/score_agents.py`, `config/scoring.json`): blends seller, buyer and offer scores into one coaching index with a range, saved as `data/fake/rankings.csv`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_fake_data.py        # writes data/fake/ (not committed; seed 42 repeats exactly)
cd src
python naive_ranking.py && python agent_model.py && python shrinkage.py && python offer_win_rates.py
python stress_test.py && python price_model.py && python speed_and_pricing.py
python bootstrap_ranges.py && python score_agents.py   # bootstrap takes 2-3 minutes
cd ..
streamlit run app/coaching_app.py
pytest                                   # automated tests
```

## What the fake data shows (graded against the hidden answer key)

| Test | Result |
|---|---|
| Naive ranking (average sold price) when better agents get pricier / random / cheaper homes | 0.83 / 0.20 / -0.71 rank agreement with true skill |
| Model, same three worlds | 0.76 / 0.73 / 0.72 |
| Price model median error (cross-validated) | 3.2% |
| Coaching index rank agreement with the true blend of skills | 0.74 |
| Agents called "clearly above" or "clearly below" average who truly are | 100% (30 of 150 agents); the other 120 are "not clearly different from average" |

Known weak spot: the buyer side. Good agents get pricier homes, and the public assessed value misses part of each home's quality, so leftover quality makes good buyer agents look like they paid more. When the model is allowed to see true home value (only possible in fake data), buyer rank agreement jumps from 0.55 to 0.87. Better home-quality data (prior sale prices, condition ratings) is the fix, not a fancier model.

## Traps this project handles

Selection bias, small samples, market timing, list-price games, hidden home quality, and target leakage. See `docs/data_dictionary.md`.

## Privacy

Never put real client, MLS or brokerage data in this repo. Private runs belong in `data/private/` (git-ignored).
