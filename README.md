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
7. **Coaching app** (`app/coaching_app.py`): Streamlit view per agent, read from `rankings.csv`: coaching index, the three scores with ranges, and speed and pricing as context.
8. **Stress test** (`src/stress_test.py`): rebuilds the fake data with selection bias flipped (better agents get cheaper homes). Average sold price collapses; the model holds.
9. **Price model** (`src/price_model.py`): home-only price, cross-validated (every home predicted by a model that never saw it). Median error about 3%.
10. **Speed and pricing** (`src/speed_and_pricing.py`): days on market vs similar homes, and list price vs model value. Shown as context, never scored, because the agent controls list price.
11. **Bootstrap ranges** (`src/bootstrap_ranges.py`): refits the model on 1,000 resampled histories (2 to 3 minutes) and flags agents with under 5 deals.
12. **Coaching table** (`src/score_agents.py`, `config/scoring.json`): blends seller, buyer and offer scores into one coaching index with a range, saved as `data/fake/rankings.csv`.
13. **Validation** (`src/validate.py`): grades every score against the hidden answer key, by number of deals, with shrinkage on and off, across 5 fake worlds, and with a check that needs no answer key. Writes the charts in `docs/`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_fake_data.py        # writes data/fake/ (not committed; seed 42 repeats exactly)
cd src
python naive_ranking.py && python agent_model.py && python shrinkage.py && python offer_win_rates.py
python stress_test.py && python price_model.py && python speed_and_pricing.py
python bootstrap_ranges.py && python score_agents.py   # bootstrap takes 2-3 minutes
python validate.py                                       # charts in docs/, about 15 seconds
cd ..
streamlit run app/coaching_app.py
pytest                                   # automated tests
```

## What the fake data shows (graded against the hidden answer key)

| Test | Result |
|---|---|
| Naive ranking (average sold price) when better agents get pricier / random / cheaper homes | 0.83 / 0.20 / -0.71 rank agreement with true skill |
| Model, same three worlds | 0.76 / 0.73 / 0.72 |
| Seller score vs true skill (standard world) | rank agreement 0.78, 95% ranges catch the truth 97% of the time |
| Buyer score vs true skill (standard world) | rank agreement 0.49, ranges catch the truth 90% (weak spot, see below) |
| Price model median error (cross-validated) | 3.2% |
| Coaching index rank agreement with the true blend of skills | 0.74 |
| Agents called "clearly above" or "clearly below" average who truly are | 100% (30 of 150 agents); the other 120 are "not clearly different from average" |

## Validation

Every agent in the fake data has a hidden true skill, so we can check the model against it. A real brokerage can't, which is why this check happens here, before anyone trusts a score.

![Estimated vs true skill](docs/validation.png)

Each dot is an agent. Seller scores line up well with the truth. Buyer scores line up much less well. Shrinkage pulls thin records toward average on purpose, so the dots sit flatter than the dashed line.

![Stress test](docs/stress_test.png)

Average sold price looks smart only when good agents happen to get pricier homes. Flip that and it ranks agents backwards (-0.71). The model holds at 0.72 to 0.76 in every world.

**Shrinkage on vs off** ([chart](docs/accuracy_by_deals.png)): shrinkage cuts the average miss by about 40% (seller 1.02 to 0.60 points, buyer 1.23 to 0.74). For agents with 1 to 4 deals it cuts the miss from 2.1 to 1.0 points (seller) and 3.6 to 1.1 (buyer). For agents with 50+ deals it barely changes anything, which is the point: it only leans on the average when the agent's own record is thin.

**Checking without an answer key.** A brokerage has no hidden truth, so `validate.py` also scores every agent twice, on the first and second half of their sales by date, and asks how well the two halves agree. When homes are assigned at random, that check guesses the real accuracy well (seller 0.62 guessed vs 0.64 true, buyer 0.52 vs 0.59). With selection bias it overrates the buyer side (0.51 guessed vs 0.39 true), because a bias that repeats in both halves looks like skill. Consistent is not the same as correct.

## Honest failure list

1. **The buyer side is weak, and the cause is hidden home quality** ([chart](docs/hidden_quality.png)). The model can't see everything that makes a home valuable (view, condition, finishes). Most of each agent's score miss tracks that hidden value (correlation 0.90 seller, -0.94 buyer). Some of that is luck, which the ranges allow for. The rest is selection bias: good agents get homes with more hidden value (correlation about +0.2 with true skill). For sellers that flatters good agents, so the ranking still looks fine. For buyers it makes good agents look like they overpaid. When the model is allowed to see true home value, buyer rank agreement jumps from 0.55 to 0.87. The fix is better quality data, not a fancier model.
2. **Ranges are too narrow for the busiest agents.** Across 5 fake worlds with selection bias, the 95% range caught the truth only 84% of the time for agents with 50+ deals (both sides), against 93% when homes were assigned at random. More deals shrink the range, but they don't shrink a bias. So the agents leaders know best are the ones most likely to get an overconfident "clearly above" or "clearly below" call. A bias allowance of about plus or minus 0.5 points brings them back to about 95%, at the cost of fewer clear calls (14% of agents down to 9%). Not built yet.
3. **Agents with under 5 deals get no real feedback.** The bootstrap range catches the truth only 64% (seller) and 43% (buyer) of the time for them, so the table uses the wider formula range (93%). That is honest, but it means none of the 14 agents with 1 to 4 deals on each side is called clearly different from average. New agents get a "first look" flag, not a score.
4. **Buyer ranges are a little narrow for agents with 20 to 49 deals, even when homes are random** (87% caught). Not explained yet. Suspects: the 8% of sales with no buyer agent, and seller and buyer effects being estimated in one regression while the range formula treats them separately.
5. **The coaching index assumes its three parts' errors are independent.** Seller and buyer scores come from the same regression and the same homes, so their errors are probably linked. Not measured yet; the index range could be too narrow or too wide.
6. **The fake world was built by the same people who built the model.** It follows the model's assumptions: skill that never changes, no teams sharing credit, no dual agency, offers won without paying more. Passing these tests is necessary, not sufficient. Real data will break things we didn't simulate.

A skeptical statistics review of the whole method, with the three biggest risks and their simplest fixes, is in [docs/statistics_review.md](docs/statistics_review.md).

## Traps this project handles

Selection bias, small samples, market timing, list-price games, hidden home quality, and target leakage. See `docs/data_dictionary.md`.

## Privacy

Never put real client, MLS or brokerage data in this repo. Private runs belong in `data/private/` (git-ignored).
