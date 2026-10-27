# Data Dictionary

The columns agent-insights needs from a brokerage's sold-listings data.
The public version uses fake data only. Private copies use the brokerage's own data.

Data comes as two tables: `sales` (one row per sold home) and `agents` (one row per agent).
They connect through agent IDs.

## Table 1: sales (one row per sold home)

| Column | Type | Plain meaning | Why we need it |
|---|---|---|---|
| listing_id | text | Unique ID for the sale | Lets us find and count each sale once |
| address | text | Street address | Identifies the home. Removed before any modeling |
| zip | text | 5-digit zip code | Location is the biggest driver of price |
| beds | integer | Bedrooms | Home feature, used to estimate what the home was worth |
| baths | decimal | Bathrooms (2.5 = two full, one half) | Same |
| sqft | integer | Interior square feet | Same |
| lot_sqft | integer | Lot size in square feet | Same |
| year_built | integer | Year the home was built | Same |
| assessed_value | number | County-style public estimate of the home's value, a few percent off the truth | Counties publish tax assessments. Lets the model see part of the home's quality, so it is not mistaken for agent skill |
| list_date | date | Day it went on the market | Market timing: prices move by month |
| original_list_price | dollars | First asking price | Shows list-price games (pricing low to spark bidding) |
| final_list_price | dollars | Last asking price before sale | Compared with original to see price cuts |
| num_price_changes | integer | How many times the price changed | Many changes often means it was overpriced at first |
| sold_date | date | Closing day | Market timing |
| sold_price | dollars | What the buyer paid | The outcome we care about |
| days_on_market | integer | Days from list date to sold date | Speed of sale. Slow sales can signal overpricing |
| listing_agent_id | text | Agent who represented the seller | Credit for the seller-side result |
| buyer_agent_id | text | Agent who represented the buyer | Credit for the buyer-side result. Can be blank (buyer had no agent) |
| office | text | Office of the listing agent | Lets us compare within an office |
| team | text | Team of the listing agent, if any | Teams share credit, so we need to know |

## Table 2: agents (one row per agent)

| Column | Type | Plain meaning | Why we need it |
|---|---|---|---|
| agent_id | text | Unique agent ID (anonymized) | Links to the sales table |
| office | text | Home office | Grouping |
| team | text | Team, blank if solo | Grouping |
| years_licensed | integer | Years since first license | Experience. Helps explain results fairly |

## Table 3: offers (one row per offer, winning and losing)

| Column | Type | Plain meaning | Why we need it |
|---|---|---|---|
| offer_id | text | Unique ID for the offer | Counting |
| listing_id | text | Which listing the offer was on | Links to the sales table |
| buyer_agent_id | text | Agent who made the offer for the buyer. Can be blank | Offer win rate per buyer agent |
| offer_price | dollars | Price offered | Shows how far losing offers were from the winner |
| offer_date | date | Day the offer was made | Ordering |
| outcome | text | `won` or `lost` | Win rate = wins / offers made |

The winning offer's price equals the sale's `sold_price`.

## Optional: messy exports

Real exports are dirty. `src/make_messy_data.py` writes `sales_messy.csv` with an extra `status` column
(`sold`, `withdrawn`, `expired`), missing values, typos, prices stored as text, and duplicate rows.
`src/clean_data.py` fixes these and reports what it changed. Unsold listings are dropped (no outcome).

## Statistical traps to remember

- **Selection bias**: strong agents often get nicer homes, so raw average prices flatter them. We compare each sale to what the home was worth.
- **Small samples**: an agent with 3 sales tells us little. Results need caution or shrinking toward the average.
- **Market timing**: a sale in a hot month is not the agent's doing. We adjust for month.
- **List price games**: a low list price makes "sold over list" look great. We use estimated home value, not list price, as the yardstick.

## Fake data only: answer keys

These exist only in the fake data. Real brokerages have no such files. We use them to check our model.

- `data/fake/true_skills.csv`: each agent's hidden skill: `seller_premium` (sells for this fraction above what the home is worth), `buyer_savings` (buys this fraction below), `win_skill` (higher means offers win more often).
- `data/fake/true_home_values.csv`: each home's true value (`true_value`) and hidden quality (`quality`: view, condition) that no column records.

## Model outputs (written to data/fake/, not committed)

All scores are fractions: 0.012 means 1.2%. Every score has a 95% range (`_low`, `_high`).

- `predictions.csv` (3-1): one row per sale. `predicted_price` = what the home-only model expected at the sold date, `predicted_at_list` = at the list date. Both are cross-validated (the model never saw that home's price). `pct_error` = sold / predicted - 1.
- `speed_pricing.csv` (3-2): one row per listing agent. `speed_vs_similar` (-0.10 = sells 10% faster than similar homes), `list_vs_value` (+0.03 = first list price usually 3% above model value), `typical_list_miss`, `share_with_price_cut`. Descriptive only.
- `bootstrap_scores.csv` (3-3): seller and buyer scores with formula ranges (2C), bootstrap ranges (1,000 resamples), `under_5_deals` flag, and `fixed_k_score` (k = 10) for comparison.
- `rankings.csv` (3-4): the coaching table. Per agent: seller score, buyer savings, offer win rate (versus the brokerage average), each with a range and a group (clearly above, not clearly different, clearly below), the blended `coaching_index` with its range and `index_group`, `low_data_flag`, and speed and pricing as context. Weights and range choice live in `config/scoring.json`.
- `stress_test.csv` (3-0): how naive ranking and the model hold up when selection bias is flipped.
- `validation_summary.csv` (4-0, fake data only): per side, rank agreement and correlation with true skill (shrunk and raw), average miss in percentage points, share of 95% ranges that caught the truth for each range type, and how strongly the miss follows hidden home value.
- `validation_by_deals.csv` (4-1): the same miss and range checks split by number of deals (1-4, 5-9, 10-19, 20-49, 50+), with shrinkage on and off.
- `coverage_across_worlds.csv` (4-1): range coverage by number of deals across 5 fake worlds, with and without selection bias.
- `time_split_check.csv` (4-1): scores from the first vs second half of the sales by date. `reliability` and `implied_accuracy` need no answer key; `actual_accuracy` uses it to show when the no-answer-key check can be fooled.

Charts from `src/validate.py` are committed in `docs/`: `validation.png`, `accuracy_by_deals.png`, `stress_test.png`, `hidden_quality.png`.
