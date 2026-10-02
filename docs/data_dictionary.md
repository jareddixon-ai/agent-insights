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
