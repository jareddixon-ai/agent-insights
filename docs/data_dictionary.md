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
| assessed_value | number | County-style public estimate of the home's value, a few percent off the truth | Yes, counties publish tax assessments. Used to separate home quality from agent skill |
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

## Statistical traps to remember

- **Selection bias**: strong agents often get nicer homes, so raw average prices flatter them. We compare each sale to what the home was worth.
- **Small samples**: an agent with 3 sales tells us little. Results need caution or shrinking toward the average.
- **Market timing**: a sale in a hot month is not the agent's doing. We adjust for month.
- **List price games**: a low list price makes "sold over list" look great. We use estimated home value, not list price, as the yardstick.

## Fake data only: answer key

`data/fake/true_skills.csv` holds each fake agent's hidden skill (seller premium, buyer savings).
It exists only in the fake data. Real brokerages have no such file. We use it to check our model.
