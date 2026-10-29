# Pitch kit numbers and where they come from

Every number in the Phase 8 pitch kit (one-pager, demo script, pilot proposal, data checklist, emails) is a tested result on **fake data** (150 made-up agents, 5,000 made-up sales, seed 42). None is a result on a real brokerage. `tests/test_phase8.py` checks that each "README text" below still appears in the README, so if a result changes, the test fails and the kit must be updated.

| Number in the kit | What it means | README text it must match | README section | Produced by |
|---|---|---|---|---|
| 150 agents, 5,000 sales | Size of the fake brokerage | `150 agents, 5,000 sales` | What it does, item 1 | `src/generate_fake_data.py` |
| 0.74 | Seller score rank agreement with true skill (1 = exact order, 0 = no relation) | `rank agreement 0.74` | What the fake data shows | `src/score_agents.py` |
| 97% | Seller 95% ranges that contain the true skill | `catch the truth 97% of the time` | What the fake data shows | `src/validate.py` |
| 14 of 150, 100% | Agents called clearly above or below average, and how many of those calls were right | `100% (14 of 150 agents)` | What the fake data shows | `src/score_agents.py` |
| 135 of 150 | Agents "not clearly different from average" | `135 of 150 agents` | How to read the dashboard; Ethics | `src/score_agents.py` |
| 0.83 / 0.20 / -0.71 | Average sold price as a ranking, when better agents get pricier / random / cheaper homes | `0.83 / 0.20 / -0.71` | What the fake data shows | `src/stress_test.py` |
| 0.71 / 0.58 / 0.31 | The model in the same three worlds | `0.71 / 0.58 / 0.31` | What the fake data shows | `src/stress_test.py` |
| 4.6% | Price model median error, every home predicted by a model that never saw it | `4.6%` | What the fake data shows | `src/price_model.py` |
| 0.17 | Buyer score rank agreement (why it is context only) | `rank agreement 0.17` | What the fake data shows | `src/validate.py` |
| 0.61 / 0.67 / 0.73 | Buyer score with an independent value estimate of about 4% typical miss | `0.61 / 0.67 / 0.73` | What the fake data shows | `src/condition_test.py` |
| about 1%, 19 to under 1 | A loan appraisal misses the sale price by about 1% and wipes out almost every clear seller call | `about 1%` and `19 seller calls fall to under 1` | Condition data | `src/condition_test.py` |
| 0.89 vs 0.46 | Fair offer score vs raw win rate, rank agreement with hidden win skill | `raw win rate 0.46, fair offer score 0.89` | What the fake data shows | `src/offer_metrics.py`, `src/offer_validate.py` |
| 100 of 100 | Fake offer PDFs read back exactly, no names stored | `100 read back exactly` | What the fake data shows | `src/extract_offers.py` |
| 0.74 to 0.60 | Seller rank agreement without the last sale date and price | `0.74 to 0.60` | Data dictionary, Private copy (also README Private copy: `gives 0.60, and with it 0.74`) | `src/sale_history_test.py`, `src/readiness_check.py` |
| 4,800 of 5,050, 0.62 | Messy fake export: usable rows, seller rank agreement | `4,800 of 5,050 rows usable` and `0.62` | Private copy | `src/make_fake_export.py`, `src/run_private.py` |
| median of 19 listings | Typical agent's listings in the fake data | `at a median of 19` | Readiness check output (Phase 8 README note) | `src/readiness_check.py` |
| 3 listings per half-year | Why half-year trends are mostly noise | `3 listings per period` | How to read the dashboard | `src/trends.py` |
| 200 sales | Fewest usable sales the readiness check accepts | `Under 200 usable sales blocks scoring` | Data dictionary, Private copy | `src/readiness_check.py` |
| 10 peers | Fewest peers before the report card shows a peer chart | `at least 10 other agents` | How to read the dashboard | `app/coaching_app.py` |
| 0.65 guessed, 0.65 true | The split-half self-check (no answer key) guessed seller accuracy exactly; it is fooled on the buyer side | `0.65 guessed, 0.65 true` | Validation | `src/validate.py` |
| 0.71 to 0.55 | Seller accuracy when the condition rating is written by the listing agent (a leak) | `0.71 to 0.55` | Condition data | `src/condition_test.py` |
| about 4% | Typical miss of an independent value estimate good enough to turn buyer scores on | `about 4% typical miss` | Honest failure list, item 2 | `src/condition_test.py` |
| 1.5% | Readiness check leak warning for a value estimate this close to the sale price | `under 1.5%` | Data dictionary, Private copy | `src/readiness_check.py` |

Two numbers are pilot terms, not results: 60 days and $0. They are choices, set in the pilot proposal.

Not used in the kit on purpose: the offer score's fresh-world coverage (89.7%) and lean (0.165). They matter for the decision to keep offers context only, but they need a paragraph of explanation that a one-pager can't carry. The kit says "offer score: context only, failed one pre-set test narrowly" instead.

One number in the demo script comes from the demo snapshot, not the README: agent A133's seller score (about +7%, range +4.7% to +10.7%, 62 listings) in `demo/rankings.csv`. It is what the hosted demo shows on that agent's card.

The pitch kit itself (one-pager, demo script, pilot proposal, NDA and data checklist, outreach emails) lives in Jared's Claude Docs, not in this repo, because it is written for one person's outreach.
