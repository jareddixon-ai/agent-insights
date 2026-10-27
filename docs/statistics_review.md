# Statistics review (Phase 4)

Prompt used: "Act as a skeptical statistician reviewing agent-insights. Find the three biggest ways these agent scores could be unfair or wrong. For each one, give the simplest fix. Be direct."

This review was written by Claude after building the validation in `src/validate.py`. Getting an outside statistician to read it before a pilot is still worth doing: the people who built a model are the worst at spotting its blind spots.

## What already holds up

- The model compares each sale with what the home itself predicts (size, lot, zip, month, trend, assessed value), so an agent isn't credited for working in a pricey zip or a hot month.
- No agent-controlled number (list price) feeds the price model, so the score can't be gamed by listing low.
- Shrinkage and ranges stop thin records from topping the list. 120 of 150 agents land in "not clearly different from average", which is honest.
- The stress test proves the model doesn't depend on which way selection bias points.

## 1. Real Santa Barbara assessed values won't behave like our fake ones

**The problem.** The model leans on assessed value to see home quality (median price error 3.2% with it, 5.0% without). In our fake data, assessed value is the true value plus or minus about 4.5%. In California it isn't. Under Proposition 13 the assessed value is reset to the purchase price when a home changes hands, then rises by at most 2% a year until it sells again ([Sacramento County Assessor](https://assessor.saccounty.gov/us/en/topics-a-to-z/prop13--and-real-property-assessment.html)). A Montecito home bought in 1995 can carry an assessed value a fraction of its market value.

**Why it's unfair.** Agents who list long-held homes (older owners, estates, family trusts) would sell homes the model thinks are cheap, and look like stars. Buyer agents on those homes would look like they overpaid. That is the hidden-quality bias from failure item 1, made much bigger, and it lands on agents by the kind of client they serve.

**Simplest fix.** Ask the brokerage for two more columns: year of the last sale and the last sale price. Use years since last sale next to assessed value, or bring the last sale price forward with a local price index. Before the pilot, add a Prop 13-style assessed value to the fake data and rerun the stress test to see how much it hurts.

## 2. The ranges ignore bias, so the busiest agents get overconfident calls

**The problem.** A 95% range is built from luck alone: how much an agent's score would bounce if their deals had gone a bit differently. More deals means a narrower range. But hidden home quality pushes busy agents' scores the same way deal after deal, and more deals don't shrink that push. Across 5 fake worlds, ranges for agents with 50+ deals caught the truth 84% of the time instead of 95%.

**Why it's unfair.** Busy agents are the ones leaders know and talk about. They are the most likely to be told "clearly below average" when the truth is "about average".

**Simplest fix.** Add a bias allowance to `config/scoring.json`: widen every range by about plus or minus 0.5 points (combined with the luck range, not added on top). In the fake data that brings busy agents back to about 95% and cuts clear calls from 14% of agents to 9%. On real data, size it with the first half vs second half check in `validate.py` and report it next to the scores.

## 3. Offer win rate can reward the client's wallet, not the agent

**The problem.** In real deals the highest offer usually wins. So an agent whose clients have more cash, or who tells clients to bid high, wins more offers. Our fake offers don't work that way: losing offers are picked by hidden win skill, not price. So the validation can't catch this, and the 0.80 rank agreement for win rates says nothing about it.

**Why it's unfair.** An agent who coaches clients to bid sensibly loses more offers and gets a lower win rate for doing right by them. The buyer score partly catches overpaying, but the two are separate numbers and the index adds them up with fixed weights.

**Simplest fix.** Until Phase 5 adds offer price vs list price, financing, and the number of competing offers, show win rate only beside the buyer score, and drop its weight in the index to 0. In Phase 5, make the fake offers depend on price so the trade-off exists in the test data.

## Smaller issues

- **Skill is assumed constant.** A new agent's early deals count against them forever. Weight recent deals more, or show a trend.
- **Dual agency and teams.** The fake data never has one agent on both sides or two agents sharing credit. Real data does. Flag those sales and decide the credit rule before the pilot.
- **The coaching index treats its three parts' errors as independent.** Seller and buyer scores share one regression and the same homes, so their errors are probably linked. Measure it with the bootstrap draws.
- **Shrinkage treats the estimated spread between agents as known.** That makes ranges for thin records a bit too narrow. The table already uses the wider of two ranges, which covers most of it.
- **The answer key came from the same people as the model.** Passing our own tests is necessary, not sufficient.
