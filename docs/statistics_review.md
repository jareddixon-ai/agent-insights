# Statistics review (Phase 4, updated through Phase 5)

Prompt used: "Act as a skeptical statistician reviewing agent-insights. Find the three biggest ways these agent scores could be unfair or wrong. For each one, give the simplest fix. Be direct."

This review was written by Claude after building the validation in `src/validate.py`. Getting an outside statistician to read it before a pilot is still worth doing: the people who built a model are the worst at spotting its blind spots.

## Update after Phase 5

- **Correction: "hidden quality, fix it with condition data" was only half right.** A 1-to-5 condition rating, even a perfect one, moves buyer rank agreement from 0.17 to 0.20 (`src/condition_test.py`). In California data the last sale price already carries most of a home's lasting quality. What is still hidden is what changed since that sale, luck in that sale, and the price model's shape: it is wrong at the cheap and expensive ends, and selection bias lines that error up with skill. Knowing every part of value but in the wrong shape gives 0.54; true value gives 0.87. What works is an independent estimate of what the home is worth when it lists: about 4% typical miss or better (about 3% median miss against the sale price) clears the bar in every kind of world, and also fixes the seller score's lean on selection bias (0.80 / 0.75 / 0.75 instead of 0.71 / 0.58 / 0.31).
- **New trap: "helpful" data that contains the answer.** A loan appraisal is made after the contract, with the price in hand. In the test it looks like the best estimate of all (1% median miss against the sale price) but it soaks up half of each agent's effect, so almost no agent is called clearly different. A condition rating written by the listing agent does the same to the seller score (0.71 to 0.55 in the standard world, 0.31 to 0.00 when good agents get cheaper homes).
- **Issue 3 (offer win rate): fixed with a different metric, not yet turned on.** Fake offers now depend on price and terms, and the raw win rate indeed rewards overpaying (rank agreement -0.42 with buyer savings). The fair offer score compares offers on the same home at the same price and terms (a conditional logit, the standard model for 1 winner out of N). Hidden home quality cancels because every offer in a contest is on the same home. Rank agreement with hidden win skill 0.89 (raw: 0.46), 98% of its clear calls right. It still missed the rule set before testing: ranges catch the truth 87% to 88% (rule 90%), and it leans slightly toward clients who pay less (up to 0.16, rule 0.15). Weight stays 0 until wider ranges pass on fresh worlds.
- **Smaller issue measured: seller and offer score errors are not linked** (correlation 0.00 to 0.04), so the index's independence assumption holds for that pair. Seller and buyer are still unmeasured.

## Update after Phase 4.5

- **Issue 1 (California tax values): fixed in the fake data and the model.** Assessed values now follow Prop 13, and the model uses the last sale date and price. Without them, estate specialists looked 3.0 points better than they are; with them, 0.0. The cost is real, though: buyer rank agreement fell from 0.55 to 0.17, because county records show so little about condition. See `src/sale_history_test.py` and `docs/sale_history.png`.
- **Issue 2 (ranges ignore bias): allowance added**, 0.008 in `config/scoring.json` (about 0.8 points of bias, one standard deviation). Busy agents are now caught 93% (seller) and 97% (buyer) of the time. Correction to the fix below: the first half vs second half check can NOT size this allowance. A bias that repeats in both halves is invisible to it (it guessed buyer accuracy at 0.52 when the truth was 0.00). On real data the allowance is an assumption, so it lives in the config where anyone can see it.
- **Issue 3 (offer win rate): weight set to 0.** Shown as context only until Phase 5 makes fake offers depend on price.
- **New issue found and fixed: the spread between agents could collapse to zero.** The old estimate (spread of raw scores minus average noise) was dominated by agents with 1 or 2 deals. In 2 of 5 fake worlds it said buyer agents don't differ at all, which pulled every buyer score to exactly 0 with almost no range (17% to 19% of ranges caught the truth). Shrinkage now uses the meta-analysis estimate (DerSimonian and Laird), where busy agents count more. This also explains the Phase 4 puzzle about narrow buyer ranges.
- **New decision: buyer scores are context only.** With California records they match true skill at 0.17, and their few clear calls were right only about half the time.

The original review follows, unchanged except where marked.

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

**Simplest fix.** Add a bias allowance to `config/scoring.json`: widen every range to allow for about 0.5 points of bias (combined with the luck range, not added on top). In the fake data that brings busy agents back to about 95% and cuts clear calls from 14% of agents to 9%. ~~On real data, size it with the first half vs second half check in `validate.py`.~~ (Wrong: see the update above.)

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
