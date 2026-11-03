# Pilot readout for the owner: {{DATE}}

Fake-data wording stays in the repo. On a real pilot, `python src/make_readout.py` fills this into the brokerage's private folder, and nothing here is copied back.

{{DISCLOSURE}}

## What we did
- Ran agent-insights on the brokerage's own machine, on the columns the brokerage mapped. No client names, contact details or notes were read.
- Agent codes are random codes. The brokerage holds the key.
- The broker used the coaching briefs before one-on-one conversations. The tool shows nothing to agents.

## Success measures (signed in week 1, before any scores)

| Measure | How it is measured | Target | Result | Met? |
|---|---|---|---|---|
| The data is ready | The readiness check says ready, with last sale date and price for most homes | Ready, with sale history | | |
| The scores agree with themselves | `validate_private.py`: split-half agreement, seller side | 0.50 or better | see below | |
| No clear call comes from a data error | The broker checks every Strength and Room to grow against the source records, using the "this sale looks wrong" tag | None | | |
| The briefs are useful | The broker rates each brief they used before a one-on-one | At least 7 of 10 rated useful | | |
| Privacy held | Nothing left the machine, only mapped columns were read, deletion done with a signed letter | Yes | | |

## Split-half check (seller side only, counts only)
{{SPLIT_HALF}}

This shows the seller scores are consistent from one half of the data to the other. It cannot show they are correct: a bias that repeats in both halves, like hidden home quality, looks the same as skill. The buyer side is refused, because the check is fooled there.

## What the scores can and cannot say
- Most agents will be "not clearly different from average". That is the honest answer, not a failure.
- A range that includes 0 means there is nothing to act on yet.
- An agent with under 5 sales shows "building a track record", not a score.
- The tool is never used for pay, commission splits, hiring or firing.

## Decision
Continue, change, or stop. If stopped early, deletion happens within 5 business days, with the brokerage's contact watching, and a signed letter.
