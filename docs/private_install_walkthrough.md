# Installing agent-insights on a brokerage machine

For the person at the brokerage who sets it up (an office manager, an operations lead, or the broker). About an hour for the first run, most of it waiting. Rehearsed end to end on the messy fake export (renamed columns, text prices, duplicates, unsold rows, fake names and emails that are never read). No real data was used.

The promises this walkthrough keeps: it runs on a machine you control, it reads only the columns you map, it never needs client names, contact details or notes, agent codes are random codes with the key kept by you, and nothing leaves the machine.

## Before any data moves
1. **Sign the NDA and the data agreement.** Your MLS rules decide what data you may share and with whom. Check them first.
2. **Turn on full-disk encryption** (FileVault on a Mac, BitLocker on Windows). Deleting a file on a modern drive does not guarantee the bytes are gone. Encryption keeps deleted data unreadable.
3. **Use random agent codes** (A001, A002 and so on). Keep the key that links codes to names in your own files. Never put names or emails in the agent column. The readiness check blocks IDs with spaces or an @.
4. **Decide what your agents are told.** That is your decision, and this tool does not give legal advice. See `docs/builder_vs_agent_note.md` for questions to take to your own lawyer.

## Install (about 10 minutes)
5. Open a terminal on the machine. Get a copy of the code: `git clone https://github.com/jareddixon-ai/agent-insights` and `cd agent-insights`.
6. Make an isolated Python environment (a private folder of packages for this one tool): `python3.12 -m venv .venv`, then `source .venv/bin/activate`.
7. Install the exact package versions: `pip install -r requirements.txt`.

## Your data (about 20 minutes)
8. Make the private folder: `mkdir data/private`. It is already git-ignored, so one "commit all" cannot publish it.
9. Your staff copy the export files into `data/private/`: a closed sales export (required), an agent roster (optional), offers (optional).
10. Map your columns in `config/column_mapping.json`: our name on the left, your exact header on the right, `null` for anything you don't have. Columns you don't list are never read. `config/column_mapping_demo.json` is a filled-in example.
11. Run the readiness check: `python src/readiness_check.py`. It prints counts only. On the messy fake export it said: 4,800 of 5,050 rows usable, 149 listing agents at a median of 19 listings each, no last sale date (so California tax values mislead; ask the county records for it), and no value estimate (so buyer scores stay context only). Fix what it blocks, then run it again.

## Score and check (about 10 minutes)
12. Score: `python src/run_private.py`. It stores only the columns it uses in `data/private/standard/`, writes scores to `data/private/scored/`, and prints the dashboard command. On the fake export, 200 resamples took about a minute. The full 1,000 take a few minutes.
13. Check the scores against themselves: `python src/validate_private.py`. Seller side only. On the fake export: the two halves agreed at r = 0.59 with 104 agents compared, which meets the 0.50 target. It cannot show the scores are correct, only that they are consistent. It refuses the buyer side on purpose.
14. Open the dashboard with the command step 12 printed (`AGENT_INSIGHTS_DATA="..." streamlit run app/coaching_app.py`). It opens on the team overview. Pick an agent in the sidebar and open the coaching brief before a one-on-one.

## At the end
15. Fill the owner readout: `python src/make_readout.py`.
16. Delete everything: `python src/delete_private.py` (a dry run that lists every file), then `--confirm` while your contact watches. Work through the hand checklist and sign the letter in `docs/deletion_letter_template.md`.

## If something fails
- "Refusing to run: the data folder ... git does not ignore it": move the files to `data/private/` or a folder outside the repo.
- "Under 200 usable sales blocks scoring": too little data for the price model. Add more months.
- Agent IDs contain spaces or an @: replace them with codes.
