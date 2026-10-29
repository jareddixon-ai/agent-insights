"""Phase 7: readiness check for a private copy (a brokerage running agent-insights on its own data).

Reads the brokerage's export through config/column_mapping.json and says, in plain sentences,
what is there, what is missing, and what that turns off. It never changes your files.

Privacy by design:
  - It refuses to run if the data folder sits inside this repo but is not git-ignored
    (otherwise one "commit all" would publish client data).
  - It reads only the columns named in the mapping. Names, addresses, emails, phones and notes
    in your export are never loaded, so they can't be printed or stored.
  - It prints counts and our own column names only, never a value from your files (not even agent IDs).

Run:  python src/readiness_check.py                       (uses config/column_mapping.json)
      python src/readiness_check.py --mapping config/column_mapping_demo.json
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from clean_data import clean_sales

ROOT = Path(__file__).resolve().parent.parent
MAPPING = ROOT / "config" / "column_mapping.json"

# What each of our columns is for. "required" = can't score without it.
SALES_REQUIRED = ["listing_id", "zip", "beds", "baths", "sqft", "lot_sqft", "year_built", "assessed_value",
                  "list_date", "sold_date", "sold_price", "listing_agent_id", "buyer_agent_id"]
SALE_HISTORY = ["prior_sale_date", "prior_sale_price"]
CONTEXT = ["original_list_price", "num_price_changes", "days_on_market"]
SALES_OPTIONAL = ["value_estimate", "condition_rating"]
AGENT_COLS = ["agent_id", "office", "team", "years_licensed"]
OFFER_COLS = ["listing_id", "buyer_agent_id", "offer_price", "outcome", "financing", "down_payment_pct",
              "inspection_contingency", "appraisal_contingency", "loan_contingency", "days_to_close"]

NUMBERS = ["beds", "baths", "sqft", "lot_sqft", "year_built", "assessed_value", "sold_price", "prior_sale_price",
           "original_list_price", "num_price_changes", "days_on_market", "value_estimate", "condition_rating",
           "years_licensed", "offer_price", "down_payment_pct", "inspection_contingency", "appraisal_contingency",
           "loan_contingency", "days_to_close"]
DATES = ["list_date", "sold_date", "prior_sale_date"]
IDS = ["listing_agent_id", "buyer_agent_id", "agent_id"]
# Header words that suggest personal details. Used only to COUNT such columns, never to read them.
PERSONAL = re.compile(r"name|email|e-mail|phone|mobile|cell|address|street|remark|note|comment|client|contact",
                      re.IGNORECASE)

MIN_SALES = 200          # below this the price model has too little to learn from: refuse to score
MOSTLY_EMPTY = 0.5       # a mapped column that is over half blank counts as missing
LEAK_MISS = 0.015        # a value estimate this close to the sale price was probably made with the price in hand


class UnsafeFolder(Exception):
    """The data folder could end up on GitHub."""


@dataclass
class Report:
    found: list = field(default_factory=list)       # what is there
    missing: list = field(default_factory=list)     # what is missing and what that turns off
    watch: list = field(default_factory=list)       # things to check before trusting results
    blockers: list = field(default_factory=list)    # why scoring can't run yet
    tables: dict = field(default_factory=dict)      # cleaned tables in our column names (kept in memory only)
    turned_off: set = field(default_factory=set)    # parts that can't be shown: buyer_scores, offer_score, ...

    @property
    def ready(self):
        return not self.blockers

    def text(self):
        out = ["READINESS CHECK (counts only: no values from your files are shown)", ""]
        for title, lines in [("What is there", self.found), ("What is missing, and what that turns off", self.missing),
                             ("Check before trusting results", self.watch), ("Why scoring can't run yet", self.blockers)]:
            if lines:
                out += [title + ":"] + [f"  - {line}" for line in lines] + [""]
        out.append("Ready to score: yes. Next: python src/run_private.py" if self.ready
                   else "Ready to score: no. Fix the items above, then run this check again.")
        return "\n".join(out)


# ---------- safety ----------
def check_folder_safe(folder, repo=ROOT):
    """Refuse a data folder inside the repo unless git ignores it (and everything in it)."""
    folder, repo = Path(folder).resolve(), Path(repo).resolve()
    try:
        rel = folder.relative_to(repo)
    except ValueError:
        return "outside the repo folder"
    probes = [str(rel / "any_new_file.csv")] + [str(p.relative_to(repo)) for p in folder.rglob("*") if p.is_file()]
    # git prints back each path it ignores. A tracked file is never "ignored", so it fails too.
    r = subprocess.run(["git", "-C", str(repo), "check-ignore", "--stdin"], input="\n".join(probes) + "\n",
                       capture_output=True, text=True)
    ignored = set(r.stdout.splitlines()) if r.returncode in (0, 1) else set()   # 128 = not a git repo
    if set(probes) - ignored:
        raise UnsafeFolder(
                f"Refusing to run: the data folder '{rel}' is inside the repo but git does not ignore it, so its "
                "files could be committed and published. Move your files to data/private/ (already git-ignored) "
                "or to a folder outside the repo, and point data_folder in the mapping there.")
    return "inside the repo, git-ignored"


# ---------- reading ----------
def load_mapping(path=MAPPING):
    return json.loads(Path(path).read_text())


def resolve_folder(mapping, data_folder=None):
    folder = Path(data_folder or mapping["data_folder"])
    return folder if folder.is_absolute() else ROOT / folder


def read_table(folder, spec, report, label):
    """Read only the mapped columns and rename them to ours. Returns (table, our columns present)."""
    path = folder / spec["file"]
    if not path.exists():
        return None, set()
    header = list(pd.read_csv(path, nrows=0).columns)
    wanted = {ours: theirs for ours, theirs in spec["columns"].items() if theirs}
    present = {ours: theirs for ours, theirs in wanted.items() if theirs in header}
    for ours, theirs in wanted.items():
        if ours not in present:
            report.missing.append(f"The mapping points our '{ours}' at a column your {label} file doesn't have. "
                                  "Check the spelling in the mapping.")
    unused = [h for h in header if h not in present.values()]
    if unused:
        personal = sum(bool(PERSONAL.search(h)) for h in unused)
        report.found.append(f"{label.capitalize()} file: {len(unused)} column(s) are not mapped and are never read"
                            + (f", including {personal} whose header looks like personal details (names, contact, address, notes)."
                               if personal else "."))
    table = pd.read_csv(path, usecols=list(present.values()), dtype=str, keep_default_na=False, na_values=[""])
    return table.rename(columns={theirs: ours for ours, theirs in present.items()}), set(present)


def to_numbers(df):
    for col in [c for c in NUMBERS if c in df]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$,%\s]", "", regex=True), errors="coerce")
    for col in [c for c in DATES if c in df]:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    for col in [c for c in IDS if c in df]:
        df[col] = df[col].fillna("").astype(str).str.strip().str.upper()
    return df


def usable(df, cols):
    """Columns that are present and not mostly blank."""
    return [c for c in cols if c in df and df[c].isna().mean() <= MOSTLY_EMPTY]


# ---------- the check ----------
def check(mapping, data_folder=None, repo=ROOT):
    """Return a Report. Raises UnsafeFolder before reading anything if the folder could be published."""
    report = Report()
    folder = resolve_folder(mapping, data_folder)
    if not folder.is_dir():
        report.blockers.append("The data folder doesn't exist yet. Create it, put your export files in it, "
                               "and set data_folder in the mapping.")
        return report
    where = check_folder_safe(folder, repo)
    report.found.append(f"Data folder is {where}: safe from being committed.")

    sales_check(folder, mapping, report)
    if "sales" in report.tables:
        agents_check(folder, mapping, report)
        offers_check(folder, mapping, report)
    return report


def sales_check(folder, mapping, report):
    raw, present = read_table(folder, mapping["sales"], report, "sales")
    if raw is None:
        report.blockers.append(f"No sales file ('{mapping['sales']['file']}'). Sales are the core of every score.")
        return
    lacking = [c for c in SALES_REQUIRED if c not in present]
    if lacking:
        report.blockers.append("The sales file has no column for: " + ", ".join(lacking)
                               + ". The model needs these to estimate what each home was worth (see docs/data_dictionary.md).")
        return

    df = to_numbers(raw)
    for c in SALE_HISTORY + CONTEXT + SALES_OPTIONAL:   # mostly-blank optional columns count as missing
        if c in df and c not in usable(df, [c]):
            report.missing.append(f"'{c}' is over half blank, so it is treated as missing.")
            df = df.drop(columns=c)
    if not all(c in df for c in SALE_HISTORY):
        df = df.drop(columns=[c for c in SALE_HISTORY if c in df])
    clean, rep = clean_sales(df)
    before = len(clean)
    bad = (clean[SALES_REQUIRED[:-1]].isna().any(axis=1)
           | (clean[["sqft", "lot_sqft", "sold_price", "assessed_value"]] <= 0).any(axis=1)
           | (clean["listing_agent_id"] == "") | clean["zip"].isin(["", "nan"]))
    clean = clean[~bad].reset_index(drop=True)
    # The model can't use a row with a blank in a feature column it uses (sale history, value estimate).
    model_extras = [c for c in SALE_HISTORY + SALES_OPTIONAL if c in clean]
    gaps = clean[model_extras].isna().any(axis=1) if model_extras else pd.Series(False, index=clean.index)
    if gaps.any():
        report.watch.append(f"{int(gaps.sum())} sales dropped for a blank in " + ", ".join(model_extras)
                            + " (for example new homes with no last sale). If that is many, map that column to null.")
        clean = clean[~gaps].reset_index(drop=True)
    report.found.append(f"{rep['rows_in']:,} sales rows read; {len(clean):,} usable after cleaning "
                        f"({rep['duplicates_dropped']} duplicates, {rep['unsold_dropped']} unsold, "
                        f"{before - len(clean)} with a key value missing or unreadable). "
                        f"Blank home features filled from the zip's typical home: {sum(rep['filled'].values())} cells.")

    # Size: how many clear calls to expect.
    listings = clean["listing_agent_id"].value_counts()
    median = float(listings.median()) if len(listings) else 0.0
    months = (clean["sold_date"].max() - clean["sold_date"].min()).days / 30.4 if len(clean) else 0
    report.found.append(f"{clean['listing_agent_id'].nunique()} listing agents, a median of {median:.0f} listings each; "
                        f"{(listings < 5).sum()} have under 5. Sales cover about {months:.0f} months.")
    if len(clean) < MIN_SALES:
        report.blockers.append(f"Only {len(clean)} usable sales. The price model needs at least {MIN_SALES} "
                               "to learn what homes are worth; with fewer, every score would be mostly noise.")
    if median < 5:
        report.watch.append("Most agents have under 5 listings, so almost every agent will be 'not clearly different "
                            "from average', and every card is a first look only.")
    else:
        report.watch.append(f"With a median of {median:.0f} listings per agent, expect most agents to be 'not clearly "
                            "different from average' (fake data: 135 of 150 at a median of 19). That is honest, "
                            "not a failure.")
    if months < 24:
        report.watch.append("Under 2 years of sales: the year-by-year trend will have one or two points, and "
                            "half-year periods will be mostly small samples.")
    no_buyer = (clean["buyer_agent_id"] == "").mean()
    if no_buyer > 0:
        report.found.append(f"{no_buyer:.0%} of sales have no buyer agent; those sales give no buyer-side credit.")

    # Agent IDs are shown to leaders on the dashboard, so they must be codes, not names or emails.
    ids = pd.concat([clean["listing_agent_id"], clean["buyer_agent_id"]])
    if ids.str.contains(r"\s|@", regex=True).any():
        report.blockers.append("Some agent IDs contain spaces or an @, so they may be names or emails. The dashboard "
                               "shows agent IDs to leaders: replace them with codes (like A001) before scoring.")

    # Optional parts, and what missing them turns off.
    if all(c in clean for c in SALE_HISTORY):
        report.found.append("Last sale date and price are there: the model can tell long-held homes (low tax values) "
                            "from cheaper homes.")
    else:
        report.missing.append("No last sale date and price. In California, tax values lag far behind for long-held "
                              "homes, so agents who list them look better than they are. In our fake data, dropping the "
                              "last sale nearly doubled the spread of seller scores and cut rank agreement with true "
                              "skill from 0.74 to 0.60. Scores still run, with that bias; county records have it.")
    if all(c in clean for c in CONTEXT):
        report.found.append("List price, price changes and days on market are there: speed and pricing context is on.")
    else:
        report.turned_off.add("speed_pricing")
        report.missing.append("Missing " + ", ".join(c for c in CONTEXT if c not in clean)
                              + ": no speed or pricing context, and no offers attracted.")
    if "value_estimate" in clean:
        miss = (clean["sold_price"] / clean["value_estimate"] - 1).abs().median()
        report.found.append(f"Independent value estimate is there (median miss against the sale price {miss:.1%}). "
                            "The price model will use it.")
        if miss < LEAK_MISS:
            report.watch.append("The value estimate sits suspiciously close to the sale price. A loan appraisal, made "
                                "after the contract with the price in hand, looks like this and hides every agent's "
                                "effect. Check who made it and when: it must be made before, and without, the price.")
        report.watch.append("Buyer scores stay context only until your own validation shows they work with this "
                            "estimate (config/scoring.json, 'validation'). Don't change that block by hand.")
    else:
        report.turned_off.add("buyer_scores")
        report.missing.append("No independent value estimate (like a pre-listing appraisal or broker price opinion "
                              "made before the sale): buyer scores stay context only, never 'clearly above or below'.")
    if "condition_rating" in clean:
        report.watch.append("A condition rating barely helped in testing. If the listing agent wrote it, drop it: "
                            "it can soak up the agent's own result.")
    report.tables["sales"] = clean


def agents_check(folder, mapping, report):
    sales = report.tables["sales"]
    in_sales = sorted(set(sales["listing_agent_id"]) | set(sales["buyer_agent_id"]) - {""})
    raw, present = read_table(folder, mapping["agents"], report, "agents")
    if raw is None or "agent_id" not in present:
        report.missing.append("No agents file with an agent ID: office and team filters won't work "
                              "(everyone shows as office 'Unknown').")
        agents = pd.DataFrame({"agent_id": in_sales})
    else:
        agents = to_numbers(raw).drop_duplicates("agent_id")
        extra = set(in_sales) - set(agents["agent_id"])
        if extra:
            report.watch.append(f"{len(extra)} agent(s) in the sales file are not in the agents file; they are added "
                                "with office 'Unknown'.")
            agents = pd.concat([agents, pd.DataFrame({"agent_id": sorted(extra)})], ignore_index=True)
        report.found.append(f"{len(agents)} agents in the roster.")
    for col, fill in [("office", "Unknown"), ("team", ""), ("years_licensed", np.nan)]:
        if col not in agents:
            if raw is not None:
                report.missing.append(f"No '{col}' column in the agents file"
                                      + (": office filter shows everyone as 'Unknown'." if col == "office" else
                                         ": team filter shows everyone as 'No team'." if col == "team" else
                                         ": shown as blank (not used in any score)."))
            agents[col] = fill
    agents["office"] = agents["office"].fillna("Unknown")
    agents["team"] = agents["team"].fillna("")
    report.tables["agents"] = agents[AGENT_COLS]


def offers_check(folder, mapping, report):
    raw, present = read_table(folder, mapping["offers"], report, "offers")
    if raw is None:
        report.turned_off.add("offer_score")
        report.missing.append("No offer data: no offer score and no offers attracted. Offers usually live in PDFs; "
                              "src/extract_offers.py turns them into a table without storing buyer names.")
        return
    lacking = [c for c in OFFER_COLS if c not in present]
    if lacking:
        report.turned_off.add("offer_score")
        report.missing.append("The offers file has no column for: " + ", ".join(lacking)
                              + ". No offer score: comparing offers fairly needs price and every term.")
        return
    o = to_numbers(raw)
    for col in ["outcome", "financing"]:
        spelled = {k.strip().lower(): v for k, v in mapping.get("values", {}).get(col, {}).items() if not k.startswith("_")}
        o[col] = o[col].fillna("").astype(str).str.strip().str.lower().map(spelled)
    unknown = o[["outcome", "financing"]].isna().any(axis=1) | o[OFFER_COLS[2:]].isna().any(axis=1)
    sales_ids = set(report.tables["sales"]["listing_id"].astype(str))
    o["listing_id"] = o["listing_id"].astype(str).str.strip()
    orphan = ~o["listing_id"].isin(sales_ids)
    o = o[~unknown & ~orphan].reset_index(drop=True)
    contests = o.groupby("listing_id").size()
    winners = o[o["outcome"] == "won"].groupby("listing_id").size()
    bad_contest = contests.index[winners.reindex(contests.index).fillna(0) != 1]
    o = o[~o["listing_id"].isin(bad_contest)].reset_index(drop=True)
    multi = int((o.groupby("listing_id").size() > 1).sum())
    report.found.append(f"{len(o):,} usable offers ({int(unknown.sum())} with an unreadable value, "
                        f"{int(orphan.sum())} on listings not in the sales file, {len(bad_contest)} listings dropped "
                        f"for not having exactly one winner). {multi} listings had 2+ offers.")
    if multi < 200:
        report.watch.append("Under 200 listings with 2+ offers: only those inform the offer score, so its ranges "
                            "will be wide.")
    report.watch.append("Offer score stays context only until it passes validation (it failed narrowly on fake data).")
    report.tables["offers"] = o[OFFER_COLS]


def main():
    parser = argparse.ArgumentParser(description="Readiness check for a private copy.")
    parser.add_argument("--mapping", default=str(MAPPING), help="column mapping file")
    parser.add_argument("--data", default=None, help="data folder (overrides data_folder in the mapping)")
    args = parser.parse_args()
    try:
        report = check(load_mapping(args.mapping), args.data)
    except UnsafeFolder as e:
        raise SystemExit(str(e))
    print(report.text())
    raise SystemExit(0 if report.ready else 1)


if __name__ == "__main__":
    main()
