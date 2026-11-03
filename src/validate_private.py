"""Phase 9: a self-check for a private copy. SELLER SIDE ONLY. Counts only in the report.

A brokerage has no answer key, so a private copy can't grade itself the way src/validate.py grades the fake
data. What it can do is a split-half check: score every agent twice, once on the first half of the sales
(by date) and once on the second half. If scores are mostly skill, an agent's two halves agree. The
agreement r gives two numbers (Spearman-Brown, the standard way to project reliability from half the data):
    reliability of the full record = 2r / (1 + r)
    implied accuracy               = square root of that reliability (a guess at how closely the scores
                                     follow true skill)

What it CAN say: the scores are consistent, not noise. This is the "split-half agreement of 0.5 or better"
success measure in the pilot proposal.
What it CAN'T say: that they are correct. A bias that repeats in both halves, like a hidden home quality
that one agent's listings keep having, looks exactly like skill. Consistent is not the same as valid.

The BUYER side is refused on purpose. On the fake data the same check guessed 0.52 for the buyer side while
the true accuracy was 0.00: hidden home quality repeats in both halves and the check can't tell it from skill.
A check that reports a comfortable number about a broken score is worse than no check.

Report: counts and summary statistics only. No agent code, listing code, date or price is printed or written.

Run:  python src/validate_private.py                                  (reads <data_folder>/standard/sales.csv)
      python src/validate_private.py --mapping config/column_mapping_demo.json
Writes <data_folder>/validation_report.txt next to the scored files (same private, git-ignored folder).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agent_model import fit_agent_effects
from readiness_check import MAPPING, UnsafeFolder, check_folder_safe, load_mapping, resolve_folder

TARGET = 0.5          # split-half agreement signed in week 1 of the pilot
MIN_LISTINGS = 5      # an agent needs this many listings in EACH half to be compared
MIN_AGENTS = 20       # fewer agents than this: too few to judge
Z95 = 1.96

BUYER_REFUSAL = ("Buyer side: REFUSED. Hidden home quality repeats in both halves and looks exactly like skill, so "
                 "this check can't tell a good buyer's agent from a lucky one. On the fake data it guessed 0.52 "
                 "while the true accuracy was 0.00. Buyer scores stay context only until an independent value "
                 "estimate made before the sale is in the data and the full validation passes.")


class BuyerSideRefused(Exception):
    pass


def split_half(sales, min_listings=MIN_LISTINGS, side="seller"):
    """The split-half numbers for the seller side. side='buyer' raises BuyerSideRefused."""
    if side != "seller":
        raise BuyerSideRefused(BUYER_REFUSAL)
    sold = pd.to_datetime(sales["sold_date"])
    early = (sold <= sold.median()).to_numpy()
    first, _, _ = fit_agent_effects(sales[early].reset_index(drop=True))
    second, _, _ = fit_agent_effects(sales[~early].reset_index(drop=True))
    halves = pd.concat([first["seller_effect"].rename("a"), first["listings"].rename("na"),
                        second["seller_effect"].rename("b"), second["listings"].rename("nb")], axis=1).dropna()
    enough = halves[(halves["na"] >= min_listings) & (halves["nb"] >= min_listings)]
    out = {"sales": int(len(sales)), "listing_agents": int(sales["listing_agent_id"].nunique()),
           "agents_compared": int(len(enough)), "min_listings": int(min_listings)}
    if len(enough) < MIN_AGENTS:
        out.update(r=np.nan, low=np.nan, high=np.nan, reliability=np.nan, accuracy=np.nan, verdict="too few")
        return out
    r = float(enough["a"].corr(enough["b"]))
    z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1 / np.sqrt(len(enough) - 3)
    low, high = float(np.tanh(z - Z95 * se)), float(np.tanh(z + Z95 * se))
    reliability = 2 * r / (1 + r) if r > -1 else np.nan
    accuracy = float(np.sqrt(max(reliability, 0)))
    verdict = "meets" if r >= TARGET else "below"
    out.update(r=r, low=low, high=high, reliability=float(reliability), accuracy=accuracy, verdict=verdict)
    return out


def build_report(stats):
    """Plain text for the pilot readout. Counts and summary numbers only."""
    lines = ["PRIVATE VALIDATION REPORT: seller side, split-half check", ""]
    lines.append(f"Sales used: {stats['sales']:,}. Listing agents: {stats['listing_agents']}. Agents with at least "
                 f"{stats['min_listings']} listings in each half: {stats['agents_compared']}.")
    if stats["verdict"] == "too few":
        lines += [f"Too few agents to judge (fewer than {MIN_AGENTS}). No number is reported, because a correlation "
                  "from a handful of agents is mostly luck.", ""]
    else:
        lines += [f"The two halves agree: r = {stats['r']:.2f} (95% range {stats['low']:.2f} to {stats['high']:.2f}).",
                  f"Projected reliability of the full record: {stats['reliability']:.2f}. "
                  f"Implied accuracy (a guess): {stats['accuracy']:.2f}.",
                  f"Pilot target: {TARGET:.2f} or better. Result: "
                  + ("MEETS the target." if stats["verdict"] == "meets" else "BELOW the target.")
                  + (f" The low end of the range ({stats['low']:.2f}) is below the target, so call it close, not settled."
                     if stats["verdict"] == "meets" and stats["low"] < TARGET else ""), ""]
    lines += ["What this shows: the seller scores are consistent from one half of the data to the other, so they are "
              "not mostly noise.",
              "What it cannot show: that they are correct. A bias that repeats in both halves, such as hidden home "
              "quality, looks the same as skill. Consistent is not the same as valid.", "",
              BUYER_REFUSAL, "",
              "This report holds counts and summary numbers only: no agent code, listing, date or price."]
    return "\n".join(lines)


def load_standard_sales(folder):
    """Read the sales table run_private.py wrote (our column names only)."""
    sales = pd.read_csv(Path(folder) / "standard" / "sales.csv", dtype={"listing_agent_id": str, "buyer_agent_id": str})
    sales["buyer_agent_id"] = sales["buyer_agent_id"].fillna("")
    return sales


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split-half self-check for a private copy (seller side only).")
    parser.add_argument("--mapping", default=str(MAPPING))
    parser.add_argument("--data", default=None, help="data folder (overrides data_folder in the mapping)")
    parser.add_argument("--side", default="seller", choices=["seller", "buyer"])
    parser.add_argument("--min-listings", type=int, default=MIN_LISTINGS)
    args = parser.parse_args(argv)
    if args.side == "buyer":
        raise SystemExit(BUYER_REFUSAL)
    folder = resolve_folder(load_mapping(args.mapping), args.data)
    try:
        check_folder_safe(folder)          # same rule as the readiness check: never write where one commit could publish it
    except UnsafeFolder as e:
        raise SystemExit(str(e))
    if not (folder / "standard" / "sales.csv").exists():
        raise SystemExit("No standard/sales.csv in the data folder. Run src/run_private.py first.")
    report = build_report(split_half(load_standard_sales(folder), args.min_listings))
    (folder / "validation_report.txt").write_text(report + "\n")
    print(report)
    print(f"\nSaved to {folder / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
