"""Phase 5 follow-up (backlog item 1): make the offer score's 95% ranges honest.

Problem (5-4): the fair offer score's ranges caught the truth only 87-88% of the time
(rule: 90%+), so the score kept weight 0.

Method, set BEFORE looking at any result:
  1. Calibrate on seeds 42-46 (the same 15 worlds as 5-4). Try two ways to widen:
       multiplier : range half-width = m x 1.96 x standard error
       allowance  : range half-width = 1.96 x sqrt(standard error^2 + a^2)  (like the seller side)
     For each, take the smallest setting on its grid that reaches 90%+ coverage in every
     kind of world. Keep the one with the narrower average range (less overclaiming of doubt).
  2. Confirm on FRESH seeds 47-51 with that setting frozen, using the unchanged 5-4 rule:
     in every kind of world, win rank 0.5+, buyer-savings lean within +/-0.15,
     coverage 90%+, clear calls 90%+ right.
  3. Weights change only if step 2 passes.

The slow part (fitting the contest model with a cross-validated penalty) runs once per
world and is cached in data/fake/offer_calibration/, so re-grading is instant.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import generate_fake_data as gen
import offer_metrics as om
from agent_model import DATA

CALIBRATE_SEEDS = [42, 43, 44, 45, 46]
CONFIRM_SEEDS = [47, 48, 49, 50, 51]
WORLDS = {0.4: "pricier", 0.0: "random", -0.4: "cheaper"}
RULE = {"win_rank": 0.5, "savings_rank": 0.15, "coverage": 0.90, "calls_right": 0.90}
MULTIPLIERS = np.round(np.arange(1.0, 2.001, 0.05), 2)
ALLOWANCES = np.round(np.arange(0.0, 1.001, 0.02), 2)   # in % of price
CACHE = DATA / "offer_calibration"   # one file per fitted world


def agent_rows(seed, strength, offers=None, skills=None):
    """One row per buyer agent in one world: estimate and standard error in % of price, and the truth."""
    if offers is None:
        t = gen.generate(seed=seed, selection_strength=strength, quiet=True)
        offers, skills = t["offers"], t["skills"]
    sk = skills.set_index("agent_id")
    fair = om.fair_offer_scores(offers)
    per_pct = fair.attrs["terms"]["price_pct"]
    truth = 100 * gen.WIN_SKILL_VALUE * (sk["win_skill"] - sk["win_skill"].reindex(fair.index).mean())
    return pd.DataFrame({"seed": seed, "world": WORLDS[strength], "score": fair["score"],
                         "est": fair["effect"] / per_pct, "se": fair["effect_se"] / per_pct,
                         "truth": truth.reindex(fair.index), "offers": fair["offers"],
                         "win_skill": sk["win_skill"].reindex(fair.index),
                         "buyer_savings": sk["buyer_savings"].reindex(fair.index),
                         "penalty": fair.attrs["penalty"]})


def build_raw(seeds):
    """Fit every world not yet cached (one small file per world, so two runs can share the work)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for seed in seeds:
        for strength, world in WORLDS.items():
            f = CACHE / f"{seed}_{world}.csv"
            if not f.exists():
                agent_rows(seed, strength).reset_index(names="agent_id").to_csv(f, index=False)
                print(f"  fitted seed {seed}, {world} world", flush=True)
            parts.append(pd.read_csv(f))
    return pd.concat(parts, ignore_index=True)


def half_width(d, form, value):
    if form == "multiplier":
        return value * om.Z95 * d["se"]
    return om.Z95 * np.sqrt(d["se"] ** 2 + value ** 2)


def grade(d, form="multiplier", value=1.0):
    """The 5-4 rule metrics, averaged over seeds within each kind of world."""
    half = half_width(d, form, value)
    d = d.assign(half=half, covered=(d["truth"] - d["est"]).abs() <= half,
                 clear=(d["est"] - half > 0) | (d["est"] + half < 0))
    d["right"] = np.sign(d["est"]) == np.sign(d["truth"])
    rank = lambda g, col: g["score"].rank().corr(g[col].rank())
    per_world = d.groupby(["world", "seed"], sort=False).apply(lambda g: pd.Series({
        "win_rank": rank(g, "win_skill"), "savings_rank": rank(g, "buyer_savings"),
        "coverage": g["covered"].mean(), "clear_calls": g["clear"].sum(),
        "calls_right": g.loc[g["clear"], "right"].mean(), "avg_half_width": g["half"].mean()}),
        include_groups=False)
    return per_world.groupby("world", sort=False).mean()


def passes(out):
    ok = ((out["win_rank"] >= RULE["win_rank"]) & (out["savings_rank"].abs() <= RULE["savings_rank"])
          & (out["coverage"] >= RULE["coverage"]) & (out["calls_right"] >= RULE["calls_right"]))
    return bool(ok.all()), ok


def calibrate(d):
    """Smallest setting per form reaching 90%+ coverage in every kind of world; keep the narrower form."""
    best = {}
    for form, grid in [("multiplier", MULTIPLIERS), ("allowance", ALLOWANCES)]:
        for value in grid:
            out = grade(d, form, value)
            if (out["coverage"] >= RULE["coverage"]).all():
                best[form] = (float(value), out["avg_half_width"].mean())
                break
    form = min(best, key=lambda f: best[f][1])
    return form, best[form][0], best


def coverage_by_offers(d, form, value):
    half = half_width(d, form, value)
    bins = pd.cut(d["offers"], [0, 10, 30, 80, 10_000], labels=["under 10", "10-29", "30-79", "80+"])
    return pd.DataFrame({"before": (d["truth"] - d["est"]).abs() <= om.Z95 * d["se"],
                         "after": (d["truth"] - d["est"]).abs() <= half}).groupby(bins, observed=True).mean()


def main():
    pd.set_option("display.width", 200)
    raw = build_raw(CALIBRATE_SEEDS + CONFIRM_SEEDS)
    cal, conf = raw[raw["seed"].isin(CALIBRATE_SEEDS)], raw[raw["seed"].isin(CONFIRM_SEEDS)]

    print("\nSTEP 1: CALIBRATE on seeds 42-46")
    print("Before widening:\n", grade(cal).round(3).to_string())
    form, value, best = calibrate(cal)
    for f, (v, w) in best.items():
        print(f"  {f}: smallest setting reaching 90% everywhere = {v} (average half-width {w:.2f}% of price)")
    print(f"Chosen: {form} = {value}")
    print("After widening:\n", grade(cal, form, value).round(3).to_string())
    print("Coverage by offers made (calibration worlds):\n", coverage_by_offers(cal, form, value).round(3).to_string())

    print("\nSTEP 2: CONFIRM on fresh seeds 47-51 (setting frozen)")
    out = grade(conf, form, value)
    print(out.round(3).to_string())
    ok, by_world = passes(out)
    print("Rule:", RULE)
    print("Passes in each kind of world:", by_world.to_dict())
    print("Decision: offer score", "EARNS weight (seller 0.5, offer 0.2)" if ok else "stays context only")

    summary = pd.concat({"calibrate_before": grade(cal), "calibrate_after": grade(cal, form, value),
                         "confirm_before": grade(conf), "confirm_after": out}, names=["step"])
    summary.round(4).to_csv(DATA / "offer_calibration.csv")
    json.dump({"form": form, "value": value, "passed": ok}, open(DATA / "offer_calibration.json", "w"))
    return form, value, ok


if __name__ == "__main__":
    main()
