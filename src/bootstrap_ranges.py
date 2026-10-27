"""Step 3-3: bootstrap ranges, a small-sample flag, and fixed-k shrinkage for comparison.

Bootstrap: make 1,000 "alternate histories" of the brokerage by drawing 5,000 sales at
random WITH replacement from the real 5,000 (some sales repeat, some drop out). Refit the
model on each one. If an agent's score swings a lot across histories, we can't pin it
down. The middle 95% of those 1,000 scores is the bootstrap range.

Why also keep the formula range from 2C? Two methods that agree give more confidence.
Where they disagree, the bootstrap is usually more honest for busy agents and LESS
honest for agents with under 5 sales (with 2 sales, the resamples barely vary), so we flag them.

What neither range can see: bias. If hidden home quality pushes a score the wrong way
(the buyer side, see 3-0), every resample is pushed the same way. Ranges measure luck,
not systematic error.

Fixed-k shrinkage (the original plan): score = n / (n + k) x raw score, with k = 10.
Ours picks k from the data instead. We print both so you can compare.
"""

import sys

import numpy as np
import pandas as pd

from agent_model import DATA, fit_agent_effects
from shrinkage import shrink

N_RESAMPLES = 1000
FIXED_K = 10
SMALL_SAMPLE = 5  # flag agents with fewer deals than this


def shrunk_scores(sales):
    """Fit the model and shrink both sides. Returns two Series of shrunk scores."""
    effects, _, resid_sd = fit_agent_effects(sales)
    seller = shrink(effects["seller_effect"], effects["listings"], resid_sd)
    buyer = shrink(effects["buyer_effect"], effects["buyer_deals"], resid_sd)
    return seller, buyer


def bootstrap(sales, n_resamples=N_RESAMPLES, seed=42):
    """Refit on resampled sales. Returns one table per side: rows = resamples, columns = agents."""
    rng = np.random.default_rng(seed)
    seller_draws, buyer_draws = [], []
    for i in range(n_resamples):
        idx = rng.integers(0, len(sales), len(sales))
        s, b = shrunk_scores(sales.iloc[idx].reset_index(drop=True))
        seller_draws.append(s["score"])
        buyer_draws.append(b["score"])
        if (i + 1) % 100 == 0:
            print(f"  resample {i + 1} of {n_resamples}", flush=True)
    return pd.DataFrame(seller_draws).reset_index(drop=True), pd.DataFrame(buyer_draws).reset_index(drop=True)


def summarize(full, draws):
    """Combine the full-data score with its bootstrap and formula ranges."""
    out = full[["raw", "score", "deals", "weight"]].copy()
    out["boot_low"] = draws.quantile(0.025).reindex(out.index)
    out["boot_high"] = draws.quantile(0.975).reindex(out.index)
    out["boot_sd"] = draws.std().reindex(out.index)
    out["formula_low"], out["formula_high"] = full["low"], full["high"]
    out["in_resamples"] = draws.notna().mean().reindex(out.index).fillna(0)  # share of histories the agent appears in
    out["under_5_deals"] = out["deals"] < SMALL_SAMPLE
    out["fixed_k_score"] = out["deals"] / (out["deals"] + FIXED_K) * out["raw"]
    return out


def grade(t, truth, label):
    tr = (truth - truth.mean()).reindex(t.index)
    caught = lambda lo, hi, m: ((tr >= t[lo]) & (tr <= t[hi]))[m].mean()
    big, small = ~t["under_5_deals"], t["under_5_deals"]
    implied_k = (t["deals"] * (1 - t["weight"]) / t["weight"]).median()
    err = lambda col: (t[col] - tr).abs().mean() * 100
    print(f"{label}: 95% range caught the truth | bootstrap {caught('boot_low','boot_high',big | small):.0%}"
          f" (5+ deals {caught('boot_low','boot_high',big):.0%}, under 5 deals {caught('boot_low','boot_high',small):.0%})"
          f" | formula {caught('formula_low','formula_high',big | small):.0%}")
    print(f"   error (pct points): our shrinkage {err('score'):.2f} (acts like k = {implied_k:.0f})"
          f" | fixed k = {FIXED_K}: {err('fixed_k_score'):.2f} | no shrinkage {err('raw'):.2f}")


def main(n_resamples=N_RESAMPLES):
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    skills = pd.read_csv(DATA / "true_skills.csv").set_index("agent_id")
    print(f"BOOTSTRAP: refitting the model on {n_resamples} resampled histories (about 2-3 minutes)...")
    seller_full, buyer_full = shrunk_scores(sales)
    seller_draws, buyer_draws = bootstrap(sales, n_resamples)
    seller = summarize(seller_full, seller_draws)
    buyer = summarize(buyer_full, buyer_draws)

    grade(seller, skills["seller_premium"], "Seller side")
    grade(buyer, -skills["buyer_savings"], "Buyer side ")
    print(f"Agents flagged with under {SMALL_SAMPLE} deals: {int(seller['under_5_deals'].sum())} sellers,"
          f" {int(buyer['under_5_deals'].sum())} buyer agents")

    both = pd.concat({"seller": seller, "buyer": buyer}, names=["side", "agent_id"])
    both.round(4).to_csv(DATA / "bootstrap_scores.csv")
    print("Saved bootstrap_scores.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_RESAMPLES)
