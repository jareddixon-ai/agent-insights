"""Step 3-0: stress test. Does the model still work when selection bias points the
OTHER way, and why is the buyer side weaker?

Selection bias = good agents don't get random homes. In the standard fake data, better
agents get PRICIER homes, so "average sold price" accidentally looks smart. Here we
rebuild the fake data three ways and grade every method against the answer key:
  +0.4  better agents get pricier homes (the standard data)
   0.0  homes are assigned at random
  -0.4  better agents get CHEAPER homes (the naive ranking should collapse)
Plus a diagnosis run where the model is allowed to see each home's TRUE value.
If that fixes the buyer side, the problem is hidden home quality, not the method.

Results are averaged over 5 seeds (5 different fake worlds), so one lucky draw can't fool us.
"""

import numpy as np
import pandas as pd

import generate_fake_data as gen
from agent_model import DATA, fit_agent_effects
from shrinkage import shrink

SEEDS = [42, 43, 44, 45, 46]
SCENARIOS = [  # (label, selection strength, model sees true home value?)
    ("Better agents get pricier homes (standard)", 0.4, False),
    ("Homes assigned at random", 0.0, False),
    ("Better agents get CHEAPER homes", -0.4, False),
    ("Diagnosis: standard, model sees true value", 0.4, True),
]


def rank_agree(a, b):
    """Rank correlation (Spearman): 1 = same order, 0 = unrelated, -1 = reversed."""
    a, b = a.dropna(), b.reindex(a.dropna().index)
    return a.rank().corr(b.rank())


def run_one(seed, strength, oracle):
    t = gen.generate(seed=seed, selection_strength=strength, quiet=True)
    sales, skills = t["sales"].copy(), t["skills"].set_index("agent_id")
    if oracle:  # only possible with fake data: hand the model the answer key for home value
        sales["assessed_value"] = t["homes"]["true_value"].to_numpy()
    effects, _, resid_sd = fit_agent_effects(sales)
    naive = sales.groupby("listing_agent_id")["sold_price"].mean()

    seller = shrink(effects["seller_effect"], effects["listings"], resid_sd)
    buyer = shrink(effects["buyer_effect"], effects["buyer_deals"], resid_sd)
    true_s = skills["seller_premium"] - skills["seller_premium"].mean()
    true_b = -(skills["buyer_savings"] - skills["buyer_savings"].mean())  # better buyer agent = lower price

    def caught(tab, truth):
        tr = truth.reindex(tab.index)
        return ((tr >= tab["low"]) & (tr <= tab["high"])).mean()

    return {
        "naive_seller": rank_agree(naive, true_s),
        "model_seller": rank_agree(seller["score"], true_s),
        "model_buyer": rank_agree(buyer["score"], true_b),
        "range_caught_seller": caught(seller, true_s),
        "range_caught_buyer": caught(buyer, true_b),
    }


def main():
    rows = []
    for label, strength, oracle in SCENARIOS:
        runs = pd.DataFrame([run_one(s, strength, oracle) for s in SEEDS])
        rows.append(runs.mean().rename(label))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("STRESS TEST (average of", len(SEEDS), "fake worlds). Rank agreement with true skill, 1 = perfect.")
    print("'range caught' = share of 95% ranges that contain the true skill (should be about 0.95).\n")
    print(out.round(2).to_string())
    out.round(3).to_csv(DATA / "stress_test.csv", index_label="scenario")
    print("\nSaved stress_test.csv")


if __name__ == "__main__":
    main()
