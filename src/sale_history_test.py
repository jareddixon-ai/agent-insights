"""Step 4.5: what California tax values do to the scores, and how much sale history wins back.

California's Prop 13 sets a home's assessed value to its purchase price, then lets it rise at
most 2% a year until the home sells again. Prices rose faster than that, so a home owned for
25 years carries an assessed value far below what it is worth. Until Phase 4.5 our fake
assessed values were simply the true value plus or minus about 4.5%, which no California
county produces.

We rebuild the fake data in 5 worlds (seeds) and compare four settings:
  1. Phase 4 fake data: assessed value close to market value (not how California works)
  2. California assessed values, model ignores when the home last sold
  3. California assessed values, model knows the last sale date and price (the default now)
  4. Diagnosis: the model sees each home's TRUE value (only possible in fake data)

"Estate specialist bias" = how far estate specialists' seller scores miss, minus how far
everyone else's miss, in percentage points. These agents mostly list long-held homes, which
is about their clients, not their skill. A fair model puts this near 0.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_fake_data as gen
from agent_model import DATA, fit_agent_effects
from shrinkage import shrink

DOCS = Path(__file__).resolve().parent.parent / "docs"
SEEDS = [42, 43, 44, 45, 46]
SETTINGS = [
    "Phase 4 fake data\n(assessed value near\nmarket value)",
    "California tax values,\nmodel ignores\nsale history",
    "California tax values,\nmodel uses last sale\n(default now)",
    "Diagnosis: model\nsees TRUE value",
]
SHORT = ["Phase 4\ndata", "California,\nno history", "California,\nlast sale", "True\nvalue"]
HISTORY = ["prior_sale_date", "prior_sale_price"]


def score_world(sales, skills, estate):
    """Fit, shrink, and grade one fake world against the answer key."""
    effects, _, resid_sd = fit_agent_effects(sales)
    seller = shrink(effects["seller_effect"], effects["listings"], resid_sd)
    buyer = shrink(effects["buyer_effect"], effects["buyer_deals"], resid_sd)
    true_s = (skills["seller_premium"] - skills["seller_premium"].mean()).reindex(seller.index)
    true_b = -(skills["buyer_savings"] - skills["buyer_savings"].mean()).reindex(buyer.index)
    miss = seller["score"] - true_s
    is_estate = estate.reindex(seller.index).fillna(False).astype(bool)
    caught = lambda tab, tr: ((tr >= tab["low"]) & (tr <= tab["high"])).mean()
    return {
        "seller_rank_agreement": seller["score"].rank().corr(true_s.rank()),
        "buyer_rank_agreement": buyer["score"].rank().corr(true_b.rank()),
        "leftover_error_per_sale": resid_sd,
        "estate_specialist_bias_pts": (miss[is_estate].mean() - miss[~is_estate].mean()) * 100,
        "range_caught_seller": caught(seller, true_s),
        "range_caught_buyer": caught(buyer, true_b),
    }


def run(seeds=SEEDS):
    rows = []
    for seed in seeds:
        ca = gen.generate(seed=seed, quiet=True)                         # California (default)
        old = gen.generate(seed=seed, assessed_style="market", quiet=True)
        skills, estate = ca["skills"].set_index("agent_id"), ca["estate"]
        oracle = ca["sales"].drop(columns=HISTORY).assign(assessed_value=ca["homes"]["true_value"].to_numpy())
        worlds = [old["sales"], ca["sales"].drop(columns=HISTORY), ca["sales"], oracle]
        for label, sales in zip(SETTINGS, worlds):
            rows.append({"setting": label, "seed": seed, **score_world(sales, skills, estate)})
    out = pd.DataFrame(rows).drop(columns="seed").groupby("setting", sort=False).mean()
    return out


def chart(out, path):
    from validate import BLUE, ORANGE, INK2, AXIS  # same look as the Phase 4 charts

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    x, w = np.arange(len(out)), 0.38
    labels = [s for s in out.index]
    for off, col, color, name in [(-w / 2 - 0.01, "seller_rank_agreement", BLUE, "Seller side"),
                                  (w / 2 + 0.01, "buyer_rank_agreement", ORANGE, "Buyer side")]:
        vals = out[col].to_numpy()
        a1.bar(x + off, vals, w, color=color, label=name)
        for xi, v in zip(x + off, vals):
            a1.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK2)
    a1.axhline(0, color=AXIS, lw=1)
    a1.set_ylim(min(-0.3, out["buyer_rank_agreement"].min() - 0.1), 1.05)
    a1.set_xticks(x, labels, fontsize=8)
    a1.set_ylabel("Rank agreement with true skill")
    a1.set_title("How well scores match true skill")
    a1.legend(loc="upper left")

    vals = out["estate_specialist_bias_pts"].to_numpy()
    a2.bar(x, vals, 0.55, color=BLUE)
    for xi, v in zip(x, vals):
        a2.text(xi, v + (0.05 if v >= 0 else -0.05), f"{round(v, 1) + 0:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK2)
    a2.axhline(0, color=AXIS, lw=1)
    a2.set_ylim(min(-0.5, vals.min() - 0.4), max(vals.max() * 1.15, 1))
    a2.set_xticks(x, SHORT, fontsize=8)
    a2.set_ylabel("Percentage points")
    a2.set_title("Estate specialists flattered by")
    fig.suptitle("California tax values (Prop 13): what they cost, and what sale history wins back "
                 "(5 fake worlds)", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main():
    out = run()
    pd.set_option("display.width", 160)
    print("PROP 13 TEST (average of", len(SEEDS), "fake worlds)")
    print(out.rename(index=lambda s: s.replace("\n", " ")).round(2).to_string())
    out.rename(index=lambda s: s.replace("\n", " ")).round(4).to_csv(DATA / "sale_history_test.csv")
    DOCS.mkdir(exist_ok=True)
    chart(out, DOCS / "sale_history.png")
    print("\nSaved sale_history_test.csv and docs/sale_history.png")


if __name__ == "__main__":
    main()
