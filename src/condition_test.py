"""Step 5-0: how good must condition data be before buyer scores are usable?

Hidden home quality (condition, finishes) has been the main suspect on both sides. County
records can't see it, and good agents tend to get nicer homes (selection bias), so the model
gives agents credit or blame for the house. We test two kinds of extra data, each at several
accuracy levels, in 3 kinds of world (good agents get pricier homes, homes assigned at random,
good agents get cheaper homes), 5 fake worlds (seeds) each:

  A. A 1-to-5 condition rating (an inspector's grade). Accuracy = how closely it tracks
     the hidden quality (0 = a guess, 1 = a perfect read).
  B. An independent estimate of what the home is worth when it lists (a pre-listing
     appraisal). Accuracy = its typical miss in %.

Bar for "usable", set BEFORE looking at results (so we can't move the goalposts):
  buyer rank agreement at least 0.5 in all three kinds of world, AND
  at least 90% of its "clearly above/below average" calls right in every kind of world.

Two leakage runs, where the extra data quietly contains the agent's own result:
  - the listing agent writes the condition rating and better sellers rate their own listings
    higher (puffery);
  - the value estimate is a loan appraisal, made after the contract with the price in hand,
    so it sits halfway between the true value and the sale price.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_fake_data as gen
from agent_model import DATA, fit_agent_effects
from score_agents import Z95, load_config
from shrinkage import shrink

DOCS = Path(__file__).resolve().parent.parent / "docs"
SEEDS = [42, 43, 44, 45, 46]
WORLDS = {0.4: "Good agents get pricier homes (standard)", 0.0: "Homes assigned at random",
          -0.4: "Good agents get cheaper homes"}
RATING_ACCURACY = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
ESTIMATE_ERROR = [0.10, 0.06, 0.04, 0.03, 0.02, 0.01]
PUFFERY = 0.5   # listing agent rates own listings up by half a step per standard deviation of skill
ANCHOR = 0.5    # loan appraisal sits halfway between true value and the sale price
BAR_RANK, BAR_CALLS = 0.5, 0.90


def runs():
    """(kind, level, leak) for every version of the data we test."""
    out = [("none", 0.0, False)]
    out += [("rating", a, False) for a in RATING_ACCURACY] + [("rating", 0.8, True)]
    out += [("estimate", e, False) for e in ESTIMATE_ERROR] + [("estimate", 0.02, True)]
    return out


def add_data(t, kind, level, leak, seed):
    if kind == "rating":
        return gen.add_condition_rating(t["sales"], t["homes"], t["skills"], level, seed, PUFFERY if leak else 0.0)
    if kind == "estimate":
        return gen.add_value_estimate(t["sales"], t["homes"], level, seed, ANCHOR if leak else 0.0)
    return t["sales"]


def grade_side(tab, truth, allowance):
    """Rank agreement, and how many 'clear' calls are right, using the table's bias allowance."""
    t = truth.reindex(tab.index)
    half = np.sqrt((tab["high"] - tab["score"]) ** 2 + (Z95 * allowance) ** 2)
    clear = (tab["score"] - half > 0) | (tab["score"] + half < 0)
    right = (np.sign(tab["score"]) == np.sign(t))[clear]
    return tab["score"].rank().corr(t.rank()), clear.sum(), right.mean() if clear.any() else np.nan


def score_world(sales, skills, allowance):
    effects, _, resid_sd = fit_agent_effects(sales)
    seller = shrink(effects["seller_effect"], effects["listings"], resid_sd)
    buyer = shrink(effects["buyer_effect"], effects["buyer_deals"], resid_sd)
    true_s = skills["seller_premium"] - skills["seller_premium"].mean()
    true_b = -(skills["buyer_savings"] - skills["buyer_savings"].mean())   # lower price = better buyer agent
    s_rank, s_n, s_right = grade_side(seller, true_s, allowance)
    b_rank, b_n, b_right = grade_side(buyer, true_b, allowance)
    return {"seller_rank": s_rank, "seller_clear": s_n, "seller_right": s_right,
            "buyer_rank": b_rank, "buyer_clear": b_n, "buyer_right": b_right, "leftover_error": resid_sd}


def run(seeds=SEEDS, worlds=WORLDS, versions=None):
    allowance = load_config().get("bias_allowance", 0)
    rows = []
    for strength in worlds:
        for seed in seeds:
            t = gen.generate(seed=seed, selection_strength=strength, quiet=True)
            skills = t["skills"].set_index("agent_id")
            for kind, level, leak in versions or runs():
                sales = add_data(t, kind, level, leak, seed)
                if kind == "rating":   # how closely the rounded 1-5 rating really tracks quality
                    measured = np.corrcoef(sales["condition_rating"], t["homes"]["quality"])[0, 1]
                elif kind == "estimate":   # typical miss vs true value, in %
                    worth = t["homes"]["true_value"].to_numpy() * gen.market_factor(sales["list_date"])
                    measured = np.std(np.log(sales["value_estimate"] / worth))
                else:
                    measured = 0.0
                # What a brokerage CAN measure: median miss of the estimate vs the sale price
                vs_sale = (np.median(np.abs(np.log(sales["value_estimate"] / sales["sold_price"])))
                           if kind == "estimate" else np.nan)
                rows.append({"kind": kind, "level": level, "leak": leak, "world": strength, "seed": seed,
                             "measured": measured, "median_miss_vs_sale": vs_sale,
                             **score_world(sales, skills, allowance)})
    raw = pd.DataFrame(rows)
    return raw.drop(columns="seed").groupby(["kind", "level", "leak", "world"], sort=False).mean().reset_index()


def verdict(out, kind):
    """Least demanding level of this kind of data where buyer scores clear the bar in every world.
    For ratings that is the LOWEST accuracy, for estimates the LARGEST error."""
    honest = out[(out["kind"] == kind) & ~out["leak"]]
    ok = honest.groupby("level").apply(
        lambda g: (g["buyer_rank"] >= BAR_RANK).all() and (g["buyer_right"].fillna(0) >= BAR_CALLS).all(),
        include_groups=False)
    passing = ok[ok].index
    if not len(passing):
        return None
    return float(passing.min() if kind == "rating" else passing.max())


def why_ratings_fail(seeds=SEEDS):
    """Standard world. Hand the model ever more of the truth to see what the buyer side is missing.
    Uses the value_estimate column as a pipe for each piece (the model takes its log)."""
    rows = []
    for seed in seeds:
        t = gen.generate(seed=seed, quiet=True)
        skills, homes = t["skills"].set_index("agent_id"), t["homes"]
        pieces = {
            "Today (county records and last sale)": None,
            "Plus PERFECT current quality (exact, not a 1-5 grade)": np.exp(homes["quality"]) * 1e6,
            "Plus every part of value, but in the wrong shape": homes["true_value"] - gen.PRICE_FLOOR,
            "Plus true value (right shape)": homes["true_value"],
        }
        for label, est in pieces.items():
            sales = t["sales"] if est is None else t["sales"].assign(value_estimate=est.to_numpy())
            rows.append({"setting": label, **score_world(sales, skills, 0.0)})
    return pd.DataFrame(rows).groupby("setting", sort=False)[["buyer_rank", "seller_rank", "leftover_error"]].mean()


def chart(out, path):
    from validate import AQUA, BLUE, INK2, MUTED, ORANGE

    colors = {0.4: BLUE, 0.0: AQUA, -0.4: ORANGE}
    short = {0.4: "Good agents get pricier homes", 0.0: "Homes assigned at random",
             -0.4: "Good agents get cheaper homes"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharey="row")
    base = out[out["kind"] == "none"].set_index("world")
    for col, kind in enumerate(["rating", "estimate"]):
        d = out[out["kind"] == kind]
        for row, side in enumerate(["buyer", "seller"]):
            ax = axes[row, col]
            for w in WORLDS:
                g = d[(d["world"] == w) & ~d["leak"]].sort_values("measured")
                x = g["measured"] if kind == "rating" else g["measured"] * 100
                ax.plot(x, g[f"{side}_rank"], "-o", color=colors[w], ms=4.5, lw=2, label=short[w])
                ax.axhline(base.loc[w, f"{side}_rank"], color=colors[w], lw=1, ls=":")
                leak = d[(d["world"] == w) & d["leak"]]
                lx = leak["measured"] if kind == "rating" else leak["measured"] * 100
                ax.plot(lx, leak[f"{side}_rank"], "x", color=colors[w], ms=9, mew=2.2)
            if side == "buyer":
                ax.axhline(BAR_RANK, color=MUTED, lw=1, ls="--")
            if kind == "estimate":
                ax.invert_xaxis()
    axes[0, 0].set_title("A. Condition rating (1 to 5)")
    axes[0, 1].set_title("B. Independent value estimate")
    axes[1, 0].set_xlabel("Rating accuracy (how closely it tracks true condition, 1 = perfect)")
    axes[1, 1].set_xlabel("Typical miss of the estimate, % (smaller = better, to the right)")
    axes[0, 0].set_ylabel("Buyer score: rank agreement")
    axes[1, 0].set_ylabel("Seller score: rank agreement")
    axes[0, 0].text(0.3, BAR_RANK + 0.03, "bar for usable buyer scores", color=INK2, fontsize=8.5)
    axes[0, 1].legend(loc="lower left", fontsize=8)
    axes[1, 1].text(0.99, 0.03, "dotted = no extra data\nx = leaky version: rating written by the listing agent (A),\n"
                    "loan appraisal made knowing the price (B: order looks fine,\nbut almost no agent is called clearly different)",
                    transform=axes[1, 1].transAxes, ha="right", va="bottom", color=INK2, fontsize=8)
    axes[0, 0].set_ylim(min(-0.1, out["buyer_rank"].min() - 0.08), 1.0)
    axes[1, 0].set_ylim(min(0.0, out["seller_rank"].min() - 0.08), 1.0)
    fig.suptitle("How good must condition data be? Rank agreement with true skill (5 fake worlds per point)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main():
    out = run()
    pd.set_option("display.width", 180)
    show = out.assign(world=out["world"].map({0.4: "pricier", 0.0: "random", -0.4: "cheaper"}))
    print("CONDITION DATA TEST (average of", len(SEEDS), "fake worlds per row)")
    print("rating: level = target accuracy, measured = how closely the 1-5 rating tracks quality")
    print("estimate: level = target miss, measured = its real typical miss (0.03 = 3%)")
    print(show.round(2).to_string(index=False))
    print(f"\nBar: buyer rank agreement >= {BAR_RANK} and >= {BAR_CALLS:.0%} of clear calls right, in every world.")
    print("Condition rating, lowest accuracy that clears it:", verdict(out, "rating") or "none tested")
    print("Value estimate, largest typical miss that clears it:", verdict(out, "estimate") or "none tested")
    print("\nWHY A CONDITION RATING BARELY HELPS (standard world, 5 fake worlds)")
    why = why_ratings_fail()
    print(why.round(2).to_string())
    why.round(4).to_csv(DATA / "condition_why.csv")
    out.round(4).to_csv(DATA / "condition_test.csv", index=False)
    DOCS.mkdir(exist_ok=True)
    chart(out, DOCS / "condition_data.png")
    print("Saved condition_test.csv and docs/condition_data.png")


if __name__ == "__main__":
    main()
