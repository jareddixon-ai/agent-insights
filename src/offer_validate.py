"""Step 5-4: do the new offer metrics work, and does the offer score earn weight back?

Graded against the answer key in 3 kinds of world (good agents get pricier homes, homes at
random, good agents get cheaper homes), 5 fake worlds each.

Rule for giving the offer score weight, set BEFORE looking at results:
  in every kind of world,
    rank agreement with hidden win skill            at least 0.5
    rank agreement with buyer savings (overpaying)  between -0.15 and 0.15 (it must not reward overpaying)
    95% ranges catch the truth                      at least 90% of the time
    "clearly above/below average" calls right       at least 90%
Since the Phase 5 follow-up, ranges use the widening calibrated in offer_calibrate.py, and
seeds 42-46 are the calibration worlds; the fresh-world confirmation (seeds 47-51) lives there.
The buyer score stays at weight 0: step 5-0 showed it needs an independent value estimate
(about 4% typical miss or better), which this data doesn't have.

Also checked: offers attracted (home only vs allowing for list price), whether seller and
offer score errors are linked (the coaching index assumes they are not), and the coaching
index with and without the offer score.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generate_fake_data as gen
import offer_metrics as om
from agent_model import DATA, fit_agent_effects
from offer_win_rates import win_rates
from price_model import cross_validated_predictions
from score_agents import GOAL_WEIGHTS, build_table, load_config, true_blend
from shrinkage import shrink

DOCS = Path(__file__).resolve().parent.parent / "docs"
SEEDS = [42, 43, 44, 45, 46]
WORLDS = {0.4: "pricier", 0.0: "random", -0.4: "cheaper"}
CONFIGS = {"Seller only (Phase 4.5)": {"seller": 1.0, "buyer": 0.0, "offer_win": 0.0},
           "Seller + offers (Phase 5)": {"seller": 0.5, "buyer": 0.0, "offer_win": 0.2}}
RULE = {"win_rank": 0.5, "savings_rank": 0.15, "coverage": 0.90, "calls_right": 0.90}


def as_boot(t):
    """Shape a shrinkage table like bootstrap_scores.csv (formula range only; no bootstrap here)."""
    return pd.DataFrame({"score": t["score"], "deals": t["deals"], "formula_low": t["low"],
                         "formula_high": t["high"], "boot_low": np.nan, "boot_high": np.nan})


def one_world(seed, strength):
    t = gen.generate(seed=seed, selection_strength=strength, quiet=True)
    sales, offers, agents = t["sales"], t["offers"], t["agents"]
    sk = t["skills"].set_index("agent_id")
    rank = lambda x, truth: x.rank().corr(truth.reindex(x.index).rank())

    raw = win_rates(offers)["win_rate"]
    comp = om.competition_adjusted(offers)
    fair = om.fair_offer_scores(offers)
    per_pct = fair.attrs["terms"]["price_pct"]          # model weight of 1% more price
    est = fair["effect"] / per_pct                       # agent effect in "% of price" terms
    half = om.Z95 * om.widened_se(fair["effect_se"], per_pct, om.offer_range_setting()) / per_pct   # calibrated widening
    truth = 100 * gen.WIN_SKILL_VALUE * (sk["win_skill"] - sk["win_skill"].reindex(fair.index).mean())
    truth = truth.reindex(fair.index)
    clear = (est - half > 0) | (est + half < 0)

    effects, _, resid_sd = fit_agent_effects(sales)
    seller = shrink(effects["seller_effect"], effects["listings"], resid_sd)
    buyer = shrink(effects["buyer_effect"], effects["buyer_deals"], resid_sd)
    seller_miss = seller["score"] - (sk["seller_premium"] - sk["seller_premium"].mean()).reindex(seller.index)
    offer_miss = (est - truth) / 100
    both = seller_miss.index.intersection(offer_miss.index)

    attract = om.offers_attracted(sales, offers, cross_validated_predictions(sales))
    lists_low = -(sales["original_list_price"] / sales["sold_price"] - 1).groupby(sales["listing_agent_id"]).mean()

    row = {
        "win_rank_raw": rank(raw, sk["win_skill"]), "win_rank_competition": rank(comp, sk["win_skill"]),
        "win_rank_fair": rank(fair["score"], sk["win_skill"]),
        "savings_rank_raw": rank(raw, sk["buyer_savings"]),
        "savings_rank_competition": rank(comp, sk["buyer_savings"]),
        "savings_rank_fair": rank(fair["score"], sk["buyer_savings"]),
        "coverage_fair": ((truth >= est - half) & (truth <= est + half)).mean(),
        "clear_calls_fair": clear.sum(), "calls_right_fair": (np.sign(est) == np.sign(truth))[clear].mean(),
        "terms_cash_pct": fair.attrs["terms"]["cash"] / per_pct,
        "attract_home_rank": rank(attract["attract_home_score"], sk["seller_premium"]),
        "attract_fair_rank": rank(attract["attract_fair_score"], sk["seller_premium"]),
        "attract_home_rewards_listing_low": rank(attract["attract_home_score"], lists_low),
        "attract_fair_rewards_listing_low": rank(attract["attract_fair_score"], lists_low),
        "seller_offer_error_corr": seller_miss[both].corr(offer_miss[both]),
    }
    boot = pd.concat({"seller": as_boot(seller), "buyer": as_boot(buyer)}, names=["side", "agent_id"])
    offer = fair.join(attract, how="outer")
    speed = pd.DataFrame(np.nan, index=agents["agent_id"],
                         columns=["speed_vs_similar", "list_vs_value", "share_with_price_cut"])
    goal = true_blend(sk, GOAL_WEIGHTS)
    for name, weights in CONFIGS.items():
        cfg = {**load_config(), "weights": weights}
        tab = build_table(boot, offer, speed, agents, cfg)
        tab = tab[tab["coaching_index"].notna()]
        called = tab["index_group"] != "Not clearly different from average"
        row[f"{name}|goal_rank"] = rank(tab["coaching_index"], goal)
        row[f"{name}|clear_calls"] = called.sum()
        row[f"{name}|calls_right"] = (np.sign(tab["coaching_index"]) == np.sign(goal.reindex(tab.index)))[called].mean()
    return row


def run(seeds=SEEDS, worlds=WORLDS):
    rows = [{"world": w, **one_world(seed, s)} for s, w in worlds.items() for seed in seeds]
    return pd.DataFrame(rows).groupby("world", sort=False).mean()


def decide(out):
    ok = ((out["win_rank_fair"] >= RULE["win_rank"]) & (out["savings_rank_fair"].abs() <= RULE["savings_rank"])
          & (out["coverage_fair"] >= RULE["coverage"]) & (out["calls_right_fair"] >= RULE["calls_right"]))
    return bool(ok.all()), ok


def chart(out, path):
    from validate import AQUA, BLUE, INK2, ORANGE

    avg = out.mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    names = ["Raw win rate", "Adjusted for\ncompetition", "Fair offer score\n(same home, same\nprice and terms)"]
    keys = ["raw", "competition", "fair"]
    x, w = np.arange(3), 0.36
    for off, col, color, label in [(-w / 2 - 0.01, "win_rank", BLUE, "Matches hidden win skill (want high)"),
                                   (w / 2 + 0.01, "savings_rank", ORANGE,
                                    "Matches buyer savings (want 0; below 0 = rewards overpaying)")]:
        vals = np.array([avg[f"{col}_{k}"] for k in keys])
        a1.bar(x + off, vals, w, color=color, label=label)
        for xi, v in zip(x + off, vals):
            a1.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK2)
    a1.axhline(0, color="#c3c2b7", lw=1)
    a1.set_xticks(x, names, fontsize=8.5)
    a1.set_ylim(-0.6, 1.15)
    a1.set_ylabel("Rank agreement")
    a1.set_title("Buyer agents: which offer metric is fair?")
    a1.legend(loc="upper left", fontsize=8)

    names2 = ["Home and market only", "Also allows for\nlist price"]
    vals_skill = [avg["attract_home_rank"], avg["attract_fair_rank"]]
    vals_style = [avg["attract_home_rewards_listing_low"], avg["attract_fair_rewards_listing_low"]]
    x2 = np.arange(2)
    a2.bar(x2 - w / 2 - 0.01, vals_skill, w, color=AQUA, label="Matches seller skill")
    a2.bar(x2 + w / 2 + 0.01, vals_style, w, color=ORANGE, label="Rewards listing low, a tactic (want 0)")
    for xi, v in zip(list(x2 - w / 2 - 0.01) + list(x2 + w / 2 + 0.01), vals_skill + vals_style):
        a2.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5, color=INK2)
    a2.axhline(0, color="#c3c2b7", lw=1)
    a2.set_xticks(x2, names2, fontsize=8.5)
    a2.set_ylim(-0.6, 1.15)
    a2.set_title("Listing agents: offers attracted")
    a2.legend(loc="upper left", fontsize=8)
    fig.suptitle("Offer metrics graded against the answer key (15 fake worlds: 3 kinds x 5 seeds)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main():
    out = run()
    pd.set_option("display.width", 200)
    print("OFFER METRICS vs ANSWER KEY (average of", len(SEEDS), "fake worlds per kind)")
    print(out.T.round(2).to_string())
    passed, by_world = decide(out)
    print("\nRule:", RULE)
    print("Offer score passes in each kind of world:", by_world.to_dict())
    print("Decision: offer score", "EARNS weight back" if passed else "stays context only")
    out.round(4).to_csv(DATA / "offer_validation.csv")
    DOCS.mkdir(exist_ok=True)
    chart(out, DOCS / "offer_validation.png")
    print("Saved offer_validation.csv and docs/offer_validation.png")


if __name__ == "__main__":
    main()
