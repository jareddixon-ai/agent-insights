"""Step 3-4: one coaching table per agent, saved as data/fake/rankings.csv.

Brings together:
  seller score  : % above/below what the home predicts when the agent lists it (2B + 2C + 3-3)
  buyer savings : % below what the home predicts when the agent buys it for a client
  offer score   : on the same home, at the same price and terms, how much more often the
                  agent's offers win (5-2, offer_metrics.py), in win-chance points in a 3-offer contest
  speed, pricing, offers attracted: descriptive only (3-2, 5-2), never part of the index

Coaching index: a weighted blend of the scores (weights in config/scoring.json).
Each part is first put on the same scale (divided by how much agents typically differ on
it), so a 1-point move means "one typical step" on any part. 0 = the average agent.
Since Phase 5 the seller and offer scores have weight; the buyer score is context only.

Every number keeps a 95% range. Seller and buyer ranges also get a bias allowance (4.5):
room for hidden home quality, which no amount of deals averages away. If the range
includes 0, the agent is "not clearly different from average", an honest and common answer.

Run order: generate_fake_data, price_model, speed_and_pricing, bootstrap_ranges,
offer_metrics, then this file.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent_model import DATA

CONFIG = Path(__file__).resolve().parent.parent / "config" / "scoring.json"
Z95 = 1.96  # 95% ranges are about 1.96 standard errors either side


def load_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    assert cfg["range"] in {"formula", "bootstrap", "wider"}, "range must be formula, bootstrap or wider"
    assert all(w >= 0 for w in cfg["weights"].values()) and sum(cfg["weights"].values()) > 0
    assert cfg.get("bias_allowance", 0) >= 0
    return cfg


def pick_range(t, choice):
    """Return (low, high) using the range named in the config."""
    if choice == "formula":
        return t["formula_low"], t["formula_high"]
    boot_low, boot_high = t["boot_low"].fillna(t["formula_low"]), t["boot_high"].fillna(t["formula_high"])
    if choice == "bootstrap":
        return boot_low, boot_high
    return np.minimum(t["formula_low"], boot_low), np.maximum(t["formula_high"], boot_high)


def table_range(t, cfg):
    """The range the coaching table uses for one side: the range named in the config,
    widened by the bias allowance. Combined like two independent errors (square root of
    the sum of squares), so a wide range grows a little and a narrow one grows a lot."""
    low, high = pick_range(t, cfg["range"])
    room = (Z95 * cfg.get("bias_allowance", 0)) ** 2
    return t["score"] - np.sqrt((t["score"] - low) ** 2 + room), t["score"] + np.sqrt((high - t["score"]) ** 2 + room)


def group(low, high):
    return np.where(low > 0, "Clearly above average",
                    np.where(high < 0, "Clearly below average", "Not clearly different from average"))


def build_table(boot, offer, speed, agents, cfg):
    """offer: offer_scores.csv (fair offer score with range, and offers attracted)."""
    seller = boot.xs("seller", level="side")
    buyer = boot.xs("buyer", level="side")
    s_low, s_high = table_range(seller, cfg)
    b_low, b_high = table_range(buyer, cfg)
    scored = offer[offer["score"].notna()]

    # Each part as "higher = better for the client", with its range.
    parts = {
        "seller": pd.DataFrame({"score": seller["score"], "low": s_low, "high": s_high, "n": seller["deals"]}),
        # buyer model effect is a PRICE effect (lower = better), so flip it into savings
        "buyer": pd.DataFrame({"score": -buyer["score"], "low": -b_high, "high": -b_low, "n": buyer["deals"]}),
        # already "higher = better" and centered on the average agent
        "offer_win": pd.DataFrame({"score": scored["score"], "low": scored["low"],
                                   "high": scored["high"], "n": scored["offers"]}),
    }

    table = agents.set_index("agent_id")[["office", "team", "years_licensed"]].copy()
    num, den, var = 0.0, 0.0, 0.0
    for name, p in parts.items():
        p = p.reindex(table.index)
        table[f"{name}_score"], table[f"{name}_low"], table[f"{name}_high"] = p["score"], p["low"], p["high"]
        table[f"{name}_n"] = p["n"].fillna(0).astype(int)
        scored = cfg["weights"].get(name, 0) > 0
        table[f"{name}_group"] = np.where(p["score"].isna(), "No data",
                                          group(p["low"], p["high"]) if scored else "Context only")
        spread = p["score"].std()  # how much agents typically differ on this part
        z, se = p["score"] / spread, (p["high"] - p["low"]) / (2 * Z95) / spread
        w = cfg["weights"].get(name, 0) * p["score"].notna()
        num = num + w * z.fillna(0)
        den = den + w
        var = var + (w**2) * se.fillna(0) ** 2   # assumes the parts' errors are independent (checked in 5-4)

    table["coaching_index"] = num / den.replace(0, np.nan)
    half = Z95 * np.sqrt(var) / den.replace(0, np.nan)
    table["index_low"], table["index_high"] = table["coaching_index"] - half, table["coaching_index"] + half
    table["index_group"] = np.where(table["coaching_index"].isna(), "No data",
                                    group(table["index_low"], table["index_high"]))
    total_deals = table["seller_n"] + table["buyer_n"]
    table["low_data_flag"] = total_deals < cfg["small_sample"]

    # Descriptive context only (not in the index).
    ctx = speed[["speed_vs_similar", "list_vs_value", "share_with_price_cut"]]
    table = table.join(ctx)
    attract = offer.reindex(table.index)
    table["offers_attracted"] = attract["attract_fair_score"]   # vs similar listings at a similar list price
    table["offers_attracted_low"], table["offers_attracted_high"] = attract["attract_fair_low"], attract["attract_fair_high"]
    # Agents with no scored part (for example no listings) stay in the table with "No data",
    # so their context scores can still be seen.
    return table.sort_values("coaching_index", ascending=False, na_position="last")


GOAL_WEIGHTS = {"seller": 0.5, "buyer": 0.3, "offer_win": 0.2}   # what a brokerage would ideally measure (Phase 3)


def true_blend(skills, weights):
    """Fake data only: the same blend of TRUE skills, each put on the same scale."""
    true_parts = {"seller": skills["seller_premium"], "buyer": skills["buyer_savings"], "offer_win": skills["win_skill"]}
    return sum(weights[k] * (v - v.mean()) / v.std() for k, v in true_parts.items())


def grade(table, skills, cfg):
    """Fake data only: compare the index with the same blend of TRUE skills."""
    table = table[table["coaching_index"].notna()]
    sk = skills.reindex(table.index)
    true_index = true_blend(sk, cfg["weights"])
    agree = table["coaching_index"].rank().corr(true_index.rank())
    goal = table["coaching_index"].rank().corr(true_blend(sk, GOAL_WEIGHTS).rank())
    above = table["index_group"] == "Clearly above average"
    below = table["index_group"] == "Clearly below average"
    print(f"Coaching index rank agreement with the true blend of skills: {agree:.2f} "
          f"(with the full goal blend {GOAL_WEIGHTS}: {goal:.2f})")
    print(f"Called clearly above average: {above.sum()} agents, {(true_index[above] > 0).mean():.0%} truly above average")
    print(f"Called clearly below average: {below.sum()} agents, {(true_index[below] < 0).mean():.0%} truly below average")
    print(f"Not clearly different from average: {(~above & ~below).sum()} agents")


def main():
    cfg = load_config()
    boot = pd.read_csv(DATA / "bootstrap_scores.csv", index_col=["side", "agent_id"])
    offer = pd.read_csv(DATA / "offer_scores.csv", index_col="agent_id")
    speed = pd.read_csv(DATA / "speed_pricing.csv", index_col="agent_id")
    agents = pd.read_csv(DATA / "agents.csv", keep_default_na=False)
    table = build_table(boot, offer, speed, agents, cfg)

    print(f"COACHING TABLE (weights {cfg['weights']}, ranges: {cfg['range']}, "
          f"bias allowance {cfg.get('bias_allowance', 0)})")
    for part in ["seller", "buyer", "offer_win"]:
        counts = table[f"{part}_group"].value_counts()
        print(f"  {part:9s}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    skills_file = DATA / "true_skills.csv"
    if skills_file.exists():  # only true for fake data
        grade(table, pd.read_csv(skills_file).set_index("agent_id"), cfg)
    table.round(4).to_csv(DATA / "rankings.csv", index_label="agent_id")
    print("Saved rankings.csv")


if __name__ == "__main__":
    main()
