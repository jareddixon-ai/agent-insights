"""Step 3-4: one coaching table per agent, saved as data/fake/rankings.csv.

Brings together:
  seller score  : % above/below what the home predicts when the agent lists it (2B + 2C + 3-3)
  buyer savings : % below what the home predicts when the agent buys it for a client
  offer win rate: share of the agent's offers that win (2C-style shrinkage)
  speed, pricing: descriptive only (3-2), never part of the index

Coaching index: a weighted blend of the scores (weights in config/scoring.json).
Each part is first put on the same scale (divided by how much agents typically differ on
it), so a 1-point move means "one typical step" on any part. 0 = the average agent.
Since Phase 4.5 only the seller score has weight; buyer and offer scores are context only.

Every number keeps a 95% range. Seller and buyer ranges also get a bias allowance (4.5):
room for hidden home quality, which no amount of deals averages away. If the range
includes 0, the agent is "not clearly different from average", an honest and common answer.

Run order: generate_fake_data, price_model, speed_and_pricing, bootstrap_ranges,
offer_win_rates, then this file.
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


def build_table(boot, wins, speed, agents, cfg):
    seller = boot.xs("seller", level="side")
    buyer = boot.xs("buyer", level="side")
    s_low, s_high = table_range(seller, cfg)
    b_low, b_high = table_range(buyer, cfg)
    p0 = wins["wins"].sum() / wins["offers"].sum()

    # Each part as "higher = better for the client", with its range.
    parts = {
        "seller": pd.DataFrame({"score": seller["score"], "low": s_low, "high": s_high, "n": seller["deals"]}),
        # buyer model effect is a PRICE effect (lower = better), so flip it into savings
        "buyer": pd.DataFrame({"score": -buyer["score"], "low": -b_high, "high": -b_low, "n": buyer["deals"]}),
        "offer_win": pd.DataFrame({"score": wins["win_rate"] - p0, "low": wins["low"] - p0,
                                   "high": wins["high"] - p0, "n": wins["offers"]}),
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
        var = var + (w**2) * se.fillna(0) ** 2   # assumes the parts' errors are independent

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
    # Agents with no scored part (for example no listings) stay in the table with "No data",
    # so their context scores can still be seen.
    return table.sort_values("coaching_index", ascending=False, na_position="last")


def grade(table, skills, cfg):
    """Fake data only: compare the index with the same blend of TRUE skills."""
    table = table[table["coaching_index"].notna()]
    sk = skills.reindex(table.index)
    true_parts = {"seller": sk["seller_premium"], "buyer": sk["buyer_savings"], "offer_win": sk["win_skill"]}
    true_index = sum(cfg["weights"][k] * (v - v.mean()) / v.std() for k, v in true_parts.items())
    agree = table["coaching_index"].rank().corr(true_index.rank())
    above = table["index_group"] == "Clearly above average"
    below = table["index_group"] == "Clearly below average"
    print(f"Coaching index rank agreement with the true blend of skills: {agree:.2f}")
    print(f"Called clearly above average: {above.sum()} agents, {(true_index[above] > 0).mean():.0%} truly above average")
    print(f"Called clearly below average: {below.sum()} agents, {(true_index[below] < 0).mean():.0%} truly below average")
    print(f"Not clearly different from average: {(~above & ~below).sum()} agents")


def main():
    cfg = load_config()
    boot = pd.read_csv(DATA / "bootstrap_scores.csv", index_col=["side", "agent_id"])
    wins = pd.read_csv(DATA / "offer_win_rates.csv", index_col="agent_id")
    speed = pd.read_csv(DATA / "speed_pricing.csv", index_col="agent_id")
    agents = pd.read_csv(DATA / "agents.csv", keep_default_na=False)
    table = build_table(boot, wins, speed, agents, cfg)

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
