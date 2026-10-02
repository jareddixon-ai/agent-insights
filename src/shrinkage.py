"""Step 2C: shrink small-sample agents toward the average and show uncertainty.

Plain English: an agent with 4 listings can look great or awful by luck. So we
pull each score toward the average (0), by a lot when the agent has few deals
and barely at all when they have many. This is called shrinkage (or partial pooling).

Each effect is a percent: 0.01 means "sells about 1% above what the home predicts".
"""

from pathlib import Path

import numpy as np
import pandas as pd

from agent_model import DATA, fit_agent_effects


def shrink(effect, n_deals, resid_sd):
    """Pull raw effects toward 0 (the average agent).

    Noise in one agent's raw score ~ resid_sd / sqrt(deals).  Real spread between
    agents (tau) = total spread in raw scores minus that noise.
    weight near 1 = trust the agent's own record, near 0 = lean on the average.
    """
    effect = effect.dropna()
    n = n_deals.reindex(effect.index).fillna(1).clip(lower=1)
    noise_var = resid_sd**2 / n
    tau2 = max(effect.var() - noise_var.mean(), 1e-6)
    weight = tau2 / (tau2 + noise_var)
    shrunk = weight * effect
    # 95% range: the agent's most likely value, give or take about 2 standard errors.
    half_width = 1.96 * np.sqrt(weight * noise_var)
    return pd.DataFrame(
        {"raw": effect, "score": shrunk, "low": shrunk - half_width,
         "high": shrunk + half_width, "weight": weight, "deals": n}
    )


def grade(table, truth, label):
    """Compare to the answer key: error, rank agreement, and whether the 95% range
    caught the truth (should be about 95% of the time)."""
    t = (truth - truth.mean()).reindex(table.index)
    rank = lambda a: a.rank().corr(t.rank())
    err_raw = (table["raw"] - t).abs().mean() * 100
    err_shr = (table["score"] - t).abs().mean() * 100
    small = table["deals"] < 10
    caught = ((t >= table["low"]) & (t <= table["high"])).mean() * 100
    print(f"{label}: error (pct points) raw {err_raw:.2f} -> shrunk {err_shr:.2f} | "
          f"under 10 deals: raw {(table.loc[small,'raw']-t[small]).abs().mean()*100:.2f} -> "
          f"shrunk {(table.loc[small,'score']-t[small]).abs().mean()*100:.2f}")
    print(f"   rank agreement raw {rank(table['raw']):.2f} -> shrunk {rank(table['score']):.2f} | "
          f"95% range caught the truth {caught:.0f}% of the time")
    top_s = set(table.nlargest(10, "score").index)
    top_t = set(t.nlargest(10).index)
    print(f"   top 10 correct: {len(top_s & top_t)}/10")


def main():
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    skills = pd.read_csv(DATA / "true_skills.csv").set_index("agent_id")
    effects, _, resid_sd = fit_agent_effects(sales)

    seller = shrink(effects["seller_effect"], effects["listings"], resid_sd)
    buyer = shrink(effects["buyer_effect"], effects["buyer_deals"], resid_sd)
    print("2C: SHRINKAGE vs ANSWER KEY (typical leftover error per sale:", round(resid_sd * 100, 1), "%)")
    grade(seller, skills["seller_premium"], "Seller side")
    grade(buyer, -skills["buyer_savings"], "Buyer side ")

    # Coaching view: an agent with a range that includes 0 is "not clearly different from average".
    clear = seller[(seller["low"] > 0) | (seller["high"] < 0)]
    print(f"\nSellers clearly above or below average (range excludes 0): {len(clear)} of {len(seller)}")
    seller.round(4).to_csv(DATA / "seller_scores.csv", index_label="agent_id")
    buyer.round(4).to_csv(DATA / "buyer_scores.csv", index_label="agent_id")
    print("Saved seller_scores.csv and buyer_scores.csv")


if __name__ == "__main__":
    main()
