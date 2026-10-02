"""Offer win rate for buyer agents: of the offers an agent makes, how many win?

Same small-sample problem as before, so we shrink. For win/lose counts the standard
tool is a beta-binomial model: start from the brokerage-wide win rate, as if each
agent had already made `k` offers at that rate, then add their real record.
Few offers -> stays near the average. Many offers -> follows their own record.
"""

import numpy as np
import pandas as pd

from agent_model import DATA


def win_rates(offers):
    """Raw and shrunk win rate per buyer agent, with a 95% range."""
    offers = offers[offers["buyer_agent_id"] != ""]
    g = offers.groupby("buyer_agent_id")["outcome"]
    t = pd.DataFrame({"offers": g.size(), "wins": g.apply(lambda x: (x == "won").sum())})
    t["raw_rate"] = t["wins"] / t["offers"]

    p0 = t["wins"].sum() / t["offers"].sum()  # brokerage-wide win rate
    # How many offers' worth of "average" to blend in (method of moments):
    # real spread between agents = observed spread minus what pure luck would give.
    luck_var = (p0 * (1 - p0) / t["offers"]).mean()
    real_var = max(t["raw_rate"].var() - luck_var, 1e-4)
    k = max(p0 * (1 - p0) / real_var - 1, 1.0)

    a, b = t["wins"] + k * p0, (t["offers"] - t["wins"]) + k * (1 - p0)
    t["win_rate"] = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    t["low"], t["high"] = t["win_rate"] - 1.96 * sd, t["win_rate"] + 1.96 * sd
    t.attrs["p0"], t.attrs["k"] = p0, k
    return t


def main():
    offers = pd.read_csv(DATA / "offers.csv", keep_default_na=False)
    skills = pd.read_csv(DATA / "true_skills.csv").set_index("agent_id")
    t = win_rates(offers)
    p0, k = t.attrs["p0"], t.attrs["k"]  # read before join, which drops attrs
    t = t.join(skills["win_skill"])
    r = lambda col: t[col].rank().corr(t["win_skill"].rank())
    print(f"Brokerage-wide win rate: {p0:.0%} | prior strength k = {k:.0f} offers")
    print(f"Rank agreement with hidden win skill: raw {r('raw_rate'):.2f} -> shrunk {r('win_rate'):.2f}")
    small = t["offers"] < 15
    print("Agents with under 15 offers:", int(small.sum()), "| their raw rates range",
          f"{t.loc[small,'raw_rate'].min():.0%} to {t.loc[small,'raw_rate'].max():.0%}",
          "| shrunk range", f"{t.loc[small,'win_rate'].min():.0%} to {t.loc[small,'win_rate'].max():.0%}")
    t.round(4).to_csv(DATA / "offer_win_rates.csv", index_label="agent_id")


if __name__ == "__main__":
    main()
