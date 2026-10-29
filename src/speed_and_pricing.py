"""Step 3-2: speed and pricing accuracy. DESCRIPTIVE ONLY, not part of the score.

Speed: how long an agent's listings take to sell compared with similar homes
       (same zip, same price band, listed in the same season).
       "-10%" means their homes sell 10% faster than similar homes.
Pricing: how the agent's first list price compares with what the home-only model
       (3-1) says the home was worth on the list date.
       "+3%" means they usually list 3% above that value.

Trap (list price games): list price is the agent's own lever. Listing low on purpose can
spark bidding and fast sales; listing high can mean price cuts and slow sales. Neither
changes what the home finally sells for in this data. So these numbers explain HOW an agent
works, they don't say whether the client got a better price. That is why they sit next to
the score instead of inside it.
"""

import numpy as np
import pandas as pd

from agent_model import DATA
from shrinkage import shrink

N_PRICE_BANDS = 5  # split homes into 5 equal-size price groups
Z95 = 1.96


def wilson(share, n):
    """95% range for a share (Wilson score interval): works even with 1 listing or a share of 0."""
    centre = (share + Z95**2 / (2 * n)) / (1 + Z95**2 / n)
    half = Z95 * np.sqrt(share * (1 - share) / n + Z95**2 / (4 * n**2)) / (1 + Z95**2 / n)
    return (centre - half).clip(0, 1), (centre + half).clip(0, 1)


def similar_home_groups(sales, pred):
    """Label each sale with its comparison group: zip x price band x list season."""
    band = pd.qcut(pred["predicted_at_list"], N_PRICE_BANDS, labels=False)
    season = pd.to_datetime(sales["list_date"]).dt.quarter
    return sales["zip"].astype(str) + "_" + band.astype(str) + "_" + season.astype(str)


def speed_and_pricing(sales, pred):
    """One row per listing agent."""
    d = sales.merge(pred[["listing_id", "predicted_at_list"]], on="listing_id")
    group = similar_home_groups(d, d)
    # Log ratio to the group's typical (median) days on market. 0 = typical speed.
    d["speed_log"] = np.log(d["days_on_market"]) - np.log(d.groupby(group)["days_on_market"].transform("median"))
    d["list_vs_value"] = d["original_list_price"] / d["predicted_at_list"] - 1
    d["cut"] = d["num_price_changes"] > 0

    g = d.groupby("listing_agent_id")
    out = pd.DataFrame(
        {
            "listings": g.size(),
            "list_vs_value": g["list_vs_value"].median(),          # direction: above (+) or below (-)
            "typical_list_miss": g["list_vs_value"].apply(lambda x: x.abs().median()),
            "share_with_price_cut": g["cut"].mean(),
        }
    )
    # 95% ranges for the two pricing numbers (Phase 6: the dashboard shows a range for every number).
    # Median: about 1.25 x spread / sqrt(listings), using the typical spread of one agent's listings
    # around their own median. Share with a cut: Wilson interval, which stays sensible at 0 or 1 listing.
    spread = (d["list_vs_value"] - g["list_vs_value"].transform("median")).std()
    half = Z95 * 1.2533 * spread / np.sqrt(out["listings"])
    out["list_vs_value_low"], out["list_vs_value_high"] = out["list_vs_value"] - half, out["list_vs_value"] + half
    out["cut_low"], out["cut_high"] = wilson(out["share_with_price_cut"], out["listings"])
    # Speed is noisy for thin records, so shrink it toward typical, same as the scores.
    raw_speed = g["speed_log"].mean()
    sp = shrink(raw_speed, out["listings"], d["speed_log"].std())
    out["speed_vs_similar"] = np.expm1(sp["score"])               # -0.10 = 10% faster
    out["speed_low"], out["speed_high"] = np.expm1(sp["low"]), np.expm1(sp["high"])
    return out


def main():
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    pred = pd.read_csv(DATA / "predictions.csv")
    skills = pd.read_csv(DATA / "true_skills.csv").set_index("agent_id")
    t = speed_and_pricing(sales, pred)

    print("SPEED AND PRICING (descriptive, not scored)")
    print(f"List price vs model value: middle agent {t['list_vs_value'].median():+.1%}, "
          f"range {t['list_vs_value'].min():+.1%} to {t['list_vs_value'].max():+.1%}")
    print(f"Speed vs similar homes: {t['speed_vs_similar'].min():+.0%} to {t['speed_vs_similar'].max():+.0%}")
    c = lambda a, b: t[a].corr(b if isinstance(b, pd.Series) else t[b], method="spearman")
    print(f"Agents who list higher sell slower? correlation {c('list_vs_value', 'speed_vs_similar'):+.2f}")
    print(f"Does listing close to value mean real skill? correlation of typical miss with true seller skill "
          f"{c('typical_list_miss', skills['seller_premium'].reindex(t.index)):+.2f} (near 0 = no)")
    t.round(4).to_csv(DATA / "speed_pricing.csv", index_label="agent_id")
    print("Saved speed_pricing.csv")


if __name__ == "__main__":
    main()
