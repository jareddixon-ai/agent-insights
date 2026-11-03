"""Phase 9: the sales behind every number, one row per sale. Sale ID and agent codes only.

For each closed sale: how far the sale price sat from what the home-only price model expected
(predictions.csv, built by price_model.py: every home predicted by a model that never saw it), how many
OTHER sales in the data look similar (same zip, expected value within 30%), and whether that is too few to
trust the estimate. No address, no names, no dates beyond the sale year.

This file explains numbers; it never changes them. The seller score, ranges and calls come from the joint
regression in score_agents.py and are untouched.

"Low confidence" = fewer than MIN_SIMILAR similar sales. A home in a thin part of the market (the luxury
tail, an unusual zip) has a less certain estimate of what it should sell for, and the agent's range may not
fully show that. An agent gets the label when more than LOW_SHARE of their listings are in thin parts.

Run order: after price_model.py. Writes sale_detail.csv next to sales.csv (data/fake/, or a private folder's
scored/ via run_private.py).
"""

import numpy as np
import pandas as pd

from agent_model import DATA

MIN_SIMILAR = 30        # fewer similar sales than this = low confidence for that sale
BAND = 0.30             # "similar" = expected value within +/-30%, same zip
LOW_SHARE = 1 / 3       # an agent is low confidence when more than this share of their listings are


def similar_counts(zips, expected):
    """For every sale: how many OTHER sales in the same zip have an expected value within BAND of its own."""
    out = np.zeros(len(expected), dtype=int)
    for z in pd.unique(zips):
        idx = np.flatnonzero(zips == z)
        v = np.sort(expected[idx])
        lo = np.searchsorted(v, expected[idx] / (1 + BAND), side="left")
        hi = np.searchsorted(v, expected[idx] * (1 + BAND), side="right")
        out[idx] = hi - lo - 1               # minus the sale itself
    return out


def build_sale_detail(sales, predictions):
    """sales: our standard sales table. predictions: predictions.csv (listing_id, sold_price, predicted_price)."""
    p = predictions.set_index("listing_id")
    s = sales.set_index("listing_id").join(p[["predicted_price"]], how="inner")
    expected = s["predicted_price"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "year": pd.to_datetime(s["sold_date"]).dt.year.to_numpy(),
        "listing_agent_id": s["listing_agent_id"].to_numpy(),
        "buyer_agent_id": s["buyer_agent_id"].fillna("").to_numpy(),
        "sold_vs_model": (s["sold_price"] / s["predicted_price"] - 1).round(4).to_numpy(),
        "n_similar": similar_counts(s["zip"].to_numpy(), expected),
    }, index=s.index)
    out["low_confidence"] = out["n_similar"] < MIN_SIMILAR
    return out.rename_axis("listing_id")


def agent_confidence(detail):
    """Per listing agent: how many listings, how many are low confidence, and whether the agent gets the label."""
    g = detail.groupby("listing_agent_id")["low_confidence"].agg(listings="size", low="sum")
    g["label"] = g["low"] / g["listings"] > LOW_SHARE
    return g


def main(folder=DATA):
    sales = pd.read_csv(folder / "sales.csv", keep_default_na=False)
    detail = build_sale_detail(sales, pd.read_csv(folder / "predictions.csv"))
    detail.to_csv(folder / "sale_detail.csv")
    conf = agent_confidence(detail)
    print(f"SALE DETAIL: {len(detail):,} sales; {detail['low_confidence'].mean():.1%} in thin parts of the market "
          f"(under {MIN_SIMILAR} similar sales); {int(conf['label'].sum())} of {len(conf)} listing agents get the low confidence label.")
    print("Saved sale_detail.csv")


if __name__ == "__main__":
    main()
