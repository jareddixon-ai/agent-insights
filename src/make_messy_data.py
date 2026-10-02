"""Make a deliberately messy copy of the fake sales, like a real brokerage export.
Writes data/fake/sales_messy.csv. The clean file is untouched, so we can check
that cleaning recovers it."""

import numpy as np
import pandas as pd

from agent_model import DATA

SEED = 7


def make_messy(sales, rng):
    m = sales.copy()
    n = len(m)

    # Missing values.
    for col, share in [("sqft", 0.02), ("lot_sqft", 0.03), ("assessed_value", 0.05), ("beds", 0.01)]:
        m.loc[rng.random(n) < share, col] = np.nan

    # Typos and inconsistent formatting.
    m["zip"] = m["zip"].astype(str)
    pick = rng.random(n) < 0.03
    m.loc[pick, "zip"] = m.loc[pick, "zip"] + " "          # trailing space
    pick = rng.random(n) < 0.02
    m.loc[pick, "listing_agent_id"] = m.loc[pick, "listing_agent_id"].str.lower()
    pick = rng.random(n) < 0.02
    m.loc[pick, "office"] = m.loc[pick, "office"].str.upper()
    # Prices typed as text with $ and commas.
    pick = rng.random(n) < 0.02
    m["sold_price"] = m["sold_price"].astype(object)
    m.loc[pick, "sold_price"] = m.loc[pick, "sold_price"].map(lambda v: f"${v:,.0f}")

    # Listings that never sold (withdrawn / expired): no sold price or date.
    n_off = int(0.04 * n)
    off = rng.choice(n, n_off, replace=False)
    m.loc[off, ["sold_price", "sold_date"]] = np.nan
    m["status"] = "sold"
    m.loc[off, "status"] = rng.choice(["withdrawn", "expired"], n_off)

    # Duplicate rows (same listing entered twice).
    dup = m.sample(frac=0.01, random_state=SEED)
    m = pd.concat([m, dup], ignore_index=True)
    return m.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False, dtype=str).replace("", np.nan)
    sales = sales.astype({c: float for c in ["beds", "baths", "sqft", "lot_sqft", "year_built",
                                              "assessed_value", "sold_price"]})
    messy = make_messy(sales, np.random.default_rng(SEED))
    messy.to_csv(DATA / "sales_messy.csv", index=False)
    print("Messy rows:", len(messy), "(clean had", len(sales), ")")


if __name__ == "__main__":
    main()
