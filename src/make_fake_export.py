"""Phase 7: a fake "brokerage export" for testing private-copy mode. FAKE DATA ONLY.

Starts from data/fake/ and makes it look like a real export from someone else's system:
  - their own column names ("Close Price", "MLS Number", "DOM", ...)
  - messy values (prices as "$3,450,000", lower-case agent codes, blanks, typos, unsold rows,
    duplicates; from make_messy_data.py) and US-style dates (10/29/2026)
  - missing pieces: no last sale date or price, no years licensed, no offers file
  - columns we must never read: made-up addresses, seller names, buyer emails, agent phones,
    free-text remarks, and agent names and emails in the roster

Writes data/fake/export/ (git-ignored). config/column_mapping_demo.json maps it.
Run:  python src/make_fake_export.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from agent_model import DATA
from make_messy_data import SEED, make_messy

SALES_NAMES = {"listing_id": "MLS Number", "zip": "Zip Code", "beds": "Bedrooms", "baths": "Bathrooms Total",
               "sqft": "Living Area SqFt", "lot_sqft": "Lot Size SqFt", "year_built": "Year Built",
               "assessed_value": "Tax Assessed Value", "list_date": "Listing Contract Date", "sold_date": "Close Date",
               "sold_price": "Close Price", "listing_agent_id": "List Agent Code", "buyer_agent_id": "Buyer Agent Code",
               "original_list_price": "Original List Price", "num_price_changes": "Price Change Count",
               "days_on_market": "DOM", "status": "Status", "final_list_price": "List Price"}
AGENT_NAMES = {"agent_id": "Agent Code", "office": "Office Name", "team": "Team Name"}
DROPPED = ["prior_sale_date", "prior_sale_price", "office", "team", "address"]   # missing from this export


def fake_personal(n, rng):
    """Made-up personal details, clearly fake, so tests can prove none of them leak."""
    i = np.arange(1, n + 1)
    return pd.DataFrame({
        "Street Address": [f"{k} Example Lane" for k in i],
        "Seller Name": [f"Pat Example{k:05d}" for k in i],
        "Buyer Email": [f"buyer{k:05d}@example.com" for k in i],
        "List Agent Phone": [f"805-555-01{k % 100:02d}" for k in i],
        "Private Remarks": [f"Fake note {k:05d}: seller relocating for work" for k in i],
    })


def make_export(sales, agents, rng):
    messy = make_messy(sales, rng)
    for col in ["list_date", "sold_date"]:
        messy[col] = pd.to_datetime(messy[col], errors="coerce").dt.strftime("%m/%d/%Y")
    # Some original list prices typed as text too.
    pick = rng.random(len(messy)) < 0.02
    messy["original_list_price"] = messy["original_list_price"].astype(object)
    messy.loc[pick, "original_list_price"] = messy.loc[pick, "original_list_price"].map(lambda v: f"${v:,.0f}")
    export = messy.drop(columns=DROPPED).rename(columns=SALES_NAMES)
    export = pd.concat([export, fake_personal(len(export), rng)], axis=1)
    roster = agents.drop(columns=["years_licensed"]).rename(columns=AGENT_NAMES)
    roster["Agent Full Name"] = [f"Agent Example {a}" for a in agents["agent_id"]]
    roster["Agent Email"] = [f"{a.lower()}@agents.example.com" for a in agents["agent_id"]]
    return export, roster


def main(data=DATA, out=None):
    out = Path(out) if out else Path(data) / "export"
    out.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(Path(data) / "sales.csv", keep_default_na=False, dtype=str).replace("", np.nan)
    sales = sales.astype({c: float for c in ["beds", "baths", "sqft", "lot_sqft", "year_built", "assessed_value",
                                             "prior_sale_price", "sold_price", "original_list_price",
                                             "final_list_price", "num_price_changes", "days_on_market"]})
    agents = pd.read_csv(Path(data) / "agents.csv", keep_default_na=False)
    export, roster = make_export(sales, agents, np.random.default_rng(SEED))
    export.to_csv(out / "Closed Sales Export.csv", index=False)
    roster.to_csv(out / "Agent Roster.csv", index=False)
    print(f"Fake export: {len(export)} sales rows, {len(roster)} agents, no offers file. Saved to {out}")
    return out


if __name__ == "__main__":
    main()
