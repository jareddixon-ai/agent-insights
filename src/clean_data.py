"""Clean a sales table: fix formats, drop unsold and duplicate rows, fill gaps.
Returns the clean table and a report of what changed, so nothing is silently dropped."""

import numpy as np
import pandas as pd

from agent_model import DATA


def clean_sales(raw):
    df = raw.copy()
    report = {"rows_in": len(df)}

    # 1. Formats: text -> numbers, tidy IDs and labels.
    df["sold_price"] = pd.to_numeric(
        df["sold_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    df["zip"] = df["zip"].astype(str).str.strip()
    df["listing_agent_id"] = df["listing_agent_id"].astype(str).str.strip().str.upper()
    df["buyer_agent_id"] = df["buyer_agent_id"].fillna("").astype(str).str.strip().str.upper()
    df["office"] = df["office"].astype(str).str.strip().str.title()
    for col in ["list_date", "sold_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. Duplicates: same listing entered twice.
    before = len(df)
    df = df.drop_duplicates(subset="listing_id", keep="first")
    report["duplicates_dropped"] = before - len(df)

    # 3. Listings that never sold have no outcome to learn from.
    before = len(df)
    df = df.dropna(subset=["sold_price", "sold_date"])
    report["unsold_dropped"] = before - len(df)

    # 4. Missing home features: fill with the zip's typical value (median).
    # Missing assessed_value is filled from size and zip only. Never from sold_price:
    # that would leak the outcome we are trying to explain into the model's inputs.
    filled = {}
    for col in ["sqft", "lot_sqft", "beds", "baths", "year_built"]:
        filled[col] = int(df[col].isna().sum())
        df[col] = df[col].fillna(df.groupby("zip")[col].transform("median"))
    per_sqft = (df["assessed_value"] / df["sqft"]).groupby(df["zip"]).transform("median")
    filled["assessed_value"] = int(df["assessed_value"].isna().sum())
    df["assessed_value"] = df["assessed_value"].fillna(df["sqft"] * per_sqft)
    report["filled"] = filled
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def main():
    raw = pd.read_csv(DATA / "sales_messy.csv", keep_default_na=False, na_values=[""])
    clean, report = clean_sales(raw)
    print(report)
    clean.to_csv(DATA / "sales_cleaned.csv", index=False)


if __name__ == "__main__":
    main()
