"""Step 3-1: home-only price model, checked with cross-validation.

Question it answers: "Ignoring who the agents were, what should this home sell for?"
Inputs: size, beds, baths, lot, age, assessed value, last sale (date and price), zip, month
and trend. No agents,
no list price (the agent sets list price, so using it would leak agent behavior in).

Cross-validation (5-fold): split the sales into 5 groups. Train on 4, predict the 5th,
rotate. Every home gets a prediction from a model that NEVER saw its own price, so the
error we report is honest (a model graded on homes it trained on looks better than it is).

Used later as the yardstick for speed and pricing accuracy (3-2).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from agent_model import DATA, build_features, fit_agent_effects

N_FOLDS = 5


def cross_validated_predictions(sales, drop=()):
    """Out-of-fold predicted price for every sale, at the sold date and at the list date."""
    start = pd.to_datetime(sales["sold_date"]).min()
    X_sold = build_features(sales, "sold_date", start).drop(columns=list(drop), errors="ignore")
    X_list = build_features(sales, "list_date", start).reindex(columns=X_sold.columns, fill_value=0.0)
    y = np.log(sales["sold_price"].to_numpy())

    at_sale, at_list = np.zeros(len(sales)), np.zeros(len(sales))
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=42).split(X_sold):
        model = LinearRegression().fit(X_sold.iloc[train], y[train])
        at_sale[test] = model.predict(X_sold.iloc[test])
        at_list[test] = model.predict(X_list.iloc[test])
    return pd.DataFrame(
        {
            "listing_id": sales["listing_id"],
            "sold_price": sales["sold_price"],
            "predicted_price": np.exp(at_sale).round(-3),
            "predicted_at_list": np.exp(at_list).round(-3),
        }
    )


def error_summary(pred):
    pct = (pred["sold_price"] / pred["predicted_price"] - 1).abs()
    return {"median_error": pct.median(), "within_10pct": (pct <= 0.10).mean(), "worst_5pct": pct.quantile(0.95)}


def main():
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    pred = cross_validated_predictions(sales)
    pred["pct_error"] = (pred["sold_price"] / pred["predicted_price"] - 1).round(4)
    e = error_summary(pred)
    public = ["log_assessed", "years_owned", "log_prior_sale"]   # county records: tax value and last sale
    e_no_assessed = error_summary(cross_validated_predictions(sales, drop=public))
    e_no_history = error_summary(cross_validated_predictions(sales, drop=public[1:]))

    print(f"HOME-ONLY PRICE MODEL ({N_FOLDS}-fold cross-validation, every home predicted by a model that never saw it)")
    print(f"Median error {e['median_error']:.1%} (goal: under 10%) | within 10%: {e['within_10pct']:.0%}"
          f" | 1 home in 20 is off by more than {e['worst_5pct']:.1%}")
    print(f"Same model WITHOUT assessed value or sale history: median error {e_no_assessed['median_error']:.1%}"
          " (shows how much the county records carry)")
    if "prior_sale_date" in sales:
        print(f"Same model with assessed value but NOT the last sale: median error {e_no_history['median_error']:.1%}"
              " (Prop 13: the tax value alone is a weak guide)")
    assert e["median_error"] < 0.10, "Price model misses the 10% goal"

    # The original plan's simpler agent score: average(sold / predicted) - 1 per listing agent.
    # Compare it with the joint model from 2B, using the answer key.
    skills = pd.read_csv(DATA / "true_skills.csv").set_index("agent_id")
    two_step = (sales.assign(r=pred["sold_price"] / pred["predicted_price"] - 1)
                .groupby("listing_agent_id")["r"].mean())
    agree = two_step.rank().corr(skills["seller_premium"].reindex(two_step.index).rank())
    joint = fit_agent_effects(sales)[0]["seller_effect"].dropna()
    agree_joint = joint.rank().corr(skills["seller_premium"].reindex(joint.index).rank())
    print(f"Seller score, rank agreement with truth: two-step (avg sold/predicted - 1) {agree:.2f}"
          f" | joint model from 2B {agree_joint:.2f}")

    pred.to_csv(DATA / "predictions.csv", index=False)
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
