"""Step 2B: the real model.

Idea: predict each sale's price from the HOME and the MARKET (size, zip, month,
trend), and give every agent a credit or penalty for what is left over.
We work in log prices, so every effect is a percent (1% = 0.01), not dollars.

log(sold price) = home features + zip + month + trend
                  + seller effect (listing agent) + buyer effect (buyer agent)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path(__file__).resolve().parent.parent / "data" / "fake"


def build_features(sales):
    """Home and market columns only. No agents yet, and no list price
    (list prices are set by the agent, so they would leak agent behavior)."""
    sold = pd.to_datetime(sales["sold_date"])
    X = pd.DataFrame(
        {
            "log_sqft": np.log(sales["sqft"]),
            "beds": sales["beds"],
            "baths": sales["baths"],
            "log_lot": np.log(sales["lot_sqft"]),
            # Public value estimate. It partly captures things we can't see
            # (view, condition), which is what stops hidden quality from leaking into agent scores.
            "log_assessed": np.log(sales["assessed_value"]),
            "age": sold.dt.year - sales["year_built"],
            "trend_years": (sold - sold.min()).dt.days / 365,
        }
    )
    X = X.join(pd.get_dummies(sales["zip"], prefix="zip", dtype=float))
    X = X.join(pd.get_dummies(sold.dt.month, prefix="month", dtype=float))
    return X


def agent_dummies(ids, prefix):
    """One 0/1 column per agent. Blank buyer agent (no buyer agent) gets no column."""
    return pd.get_dummies(ids.replace("", np.nan), prefix=prefix, dtype=float)


def fit_agent_effects(sales):
    """Fit one regression for everything at once, so agent credit is measured
    AFTER accounting for the home. Returns a table of seller and buyer effects (in %)."""
    y = np.log(sales["sold_price"])
    S = agent_dummies(sales["listing_agent_id"], "seller")
    B = agent_dummies(sales["buyer_agent_id"], "buyer")
    X = pd.concat([build_features(sales), S, B], axis=1)

    # fit_intercept gives a baseline; with a dummy for every agent the columns overlap,
    # so we measure each agent against the average agent below.
    model = LinearRegression().fit(X, y)
    coefs = pd.Series(model.coef_, index=X.columns)

    seller = coefs[S.columns]
    buyer = coefs[B.columns]
    seller.index = seller.index.str.replace("seller_", "")
    buyer.index = buyer.index.str.replace("buyer_", "")
    # Center each side so 0 means "average agent" (only differences are meaningful).
    out = pd.DataFrame({"seller_effect": seller - seller.mean()})
    out = out.join(pd.DataFrame({"buyer_effect": buyer - buyer.mean()}), how="outer")
    out["listings"] = sales["listing_agent_id"].value_counts()
    out["buyer_deals"] = sales["buyer_agent_id"].value_counts()
    return out, model.score(X, y)


def main():
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    skills = pd.read_csv(DATA / "true_skills.csv").set_index("agent_id")

    effects, r2 = fit_agent_effects(sales)
    effects = effects.join(skills)
    # In the fake data a better buyer agent LOWERS price, so flip the sign to compare.
    effects["true_buyer_effect"] = -effects["buyer_savings"]

    # Naive ranking from 2A, for comparison.
    naive = sales.groupby("listing_agent_id")["sold_price"].mean()
    effects["naive"] = naive

    def rank_agree(a, b):
        return a.rank().corr(b.rank())  # rank correlation: 1 = same order

    s = effects.dropna(subset=["seller_effect", "seller_premium", "naive"])
    b = effects.dropna(subset=["buyer_effect", "true_buyer_effect"])
    print("MODEL (2B) vs ANSWER KEY")
    print("Share of price variation explained (R-squared):", round(r2, 2))
    print("Seller side, rank agreement with truth: naive =", round(rank_agree(s["naive"], s["seller_premium"]), 2),
          "| model =", round(rank_agree(s["seller_effect"], s["seller_premium"]), 2))
    print("Buyer side, rank agreement with truth: model =",
          round(rank_agree(b["buyer_effect"], b["true_buyer_effect"]), 2))
    top_m = set(s.nlargest(10, "seller_effect").index)
    top_t = set(s.nlargest(10, "seller_premium").index)
    print("Model top 10 sellers that are truly top 10:", len(top_m & top_t), "(naive got 6)")

    # Small samples: error is bigger for agents with few listings.
    s = s.assign(error=(s["seller_effect"] - (s["seller_premium"] - s["seller_premium"].mean())).abs())
    big = s["listings"] >= 20
    print("Average error (percentage points): agents with 20+ listings =",
          round(100 * s.loc[big, "error"].mean(), 2), "| under 20 =", round(100 * s.loc[~big, "error"].mean(), 2))
    effects.to_csv(DATA / "model_effects.csv")
    print("Saved model_effects.csv")


if __name__ == "__main__":
    main()
