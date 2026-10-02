"""Step 2A: the naive ranking. Rank listing agents by their average sold price,
then grade that ranking against the answer key (true_skills.csv).
The point is to see how wrong the obvious approach is."""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "fake"


def naive_ranking(sales):
    """One row per listing agent: how many listings and their average sold price."""
    return (
        sales.groupby("listing_agent_id")
        .agg(listings=("sold_price", "size"), avg_sold_price=("sold_price", "mean"))
        .sort_values("avg_sold_price", ascending=False)
    )


def main():
    sales = pd.read_csv(DATA / "sales.csv")
    skills = pd.read_csv(DATA / "true_skills.csv").set_index("agent_id")

    naive = naive_ranking(sales).join(skills["seller_premium"])
    naive["naive_rank"] = naive["avg_sold_price"].rank(ascending=False).astype(int)
    naive["true_rank"] = naive["seller_premium"].rank(ascending=False).astype(int)

    # Correlation of the two rank columns (called Spearman) = how well two rankings agree (1 = identical, 0 = unrelated).
    agree = naive["naive_rank"].corr(naive["true_rank"])
    top_naive = set(naive.nsmallest(10, "naive_rank").index)
    top_true = set(naive.nsmallest(10, "true_rank").index)

    print("NAIVE RANKING (average sold price) vs ANSWER KEY (true seller skill)")
    print("Agents ranked:", len(naive))
    print("Rank agreement with truth (Spearman, 1 = perfect):", round(agree, 2))
    print("Of the naive top 10, how many are truly top 10:", len(top_naive & top_true))
    print("\nNaive top 10:")
    print(
        naive.nsmallest(10, "naive_rank")[
            ["listings", "avg_sold_price", "seller_premium", "naive_rank", "true_rank"]
        ].round({"avg_sold_price": -3, "seller_premium": 4})
    )
    print("\nMedian listings: naive top 10 =", int(naive.nsmallest(10, "naive_rank")["listings"].median()),
          "| all agents =", int(naive["listings"].median()))


if __name__ == "__main__":
    main()
