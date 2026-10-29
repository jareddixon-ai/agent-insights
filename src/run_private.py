"""Phase 7: score a private copy (a brokerage's own data) end to end, with no answer key.

  1. Runs the readiness check (src/readiness_check.py). Stops if the folder could be published
     or something required is missing.
  2. Stores ONLY the columns the metrics use, in our names, in <data_folder>/standard/.
     Never stored: addresses, client names, emails, phones, notes, any unmapped column.
  3. Scores with the same code as the fake-data pipeline and writes <data_folder>/scored/:
     rankings.csv and trends.csv (what the dashboard reads), plus the pieces behind them.
     Weights and the "validation" block come from config/scoring.json, unchanged: buyer and offer
     scores stay context only until a private validation shows they work on this data.

Run:  python src/run_private.py                    (1,000 bootstrap resamples, a few minutes)
      python src/run_private.py --mapping config/column_mapping_demo.json --resamples 200
"""

import argparse
from pathlib import Path

import pandas as pd

import bootstrap_ranges as br
import offer_metrics
import price_model
import score_agents
import speed_and_pricing
import trends
from readiness_check import CONTEXT, MAPPING, UnsafeFolder, check, load_mapping, resolve_folder

OFFER_SCORE_COLS = ["score", "low", "high", "offers", "attract_fair_score", "attract_fair_low", "attract_fair_high"]
SPEED_COLS = ["speed_vs_similar", "speed_low", "speed_high", "list_vs_value", "list_vs_value_low",
              "list_vs_value_high", "share_with_price_cut", "cut_low", "cut_high"]


def empty(columns):
    return pd.DataFrame(columns=columns, dtype=float).rename_axis("agent_id")


def score(tables, turned_off, cfg, n_resamples=br.N_RESAMPLES):
    """Same steps as the fake-data pipeline, minus everything that needs the answer key."""
    sales = tables["sales"].copy()
    for col in ["list_date", "sold_date", "prior_sale_date"]:
        if col in sales:
            sales[col] = pd.to_datetime(sales[col]).dt.strftime("%Y-%m-%d")
    pred = price_model.cross_validated_predictions(sales)
    seller_full, buyer_full = br.shrunk_scores(sales)
    seller_draws, buyer_draws = br.bootstrap(sales, n_resamples)
    boot = pd.concat({"seller": br.summarize(seller_full, seller_draws), "buyer": br.summarize(buyer_full, buyer_draws)},
                     names=["side", "agent_id"])

    has_context = "speed_pricing" not in turned_off
    ctx_sales = sales.dropna(subset=CONTEXT).reset_index(drop=True) if has_context else None
    speed = speed_and_pricing.speed_and_pricing(ctx_sales, pred) if has_context else empty(SPEED_COLS)
    if "offers" in tables and "offer_score" not in turned_off:
        offers = tables["offers"]
        fair = offer_metrics.fair_offer_scores(offers)
        if has_context:
            attract = offer_metrics.offers_attracted(ctx_sales, offers, pred)
            offer = fair.drop(columns=["effect", "effect_se"]).join(attract, how="outer")
        else:
            offer = fair.drop(columns=["effect", "effect_se"]).reindex(columns=OFFER_SCORE_COLS)
    else:
        offer = empty(OFFER_SCORE_COLS)
    table = score_agents.build_table(boot, offer, speed, tables["agents"], cfg)
    return {"predictions": pred, "bootstrap_scores": boot, "speed_pricing": speed, "offer_scores": offer,
            "rankings": table, "trends": trends.all_trends(sales, cfg)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a private copy.")
    parser.add_argument("--mapping", default=str(MAPPING))
    parser.add_argument("--data", default=None, help="data folder (overrides data_folder in the mapping)")
    parser.add_argument("--resamples", type=int, default=br.N_RESAMPLES)
    args = parser.parse_args(argv)
    mapping = load_mapping(args.mapping)
    try:
        report = check(mapping, args.data)
    except UnsafeFolder as e:
        raise SystemExit(str(e))
    print(report.text())
    if not report.ready:
        raise SystemExit(1)

    folder = resolve_folder(mapping, args.data)
    standard, scored = folder / "standard", folder / "scored"
    standard.mkdir(exist_ok=True)
    scored.mkdir(exist_ok=True)
    for name, t in report.tables.items():          # only our columns: nothing personal was ever read
        t.to_csv(standard / f"{name}.csv", index=False, date_format="%Y-%m-%d")

    print(f"\nScoring with {args.resamples} bootstrap resamples...")
    out = score(report.tables, report.turned_off, score_agents.load_config(), args.resamples)
    for name, t in out.items():
        index = name not in {"predictions", "trends"}
        t.round(4).to_csv(scored / f"{name}.csv", index=index, index_label="agent_id" if name != "bootstrap_scores" else None)
    groups = out["rankings"]["index_group"].value_counts()
    print("Coaching index: " + ", ".join(f"{k} {v}" for k, v in groups.items()))
    print(f"Saved to {scored}. Open the dashboard with:\n"
          f"  AGENT_INSIGHTS_DATA=\"{scored}\" streamlit run app/coaching_app.py")


if __name__ == "__main__":
    main()
