"""Step 6-0: each agent's seller score over time (by year and by half-year), with ranges.

Same model as 2B (home, zip, month, trend, county records, sale history), but each listing
agent gets one credit per period instead of one overall. Within each period the credits are
centered, so 0 = the average agent IN THAT PERIOD (a hot spring market lifts everyone, not
one agent: the month and trend columns take that out first).

Each period's credit is shrunk toward 0 with the same method as 2C, and its 95% range gets
the same bias allowance as the coaching table (config/scoring.json), so ranges here are never
narrower than the table's.

Trap (small samples): splitting 3 years into halves leaves most agents with a handful of
listings per period. Periods with fewer than `small_sample` listings are flagged, and big
jumps between periods are usually luck, not a change in skill (regression to the mean: a
lucky period tends to be followed by a more ordinary one).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from agent_model import DATA, agent_dummies, build_features
from score_agents import Z95, load_config
from shrinkage import shrink


def period_labels(dates, kind):
    """'2025' for years, '2025 H1' / '2025 H2' for half-years."""
    d = pd.to_datetime(dates)
    if kind == "year":
        return d.dt.year.astype(str)
    return d.dt.year.astype(str) + " H" + np.where(d.dt.month <= 6, "1", "2")


def seller_trends(sales, kind, cfg):
    """One row per (agent, period) with a shrunk seller score and its 95% range."""
    period = period_labels(sales["sold_date"], kind)
    key = sales["listing_agent_id"] + "|" + period
    y = np.log(sales["sold_price"])
    S = pd.get_dummies(key, prefix="sp", prefix_sep="=", dtype=float)
    B = agent_dummies(sales["buyer_agent_id"], "buyer")
    X = pd.concat([build_features(sales), S, B], axis=1)
    model = LinearRegression().fit(X, y)
    resid_sd = float(np.std(y - model.predict(X)))

    coefs = pd.Series(model.coef_, index=X.columns)[S.columns]
    coefs.index = coefs.index.str.replace("sp=", "")
    cells = coefs.index.to_series().str.split("|", expand=True)
    raw = pd.DataFrame({"agent_id": cells[0], "period": cells[1], "raw": coefs})
    raw["raw"] -= raw.groupby("period")["raw"].transform("mean")   # 0 = average agent that period
    deals = key.value_counts()

    rows = []
    for p, g in raw.groupby("period"):
        sh = shrink(g["raw"], deals, resid_sd)
        rows.append(g.join(sh[["score", "low", "high", "deals"]]))
    out = pd.concat(rows)
    # Same bias allowance as the coaching table, combined like two independent errors.
    room = (Z95 * cfg.get("bias_allowance", 0)) ** 2
    out["low"] = out["score"] - np.sqrt((out["score"] - out["low"]) ** 2 + room)
    out["high"] = out["score"] + np.sqrt((out["high"] - out["score"]) ** 2 + room)
    out["deals"] = out["deals"].astype(int)
    out["small_sample"] = out["deals"] < cfg["small_sample"]
    out["kind"] = kind
    return out[["kind", "agent_id", "period", "score", "low", "high", "deals", "small_sample"]] \
        .sort_values(["agent_id", "period"]).reset_index(drop=True)


def all_trends(sales, cfg):
    return pd.concat([seller_trends(sales, k, cfg) for k in ["year", "half"]], ignore_index=True)


def main():
    cfg = load_config()
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    t = all_trends(sales, cfg)
    print("SELLER SCORE OVER TIME (0 = average agent in that period)")
    for kind, g in t.groupby("kind"):
        clear = ((g["low"] > 0) | (g["high"] < 0)).mean()
        print(f"  by {kind:4s}: {g['period'].nunique()} periods, median {g['deals'].median():.0f} listings per "
              f"agent-period, {g['small_sample'].mean():.0%} flagged small, {clear:.0%} clearly above or below")
    skills_file = DATA / "true_skills.csv"
    if skills_file.exists():  # fake data only: skill is constant over time, so every period should track it
        truth = pd.read_csv(skills_file).set_index("agent_id")["seller_premium"]
        tr = (truth - truth.mean()).reindex(t["agent_id"]).to_numpy()
        for kind, m in [("year", t["kind"] == "year"), ("half", t["kind"] == "half")]:
            caught = ((tr[m] >= t.loc[m, "low"]) & (tr[m] <= t.loc[m, "high"])).mean()
            print(f"  by {kind:4s}: 95% ranges caught the (constant) true skill {caught:.0%} of the time")
    t.round(4).to_csv(DATA / "trends.csv", index=False)
    print("Saved trends.csv")


if __name__ == "__main__":
    main()
