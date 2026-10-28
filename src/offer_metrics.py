"""Step 5-2: fair offer metrics.

Raw win rate (wins / offers made, offer_win_rates.py) has two problems:
  1. Competition: winning 1 of 8 offers is harder than winning 1 of 1.
  2. Price: in real deals the highest offer usually wins, so an agent whose clients simply
     bid more wins more. That rewards overpaying, the opposite of good buyer advice.

Fair offer score (buyer agents). Every listing is its own contest with one winner. We fit a
"contest model" (a conditional logit, the standard model for picking 1 winner out of N): an
offer's chance of winning depends on its price relative to the other offers on the SAME home,
its terms (cash, down payment, contingencies, days to close), and the buyer agent. Because
every offer in a contest is on the same home, hidden home quality cancels out.
The agent part is what's left: in one sentence, "on the same home, at the same price and
terms, how much more often this agent's offers win". Listings with a single offer carry no
information (nobody to beat) and are skipped. A ridge penalty pulls agents with few offers
toward average (shrinkage), and its strength is picked by 5-fold cross-validation.
We report it as win-chance points in a typical 3-offer contest (average agent = 33%).

Offers attracted (listing agents): offers a listing drew vs what similar listings drew.
Two versions: home and market only, and also allowing for the list price. Listing low draws
more offers (a pricing tactic), so only the second version is fair. Context only, not scored.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from agent_model import DATA, build_features

TERMS = ["price_pct", "cash", "down_10_pts", "inspection", "appraisal", "loan", "close_weeks"]
PENALTIES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
Z95 = 1.96


# ---------- Contest model (conditional logit) ----------

def contest_data(offers):
    """Offers on listings with 2+ offers, sorted so each listing's offers sit together."""
    o = offers[offers.groupby("listing_id")["listing_id"].transform("size") > 1]
    o = o.sort_values("listing_id", kind="stable").reset_index(drop=True)
    log_p = np.log(o["offer_price"])
    loan = o["financing"] == "loan"
    X = pd.DataFrame({
        "price_pct": 100 * (log_p - log_p.groupby(o["listing_id"]).transform("mean")),  # % above the other offers
        "cash": (~loan).astype(float),
        "down_10_pts": np.where(loan, (o["down_payment_pct"] - 20) / 10, 0.0),
        "inspection": o["inspection_contingency"].astype(float),
        "appraisal": o["appraisal_contingency"].astype(float),
        "loan": o["loan_contingency"].astype(float),
        "close_weeks": (o["days_to_close"] - 21) / 7,
    })
    A = pd.get_dummies(o["buyer_agent_id"].replace("", np.nan), dtype=float)   # blank = no agent, baseline
    starts = np.flatnonzero(np.r_[True, o["listing_id"].to_numpy()[1:] != o["listing_id"].to_numpy()[:-1]])
    return {"Z": np.hstack([X.to_numpy(), A.to_numpy()]), "agents": list(A.columns), "starts": starts,
            "won": (o["outcome"] == "won").to_numpy(), "listing": o["listing_id"].to_numpy(),
            "n_terms": X.shape[1], "agent_ids": o["buyer_agent_id"].to_numpy()}


def _probs(theta, Z, starts):
    """Each offer's chance of winning its contest, and the log of each contest's total."""
    s = Z @ theta
    sizes = np.diff(np.r_[starts, len(s)])
    m = np.repeat(np.maximum.reduceat(s, starts), sizes)
    e = np.exp(s - m)
    tot = np.add.reduceat(e, starts)
    return s, e / np.repeat(tot, sizes), m[starts] + np.log(tot)


def _loss(theta, Z, starts, won, penalty, n_terms):
    s, p, lse = _probs(theta, Z, starts)
    a = theta[n_terms:]
    loss = -(s[won].sum() - lse.sum()) + 0.5 * penalty * a @ a
    grad = -(Z[won].sum(axis=0) - Z.T @ p) + penalty * np.r_[np.zeros(n_terms), a]
    return loss, grad


def fit_contests(d, penalty, rows=None):
    """Fit on all contests, or only the offers in `rows` (whole listings)."""
    Z, starts, won = d["Z"], d["starts"], d["won"]
    if rows is not None:
        Z, won = Z[rows], won[rows]
        lst = d["listing"][rows]
        starts = np.flatnonzero(np.r_[True, lst[1:] != lst[:-1]])
    res = optimize.minimize(_loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                            args=(Z, starts, won, penalty, d["n_terms"]))
    return res.x


def pick_penalty(d, penalties=PENALTIES, folds=5):
    """Cross-validation: fit on 4/5 of the listings, score how well we predict the winners of the rest."""
    listings = np.unique(d["listing"])
    score = {}
    for pen in penalties:
        total = 0.0
        for tr, te in KFold(folds, shuffle=True, random_state=0).split(listings):
            train_rows = np.isin(d["listing"], listings[tr])
            theta = fit_contests(d, pen, train_rows)
            Z, won, lst = d["Z"][~train_rows], d["won"][~train_rows], d["listing"][~train_rows]
            starts = np.flatnonzero(np.r_[True, lst[1:] != lst[:-1]])
            s, _, lse = _probs(theta, Z, starts)
            total += s[won].sum() - lse.sum()   # held-out log-likelihood: higher = better predictions
        score[pen] = total
    return max(score, key=score.get), score


def win_chance(a, others=2):
    """Turn an agent effect (log-odds) into the win chance in a contest with `others` average rivals."""
    return np.exp(a) / (np.exp(a) + others) - 1 / (others + 1)


def fair_offer_scores(offers, penalty=None):
    """One row per buyer agent: fair offer score with a 95% range (win-chance points in a
    3-offer contest), offers in contests, and the fitted value of each term."""
    d = contest_data(offers)
    if penalty is None:
        penalty, _ = pick_penalty(d)
    theta = fit_contests(d, penalty)
    k = d["n_terms"]
    # Uncertainty: curvature of the fit (Hessian). Its inverse gives each estimate's variance.
    _, p, _ = _probs(theta, d["Z"], d["starts"])
    Z = d["Z"]
    M = np.add.reduceat(p[:, None] * Z, d["starts"])
    H = Z.T @ (p[:, None] * Z) - M.T @ M + penalty * np.diag(np.r_[np.zeros(k), np.ones(Z.shape[1] - k)])
    se = np.sqrt(np.diag(np.linalg.inv(H)))[k:]
    a = theta[k:] - theta[k:].mean()   # 0 = the average agent
    n_offers = pd.Series(d["agent_ids"]).value_counts()
    table = pd.DataFrame({"effect": a, "effect_se": se}, index=pd.Index(d["agents"], name="agent_id"))
    table["score"] = win_chance(table["effect"])
    table["low"] = win_chance(table["effect"] - Z95 * table["effect_se"])
    table["high"] = win_chance(table["effect"] + Z95 * table["effect_se"])
    table["offers"] = n_offers.reindex(table.index).fillna(0).astype(int)
    terms = pd.Series(theta[:k], index=TERMS)
    table.attrs.update(penalty=penalty, terms=terms)
    return table


def term_values(terms):
    """What each term is worth to sellers, in % of price (term weight / price weight)."""
    per_pct = terms["price_pct"]
    return (terms.drop("price_pct") / per_pct).rename({"close_weeks": "per week to close",
                                                      "down_10_pts": "per 10 pts down"})


def competition_adjusted(offers):
    """Middle step for comparison: wins minus expected wins (1/N per offer), per offer made."""
    o = offers[offers["buyer_agent_id"] != ""]
    g = o.assign(won=o["outcome"] == "won", expected=1 / o["competing_offers"]).groupby("buyer_agent_id")
    return (g["won"].sum() - g["expected"].sum()) / g.size()


# ---------- Offers attracted (listing agents) ----------

def offers_attracted(sales, offers, predictions):
    """Per listing agent: offers drawn vs similar listings, as a ratio - 1 (+0.2 = 20% more),
    shrunk toward 0, with a 95% range. Version 'home' adjusts for home and market only;
    version 'fair' also allows for how high the list price was set vs model value."""
    n = offers.groupby("listing_id").size().reindex(sales["listing_id"]).to_numpy()
    X = build_features(sales).astype(float)
    list_vs_value = np.log(sales["original_list_price"] / predictions.set_index("listing_id")
                           ["predicted_at_list"].reindex(sales["listing_id"]).to_numpy()).to_numpy()
    out = {}
    for version, cols in [("home", X), ("fair", X.assign(list_vs_value=list_vs_value))]:
        expected = np.zeros(len(sales))
        for tr, te in KFold(5, shuffle=True, random_state=42).split(cols):
            scale = StandardScaler().fit(cols.iloc[tr])
            m = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(scale.transform(cols.iloc[tr]), n[tr])
            expected[te] = m.predict(scale.transform(cols.iloc[te]))
        g = pd.DataFrame({"got": n, "expected": expected}).groupby(sales["listing_agent_id"].to_numpy()).sum()
        ratio = g["got"] / g["expected"]
        # Shrinkage for counts (Poisson-gamma): blend in k listings' worth of "average".
        real_var = max(ratio.var() - (1 / g["expected"]).mean(), 1e-4)
        k = 1 / real_var
        shape, rate = g["got"] + k, g["expected"] + k
        out[version] = pd.DataFrame({"score": shape / rate - 1,
                                     "low": stats.gamma.ppf(0.025, shape, scale=1 / rate) - 1,
                                     "high": stats.gamma.ppf(0.975, shape, scale=1 / rate) - 1})
    both = pd.concat(out, axis=1)
    both.columns = [f"attract_{v}_{c}" for v, c in both.columns]
    both.index.name = "agent_id"
    return both


def offer_table(sales, offers, predictions, penalty=None):
    """What score_agents.py reads: fair offer score with range per buyer agent, and offers
    attracted per listing agent (both versions)."""
    fair = fair_offer_scores(offers, penalty)
    attract = offers_attracted(sales, offers, predictions)
    out = fair.drop(columns=["effect", "effect_se"]).join(attract, how="outer")
    out.attrs.update(fair.attrs)
    return out


def main():
    import price_model
    from offer_win_rates import win_rates

    offers = pd.read_csv(DATA / "offers.csv", keep_default_na=False)
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    fair = fair_offer_scores(offers)
    print(f"CONTEST MODEL: {len(contest_data(offers)['won'])} offers in contests with 2+ offers "
          f"| penalty picked by cross-validation: {fair.attrs['penalty']}")
    print("What sellers treat each term as worth, in % of price:")
    print(term_values(fair.attrs["terms"]).round(2).to_string())

    raw = win_rates(offers)
    comp = competition_adjusted(offers)
    skills_file = DATA / "true_skills.csv"
    if skills_file.exists():   # fake data only: grade against the answer key
        sk = pd.read_csv(skills_file).set_index("agent_id")
        print("\nRank agreement with hidden win skill | with buyer savings (should be near 0: not rewarding overpaying)")
        for label, x in [("Raw win rate (shrunk)", raw["win_rate"]), ("Adjusted for competition", comp),
                         ("Fair offer score", fair["score"])]:
            r = lambda col: x.rank().corr(sk[col].reindex(x.index).rank())
            print(f"  {label:26s} {r('win_skill'):5.2f} | {r('buyer_savings'):5.2f}")

    attract = offers_attracted(sales, offers, price_model.cross_validated_predictions(sales))
    if skills_file.exists():
        for v in ["home", "fair"]:
            x = attract[f"attract_{v}_score"]
            print(f"Offers attracted ({v}), rank agreement with seller skill: "
                  f"{x.rank().corr(sk['seller_premium'].reindex(x.index).rank()):.2f}")
    out = fair.drop(columns=["effect", "effect_se"]).join(attract, how="outer")
    out.round(4).to_csv(DATA / "offer_scores.csv", index_label="agent_id")
    print("Saved offer_scores.csv")


if __name__ == "__main__":
    main()
