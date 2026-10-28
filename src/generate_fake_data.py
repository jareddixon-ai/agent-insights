"""Generate fake data for agent-insights. Fake data only, never real listings."""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42  # fixed seed: same random numbers every run, so results repeat
N_AGENTS = 150
OFFICES = ["Office A", "Office B", "Office C"]
N_TEAMS = 8
SHARE_ON_A_TEAM = 0.40  # the other 60% work solo
OUT_DIR = Path("data/fake")
START_DATE = "2023-10-01"  # earliest list date (about 3 years of listings)
N_LIST_DAYS = 1003  # list dates run to mid-2026
SELECTION_STRENGTH = 0.4  # how strongly better agents get pricier homes
NO_BUYER_AGENT_SHARE = 0.08  # share of sales where the buyer had no agent
# California tax values (Prop 13): a home's assessed value is set to its purchase price
# and then rises at most 2% a year until it sells again. Prices grow faster than that,
# so the longer an owner has held a home, the further its assessed value lags behind.
ASSESSED_STYLE = "prop13"  # "prop13" (California) or "market" (assessed value near market value)
PROP13_CAP = 0.02  # most an assessed value can rise in a year
LONG_RUN_GROWTH = 0.055  # how fast prices rose per year before our 3 years of sales
ESTATE_SHARE = 0.15  # share of agents who mostly list long-held homes (estates, older owners)
MEAN_YEARS_OWNED = (13, 28)  # average years since the last sale: other agents, estate specialists
# (gives a median near 11 years, in line with US sellers in NAR's 2025 Profile of Home Buyers and Sellers)
STREETS = ["Oak", "Maple", "Cedar", "Pine", "Elm", "Birch", "Willow", "Aspen"]


def make_agents(rng, buyer_spread=0.010):
    """Build the agents table and the hidden-skills table.
    buyer_spread: how much buyer agents really differ (stress tests can change it)."""
    agent_ids = [f"A{i:03d}" for i in range(1, N_AGENTS + 1)]

    # Each agent gets one office (roughly equal sizes).
    offices = rng.choice(OFFICES, size=N_AGENTS)

    # Teams live inside one office. About 40% of agents join a team.
    team_names = [f"Team {i}" for i in range(1, N_TEAMS + 1)]
    team_office = {t: OFFICES[i % len(OFFICES)] for i, t in enumerate(team_names)}
    teams = []
    for office in offices:
        options = [t for t in team_names if team_office[t] == office]
        on_team = rng.random() < SHARE_ON_A_TEAM
        teams.append(rng.choice(options) if on_team else "")

    # Years licensed: many newer agents, fewer veterans (0 to 35 years).
    years = np.clip(rng.gamma(shape=2.0, scale=5.0, size=N_AGENTS), 0, 35).astype(int)

    agents = pd.DataFrame(
        {"agent_id": agent_ids, "office": offices, "team": teams, "years_licensed": years}
    )

    # Hidden skills: the "answer key". Independent of each other.
    # Bell-curve shaped (most agents near average), cut off at the allowed limits.
    seller_premium = np.clip(rng.normal(0.005, 0.012, N_AGENTS), -0.03, 0.04)
    buyer_savings = np.clip(rng.normal(0.005, buyer_spread, N_AGENTS), -0.02, 0.03)
    skills = pd.DataFrame(
        {
            "agent_id": agent_ids,
            "seller_premium": seller_premium.round(4),
            "buyer_savings": buyer_savings.round(4),
        }
    )
    # Hidden "busyness": some agents get many deals, some very few.
    # Veterans are a bit busier. Not saved to any file (real data has no such column).
    activity = rng.lognormal(0.0, 1.0, N_AGENTS) * (0.5 + years / 15)
    return agents, skills, activity

N_SALES = 5000
# Real Santa Barbara-area zip codes. The number is a rough price level for that area,
# in log terms (0.50 means about 65% above the baseline, -0.15 about 14% below it).
ZIP_EFFECTS = {
    "93108": 0.50,   # Montecito: the priciest
    "93110": 0.20,   # Hope Ranch / Upper State
    "93067": 0.10,   # Summerland
    "93109": 0.00,   # Mesa
    "93105": -0.05,  # Upper East / San Roque
    "93103": -0.05,  # Eastside / Riviera
    "93101": -0.10,  # Downtown
    "93111": -0.15,  # La Cumbre / Hollister
    "93013": -0.15,  # Carpinteria
    "93117": -0.20,  # Goleta
}
ZIPS = list(ZIP_EFFECTS)
PRICE_FLOOR = 2_000_000  # nothing sells far below this in the target market


def make_homes(rng, seed=SEED):
    """Build 5,000 fake homes and their true market value (what each home is worth
    before any agent skill). The true value is the answer key for later steps."""
    n = N_SALES

    zip_code = rng.choice(ZIPS, size=n)
    zip_effect = ZIP_EFFECTS

    # Size: most homes near 2,400 sqft, with a long tail of very large estates.
    sqft = np.clip(rng.lognormal(np.log(2400), 0.45, n), 900, 15000).round(-1).astype(int)
    # Bedrooms and bathrooms grow with size (plus randomness).
    beds = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 7).astype(int)
    baths = np.clip(np.round((beds * 0.7 + rng.normal(0.3, 0.6, n)) * 2) / 2, 1, 6)
    lot_sqft = np.clip(rng.lognormal(np.log(9000), 0.9, n), 2000, 250000).round(-2).astype(int)
    year_built = rng.integers(1925, 2024, n)
    age = 2024 - year_built

    # "Quality" = things we can't see in the data (condition, view, finishes).
    # It changes the price but no column records it.
    quality = rng.normal(0.0, 0.12, n)
    # Rare trophy properties (ocean-front, big estates) sit far above the pack.
    # This is what gives the price list its heavy upper tail.
    trophy = np.where(rng.random(n) < 0.02, np.clip(rng.normal(1.3, 0.5, n), 0.3, 2.4), 0.0)

    # Log price = sum of effects. (Log means each effect is a % change, which
    # matches how we describe agent skill, in %.)
    log_value = (
        14.2
        + np.array([zip_effect[z] for z in zip_code])
        + 0.75 * np.log(sqft / 2400)
        + 0.03 * (beds - 3)
        + 0.06 * (baths - 2)
        + 0.10 * np.log(lot_sqft / 9000)
        - 0.002 * age
        + quality
        + trophy
    )
    # Shifted so the cheapest homes sit near the floor instead of far below it.
    value = PRICE_FLOOR + np.exp(log_value)

    # County-style assessed value: a public estimate of the home's value, a few percent
    # off the truth. Real brokerages have something like it. Drawn from its own random
    # stream so every other column stays identical to before.
    assess_rng = np.random.default_rng([seed, 1])
    assessed = value * np.exp(assess_rng.normal(0.0, 0.045, n))

    homes = pd.DataFrame(
        {
            "listing_id": [f"L{i:05d}" for i in range(1, n + 1)],
            "zip": zip_code,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "lot_sqft": lot_sqft,
            "year_built": year_built,
            "assessed_value": assessed.round(-3),
            "true_value": value.round(-3),
            "quality": quality.round(4),
        }
    )
    return homes


def pick_agents(rng, activity, skill, home_z, strength):
    """For each home, pick one agent. Busier agents are picked more often, and
    better agents are more likely to be picked for pricier homes (selection bias)."""
    skill_z = (skill - skill.mean()) / skill.std()
    # score[home, agent] = how likely that agent is to get that home
    score = activity[None, :] * np.exp(strength * home_z[:, None] * skill_z[None, :])
    cdf = np.cumsum(score / score.sum(axis=1, keepdims=True), axis=1)
    u = rng.random(len(home_z))[:, None]
    return np.minimum((u > cdf).sum(axis=1), len(skill) - 1)  # index of chosen agent


def market_factor(dates):
    """Market effect: a spring bump and a slow upward trend. Not the agent's doing."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    month = dates.month.to_numpy()
    years_in = (dates - pd.Timestamp(START_DATE)).days.to_numpy() / 365
    return np.exp(0.03 * np.cos(2 * np.pi * (month - 5) / 12) + 0.04 * years_in)


def make_sales(rng, homes, agents, skills, activity, selection_strength=SELECTION_STRENGTH):
    """Turn homes into sales: pick agents, add dates, prices, list-price games.
    selection_strength > 0: better agents get pricier homes. < 0: cheaper homes. 0: random."""
    n = len(homes)
    home_z = ((np.log(homes["true_value"]) - np.log(homes["true_value"]).mean())
              / np.log(homes["true_value"]).std()).to_numpy()

    # Who sold it, and who bought it.
    li = pick_agents(rng, activity, skills["seller_premium"].to_numpy(), home_z, selection_strength)
    bi = pick_agents(rng, activity, skills["buyer_savings"].to_numpy(), home_z, selection_strength)
    bi = np.where(bi == li, (bi + 1) % len(agents), bi)  # an agent can't be on both sides
    has_buyer_agent = rng.random(n) >= NO_BUYER_AGENT_SHARE

    # Dates. Days on market is random (about a month on average).
    list_date = pd.Timestamp(START_DATE) + pd.to_timedelta(rng.integers(0, N_LIST_DAYS, n), unit="D")

    # Each listing agent has a pricing style: some list low to spark bidding,
    # some list high. Hidden, and it does not change what the home sells for.
    style = rng.normal(0.0, 0.04, len(agents))
    list_bias = style[li] + rng.normal(0.0, 0.02, n)
    n_changes = np.where(list_bias > 0.02, np.minimum(rng.poisson(1 + np.maximum(list_bias, 0) * 30), 4), 0)
    days = np.maximum(3, rng.gamma(2.0, 15.0, n) + 12 * n_changes).astype(int)
    sold_date = list_date + pd.to_timedelta(days, unit="D")

    market = market_factor(sold_date)

    # Sold price: true value, times the market, times BOTH agents' effects.
    seller_effect = 1 + skills["seller_premium"].to_numpy()[li]
    buyer_effect = np.where(has_buyer_agent, 1 - skills["buyer_savings"].to_numpy()[bi], 1.0)
    noise = np.exp(rng.normal(0.0, 0.02, n))  # bidding luck, small
    sold_price = homes["true_value"].to_numpy() * market * seller_effect * buyer_effect * noise

    original_list = sold_price * (1 + list_bias)
    final_list = original_list * (0.97 ** n_changes)

    sales = pd.DataFrame(
        {
            "listing_id": homes["listing_id"],
            "address": [f"{a} Demo {STREETS[b]} St" for a, b in
                        zip(rng.integers(100, 9999, n), rng.integers(0, len(STREETS), n))],
            "zip": homes["zip"],
            "beds": homes["beds"],
            "baths": homes["baths"],
            "sqft": homes["sqft"],
            "lot_sqft": homes["lot_sqft"],
            "year_built": homes["year_built"],
            "assessed_value": homes["assessed_value"],
            "list_date": list_date.date,
            "original_list_price": original_list.round(-3),
            "final_list_price": final_list.round(-3),
            "num_price_changes": n_changes,
            "sold_date": sold_date.date,
            "sold_price": sold_price.round(-3),
            "days_on_market": days,
            "listing_agent_id": agents["agent_id"].to_numpy()[li],
            "buyer_agent_id": np.where(has_buyer_agent, agents["agent_id"].to_numpy()[bi], ""),
            "office": agents["office"].to_numpy()[li],
            "team": agents["team"].to_numpy()[li],
        }
    )
    return sales


def add_sale_history(sales, homes, agents, seed=SEED):
    """California version of the public records: when the home last sold, for how much,
    and the Prop 13 assessed value that follows from it. Own random stream, so every
    other column stays identical. Returns the new sales table and the hidden estate flag."""
    rng = np.random.default_rng([seed, 4])
    ids = agents["agent_id"].to_numpy()
    # Some agents specialize in long-held homes. Not a skill, just the kind of client they serve.
    estate = pd.Series(rng.random(len(ids)) < ESTATE_SHARE, index=ids)
    is_estate = estate.reindex(sales["listing_agent_id"]).to_numpy()
    listed = pd.to_datetime(sales["list_date"])
    age = (listed.dt.year - sales["year_built"]).to_numpy()
    years = rng.gamma(2.0, np.where(is_estate, MEAN_YEARS_OWNED[1], MEAN_YEARS_OWNED[0]) / 2)
    years = np.clip(years, 0.5, np.clip(age, 0.5, 49))  # can't have owned it longer than it existed
    days_owned = (years * 365.25).astype(int)
    years = days_owned / 365.25  # whole days, so the dates and the 2% rule agree exactly

    # What it sold for back then: today's value, minus the growth since, give or take how
    # the home's condition changed (grows with time) and that sale's own bidding luck.
    condition_change = rng.normal(0.0, 0.025 * np.sqrt(years))
    luck = rng.normal(0.0, 0.03, len(sales))
    prior_price = homes["true_value"].to_numpy() * np.exp(-LONG_RUN_GROWTH * years - condition_change + luck)

    out = sales.copy()
    out["assessed_value"] = (prior_price * (1 + PROP13_CAP) ** np.floor(years)).round(-3)
    pos = out.columns.get_loc("assessed_value") + 1
    out.insert(pos, "prior_sale_date", (listed - pd.to_timedelta(days_owned, unit="D")).dt.date)
    out.insert(pos + 1, "prior_sale_price", prior_price.round(-3))
    return out, estate


CONDITION_CUTS = [-1.5, -0.5, 0.5, 1.5]  # 1-5 scale: about 7% / 24% / 38% / 24% / 7% of homes


def add_condition_rating(sales, homes, skills, accuracy, seed=SEED, puffery=0.0):
    """Phase 5-0 what-if: a 1-to-5 condition rating (1 = needs work, 5 = like new), like an
    inspector's or appraiser's grade. It is a noisy read of the hidden quality column.
    accuracy = how closely the rating tracks hidden quality before rounding to 1-5
    (1.0 = perfect read, 0 = pure guess). Rounding to 5 steps loses a little more.
    puffery > 0: the LISTING AGENT writes the rating, and better sellers rate their own
    listings higher (puffery = how many rating steps per standard deviation of skill).
    Own random stream, so every other column stays identical."""
    rng = np.random.default_rng([seed, 5])
    q = homes["quality"].to_numpy()
    z = (q - q.mean()) / q.std()
    latent = accuracy * z + np.sqrt(max(1 - accuracy**2, 0.0)) * rng.normal(0.0, 1.0, len(q))
    if puffery:
        skill = skills.set_index("agent_id")["seller_premium"]
        skill_z = ((skill - skill.mean()) / skill.std()).reindex(sales["listing_agent_id"]).to_numpy()
        latent = latent + puffery * skill_z
    out = sales.copy()
    out["condition_rating"] = 1 + np.searchsorted(CONDITION_CUTS, latent)
    return out


def add_value_estimate(sales, homes, error, seed=SEED, anchor=0.0):
    """Phase 5-0 what-if: an independent estimate of what the home is worth when it lists,
    for example a pre-listing appraisal. error = its typical miss (0.03 = about 3%).
    anchor > 0: the estimate is pulled toward the price the home actually sold for
    (anchor = 0.5 means halfway). Loan appraisals are done after the contract is signed,
    with the contract price in hand, so they behave like this. Own random stream."""
    rng = np.random.default_rng([seed, 6])
    worth_at_listing = homes["true_value"].to_numpy() * market_factor(sales["list_date"])
    log_est = np.log(worth_at_listing) + rng.normal(0.0, error, len(sales))
    if anchor:
        log_est = (1 - anchor) * log_est + anchor * np.log(sales["sold_price"].to_numpy())
    out = sales.copy()
    out["value_estimate"] = np.exp(log_est).round(-3)
    return out


# Offer terms: how much each one is worth to a seller, as a share of price (made-up sizes for the demo).
# A cash offer with no contingencies can beat a slightly higher offer that needs a loan.
TERM_VALUE = {"cash": 0.012, "per_10_pts_down": 0.003, "inspection": -0.006, "appraisal": -0.004,
              "loan": -0.006, "per_day_over_21": -0.0002}
WIN_SKILL_VALUE = 0.02    # 1 unit of hidden win_skill makes an offer look 2% stronger (presentation, timing, trust)
BID_SPREAD = 0.03         # how much competing buyers' prices differ (budgets, taste)
LOSER_LEVEL = -0.01       # losing bidders aim a little under where the winner landed, on average
CANDIDATES = 20           # would-be bidders drawn per listing; the ones who would have lost become losing offers


def offer_terms(rng, n):
    """Financing, down payment, contingencies and closing time for n offers, and how strong
    those terms make an offer look to the seller (in share of price)."""
    cash = rng.random(n) < 0.25
    down = np.where(cash, 100, rng.choice([10, 20, 25, 30, 40, 50], n, p=[0.1, 0.35, 0.2, 0.15, 0.12, 0.08]))
    inspection = rng.random(n) < 0.7
    appraisal = ~cash & (rng.random(n) < 0.6)
    loan = ~cash & (rng.random(n) < 0.75)
    days = rng.choice([14, 21, 30, 45], n, p=[0.15, 0.3, 0.4, 0.15])
    v = TERM_VALUE
    strength = (v["cash"] * cash + v["per_10_pts_down"] * np.where(cash, 0, (down - 20) / 10)
                + v["inspection"] * inspection + v["appraisal"] * appraisal + v["loan"] * loan
                + v["per_day_over_21"] * (days - 21) + rng.normal(0.0, 0.006, n))   # letters, escalation clauses...
    terms = pd.DataFrame({"financing": np.where(cash, "cash", "loan"), "down_payment_pct": down,
                          "inspection_contingency": inspection.astype(int),
                          "appraisal_contingency": appraisal.astype(int),
                          "loan_contingency": loan.astype(int), "days_to_close": days})
    return terms, strength


def make_offers(sales, agents, skills, activity, seed=SEED):
    """Every sale had one winning offer (the sold price). Add the offers that LOST, the way
    real offers work (Phase 5-1):
      - each offer's strength = its price + its terms + the buyer agent's hidden win_skill;
        the strongest offer wins, so the highest price USUALLY wins, but not always;
      - buyer agents whose clients save more (buyer_savings) also bid lower when they lose;
      - listings draw more offers when the listing agent is a stronger seller, the list price
        is low (a pricing tactic), the home is cheaper, or it is spring.
    Losing offers are would-be bidders who would NOT have beaten the real winner, so every
    sale and its sold price stay exactly as before. Own random stream: other files unchanged."""
    rng = np.random.default_rng([seed, 3])
    n, ids = len(sales), agents["agent_id"].to_numpy()
    sk = skills.set_index("agent_id").reindex(ids)
    savings, win = sk["buyer_savings"].to_numpy(), sk["win_skill"].to_numpy()
    sold, listed = pd.to_datetime(sales["sold_date"]), pd.to_datetime(sales["list_date"])
    span = (sold - listed).dt.days.to_numpy()

    # The winner. Prices are measured against the winning price: log(price / sold price).
    w_idx = pd.Series(np.arange(len(ids)), index=ids).reindex(sales["buyer_agent_id"]).to_numpy()
    has_w = ~np.isnan(w_idx)
    w_idx = np.where(has_w, w_idx, 0).astype(int)
    w_save, w_win = np.where(has_w, savings[w_idx], 0.0), np.where(has_w, win[w_idx], 0.0)
    w_terms, w_t = offer_terms(rng, n)
    w_strength = w_t + WIN_SKILL_VALUE * w_win

    # How many losing offers each listing draws.
    seller = sk["seller_premium"]
    seller_z = ((seller - seller.mean()) / seller.std()).reindex(sales["listing_agent_id"]).to_numpy()
    list_bias = (sales["original_list_price"] / sales["sold_price"] - 1).to_numpy()   # + = listed above
    log_p = np.log(sales["sold_price"].to_numpy())
    home_z = (log_p - log_p.mean()) / log_p.std()
    spring = np.cos(2 * np.pi * (sold.dt.month.to_numpy() - 4) / 12)
    expected = 1.4 * np.exp(0.35 * seller_z - 6 * list_bias - 0.3 * home_z + 0.2 * spring)
    target = np.minimum(rng.poisson(expected), 9)

    # Would-be bidders: busier agents bid more often. Not the listing agent or the winner's agent.
    c_agent = rng.choice(len(ids), size=(n, CANDIDATES), p=activity / activity.sum())
    for _ in range(2):
        clash = (ids[c_agent] == sales["listing_agent_id"].to_numpy()[:, None]) | \
                (has_w[:, None] & (c_agent == w_idx[:, None]))
        c_agent = np.where(clash, (c_agent + 1) % len(ids), c_agent)
    c_none = rng.random((n, CANDIDATES)) < NO_BUYER_AGENT_SHARE
    c_save = np.where(c_none, 0.0, savings[c_agent])
    c_win = np.where(c_none, 0.0, win[c_agent])
    winner_luck = rng.normal(0.0, 0.02, n)   # stands in for the winner's bidding luck in make_sales
    gap = (LOSER_LEVEL + (w_save[:, None] - c_save) + rng.normal(0.0, BID_SPREAD, (n, CANDIDATES))
           - winner_luck[:, None])
    c_terms, c_t = offer_terms(rng, n * CANDIDATES)
    c_strength = gap + c_t.reshape(n, CANDIDATES) + WIN_SKILL_VALUE * c_win
    loses = c_strength < w_strength[:, None]
    keep = loses & (np.cumsum(loses, axis=1) <= target[:, None])
    row, col = np.nonzero(keep)
    flat = row * CANDIDATES + col

    offset = np.floor(rng.random(len(row)) * np.maximum(span[row] - 2, 1)).astype(int)
    losers = pd.DataFrame({
        "listing_id": sales["listing_id"].to_numpy()[row],
        "buyer_agent_id": np.where(c_none[row, col], "", ids[c_agent[row, col]]),
        "days_in": offset,
        "offer_price": (sales["sold_price"].to_numpy()[row] * np.exp(gap[row, col])).round(-3),
        "outcome": "lost",
    }).join(c_terms.iloc[flat].reset_index(drop=True))
    winners = pd.DataFrame({
        "listing_id": sales["listing_id"], "buyer_agent_id": sales["buyer_agent_id"],
        "days_in": np.maximum(span - 2, 0), "offer_price": sales["sold_price"], "outcome": "won",
    }).join(w_terms)
    offers = pd.concat([winners, losers], ignore_index=True)

    # Asking price on the offer day: price cuts are spread evenly over the time on market.
    s = sales.set_index("listing_id")
    k = offers["listing_id"]
    cuts_so_far = np.floor(offers["days_in"].to_numpy() * (s["num_price_changes"].reindex(k).to_numpy() + 1)
                           / np.maximum(s["days_on_market"].reindex(k).to_numpy(), 1))
    cuts_so_far = np.minimum(cuts_so_far, s["num_price_changes"].reindex(k).to_numpy())
    offers["list_price_at_offer"] = (s["original_list_price"].reindex(k).to_numpy() * 0.97 ** cuts_so_far).round(-3)
    offers["offer_date"] = (pd.to_datetime(s["list_date"].reindex(k).to_numpy())
                            + pd.to_timedelta(offers["days_in"], unit="D")).dt.date
    offers["competing_offers"] = offers.groupby("listing_id")["listing_id"].transform("size")
    offers = offers.sort_values(["listing_id", "offer_date", "outcome"], kind="stable").reset_index(drop=True)
    offers.insert(0, "offer_id", [f"O{i:05d}" for i in range(1, len(offers) + 1)])
    cols = ["offer_id", "listing_id", "buyer_agent_id", "offer_date", "list_price_at_offer", "offer_price",
            "financing", "down_payment_pct", "inspection_contingency", "appraisal_contingency",
            "loan_contingency", "days_to_close", "competing_offers", "outcome"]
    return offers[cols]


def run_checks(sales, homes, skills):
    """Quick sanity checks. Stops the script if something is clearly wrong."""
    assert len(sales) == N_SALES
    assert sales["listing_id"].is_unique
    assert (sales["sold_price"] > 0).all()
    assert (pd.to_datetime(sales["sold_date"]) > pd.to_datetime(sales["list_date"])).all()
    assert (sales["listing_agent_id"] != sales["buyer_agent_id"]).all()
    key_cols = ["zip", "sold_price", "listing_agent_id", "list_date", "sold_date"]
    assert sales[key_cols].notna().all().all()

    per_agent = sales["listing_agent_id"].value_counts()
    print("\nSALES CHECKS")
    print("Sales:", len(sales), "| Agents with 1+ listing:", per_agent.size)
    print("Listings per agent: min", per_agent.min(), "median", int(per_agent.median()),
          "max", per_agent.max(), "| agents with under 5:", int((per_agent < 5).sum()))
    print("No buyer agent:", int((sales["buyer_agent_id"] == "").sum()), "sales")

    # Selection bias check: do better agents get pricier homes?
    d = sales.merge(homes[["listing_id", "true_value"]], on="listing_id")
    d["ratio"] = d["sold_price"] / d["true_value"]  # 1.02 means sold 2% above true value
    by_agent = d.groupby("listing_agent_id").agg(
        mean_value=("true_value", "mean"),
        mean_ratio=("ratio", "mean"),
    )
    by_agent = by_agent.join(skills.set_index("agent_id"))
    print("Correlation, seller skill vs average home value (bias, should be > 0):",
          round(by_agent["seller_premium"].corr(by_agent["mean_value"]), 2))
    print("Correlation, seller skill vs sold/true value (should be strongly > 0):",
          round(by_agent["seller_premium"].corr(by_agent["mean_ratio"]), 2))
    print("Sold price range:", int(sales["sold_price"].min()), "to", int(sales["sold_price"].max()))
    if "prior_sale_date" in sales:
        ratio = (sales["assessed_value"] / d["true_value"]).median()
        owned = (pd.to_datetime(sales["list_date"]) - pd.to_datetime(sales["prior_sale_date"])).dt.days / 365.25
        print(f"Prop 13: typical assessed value is {ratio:.0%} of true value | median years owned "
              f"{owned.median():.1f} | owned 20+ years: {(owned >= 20).mean():.0%} of sales")


def generate(seed=SEED, selection_strength=SELECTION_STRENGTH, buyer_spread=0.010,
             assessed_style=ASSESSED_STYLE, quiet=False):
    """Build every table in memory. The defaults give the standard fake data.
    Stress tests call this with other settings (for example selection_strength=-0.4).
    assessed_style="market" gives the Phase 4 world (assessed value near market value)."""
    assert assessed_style in {"prop13", "market"}
    rng = np.random.default_rng(seed)  # the random number generator
    agents, skills, activity = make_agents(rng, buyer_spread)
    homes = make_homes(rng, seed)
    skills["win_skill"] = np.random.default_rng([seed, 2]).normal(0.0, 0.5, len(skills)).round(3)
    sales = make_sales(rng, homes, agents, skills, activity, selection_strength)
    offers = make_offers(sales, agents, skills, activity, seed)
    estate = pd.Series(False, index=agents["agent_id"].to_numpy())
    if assessed_style == "prop13":
        sales, estate = add_sale_history(sales, homes, agents, seed)
    if not quiet:
        run_checks(sales, homes, skills)
    return {"agents": agents, "sales": sales, "offers": offers, "skills": skills, "homes": homes,
            "estate": estate}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = generate()

    # Public files: what a brokerage would really have.
    t["agents"].to_csv(OUT_DIR / "agents.csv", index=False)
    t["sales"].to_csv(OUT_DIR / "sales.csv", index=False)
    t["offers"].to_csv(OUT_DIR / "offers.csv", index=False)
    # Answer keys: only exist in fake data. Used to grade the model later.
    t["skills"].to_csv(OUT_DIR / "true_skills.csv", index=False)
    t["homes"][["listing_id", "true_value", "quality"]].to_csv(OUT_DIR / "true_home_values.csv", index=False)
    print("\nSaved 5 files in", OUT_DIR)


if __name__ == "__main__":
    main()
