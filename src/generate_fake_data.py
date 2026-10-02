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
STREETS = ["Oak", "Maple", "Cedar", "Pine", "Elm", "Birch", "Willow", "Aspen"]


def make_agents(rng):
    """Build the agents table and the hidden-skills table."""
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
    buyer_savings = np.clip(rng.normal(0.005, 0.010, N_AGENTS), -0.02, 0.03)
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


def make_homes(rng):
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
    assess_rng = np.random.default_rng([SEED, 1])
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


def make_sales(rng, homes, agents, skills, activity):
    """Turn homes into sales: pick agents, add dates, prices, list-price games."""
    n = len(homes)
    home_z = ((np.log(homes["true_value"]) - np.log(homes["true_value"]).mean())
              / np.log(homes["true_value"]).std()).to_numpy()

    # Who sold it, and who bought it.
    li = pick_agents(rng, activity, skills["seller_premium"].to_numpy(), home_z, SELECTION_STRENGTH)
    bi = pick_agents(rng, activity, skills["buyer_savings"].to_numpy(), home_z, SELECTION_STRENGTH)
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

    # Market effect: a spring bump and a slow upward trend. Not the agent's doing.
    month = sold_date.month.to_numpy()
    years_in = (sold_date - pd.Timestamp(START_DATE)).days.to_numpy() / 365
    market = np.exp(0.03 * np.cos(2 * np.pi * (month - 5) / 12) + 0.04 * years_in)

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


def main():
    rng = np.random.default_rng(SEED)  # the random number generator
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agents, skills, activity = make_agents(rng)
    homes = make_homes(rng)
    sales = make_sales(rng, homes, agents, skills, activity)
    run_checks(sales, homes, skills)

    # Public files: what a brokerage would really have.
    agents.to_csv(OUT_DIR / "agents.csv", index=False)
    sales.to_csv(OUT_DIR / "sales.csv", index=False)
    # Answer keys: only exist in fake data. Used to grade the model later.
    skills.to_csv(OUT_DIR / "true_skills.csv", index=False)
    homes[["listing_id", "true_value", "quality"]].to_csv(OUT_DIR / "true_home_values.csv", index=False)
    print("\nSaved 4 files in", OUT_DIR)


if __name__ == "__main__":
    main()
