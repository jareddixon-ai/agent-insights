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
    return agents, skills

N_SALES = 5000
ZIPS = ["90101", "90102", "90103", "90104", "90105", "90106", "90107", "90108", "90109", "90110"]


def make_homes(rng):
    """Build 5,000 fake homes and their true market value (what each home is worth
    before any agent skill). The true value is the answer key for later steps."""
    n = N_SALES

    zip_code = rng.choice(ZIPS, size=n)
    # Each zip has its own price level (a multiplier on value, in log terms).
    zip_effect = dict(zip(ZIPS, rng.normal(0.0, 0.25, len(ZIPS))))

    # Size: most homes near 1,800 sqft, a few very large ones.
    sqft = np.clip(rng.lognormal(np.log(1800), 0.35, n), 600, 6000).round(-1).astype(int)
    # Bedrooms and bathrooms grow with size (plus randomness).
    beds = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 7).astype(int)
    baths = np.clip(np.round((beds * 0.7 + rng.normal(0.3, 0.6, n)) * 2) / 2, 1, 6)
    lot_sqft = np.clip(rng.lognormal(np.log(6500), 0.5, n), 1500, 40000).round(-2).astype(int)
    year_built = rng.integers(1925, 2024, n)
    age = 2024 - year_built

    # "Quality" = things we can't see in the data (condition, view, finishes).
    # It changes the price but no column records it.
    quality = rng.normal(0.0, 0.08, n)

    # Log price = sum of effects. (Log means each effect is a % change, which
    # matches how we describe agent skill, in %.)
    log_value = (
        13.45
        + np.array([zip_effect[z] for z in zip_code])
        + 0.75 * np.log(sqft / 1800)
        + 0.03 * (beds - 3)
        + 0.06 * (baths - 2)
        + 0.10 * np.log(lot_sqft / 6500)
        - 0.002 * age
        + quality
    )

    homes = pd.DataFrame(
        {
            "listing_id": [f"L{i:05d}" for i in range(1, n + 1)],
            "zip": zip_code,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "lot_sqft": lot_sqft,
            "year_built": year_built,
            "true_value": np.exp(log_value).round(-3),
            "quality": quality.round(4),
        }
    )
    return homes


def main():
    rng = np.random.default_rng(SEED)  # the random number generator
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agents, skills = make_agents(rng)
    agents.to_csv(OUT_DIR / "agents.csv", index=False)
    skills.to_csv(OUT_DIR / "true_skills.csv", index=False)

    homes = make_homes(rng)
    print("Homes:", len(homes))
    print(homes[["sqft", "beds", "baths", "lot_sqft", "true_value"]].describe().round(0).to_string())
    print(homes.groupby("zip")["true_value"].median().round(-3).to_string())
    print()

    print("Agents:", len(agents))
    print(agents["office"].value_counts().sort_index().to_string())
    print("On a team:", (agents["team"] != "").sum(), "| Solo:", (agents["team"] == "").sum())
    print("Years licensed: min", agents["years_licensed"].min(), "max", agents["years_licensed"].max())
    print(skills[["seller_premium", "buyer_savings"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
