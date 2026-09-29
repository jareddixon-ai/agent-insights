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


def main():
    rng = np.random.default_rng(SEED)  # the random number generator
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agents, skills = make_agents(rng)
    agents.to_csv(OUT_DIR / "agents.csv", index=False)
    skills.to_csv(OUT_DIR / "true_skills.csv", index=False)

    print("Agents:", len(agents))
    print(agents["office"].value_counts().sort_index().to_string())
    print("On a team:", (agents["team"] != "").sum(), "| Solo:", (agents["team"] == "").sum())
    print("Years licensed: min", agents["years_licensed"].min(), "max", agents["years_licensed"].max())
    print(skills[["seller_premium", "buyer_savings"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
