"""Shared test setup: make src/ importable, build the fake data once, and score it once."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Run the generator in a temp folder so tests never touch your real data/ folder."""
    work = tmp_path_factory.mktemp("work")
    subprocess.run([sys.executable, str(ROOT / "src" / "generate_fake_data.py")],
                   cwd=work, check=True, capture_output=True)
    return work / "data" / "fake"


@pytest.fixture(scope="session")
def scored_dir(tmp_path_factory, data_dir):
    """A copy of the fake data with the scoring files written (a quick 20-resample bootstrap,
    one fake world per stress-test scenario), so the validation and app tests have real inputs."""
    import shutil

    import bootstrap_ranges as br
    import offer_metrics
    import price_model
    import score_agents
    import speed_and_pricing
    import stress_test

    out = tmp_path_factory.mktemp("scored")
    for f in data_dir.iterdir():
        shutil.copy(f, out / f.name)
    sales = pd.read_csv(out / "sales.csv", keep_default_na=False)
    full_s, full_b = br.shrunk_scores(sales)
    ds, db = br.bootstrap(sales, n_resamples=20)
    boot = pd.concat({"seller": br.summarize(full_s, ds), "buyer": br.summarize(full_b, db)},
                     names=["side", "agent_id"])
    boot.to_csv(out / "bootstrap_scores.csv")
    pred = price_model.cross_validated_predictions(sales)
    offer = offer_metrics.offer_table(sales, pd.read_csv(out / "offers.csv", keep_default_na=False), pred,
                                      penalty=0.25)   # fixed penalty: skips the cross-validation, faster
    offer.to_csv(out / "offer_scores.csv", index_label="agent_id")
    speed = speed_and_pricing.speed_and_pricing(sales, pred)
    agents = pd.read_csv(out / "agents.csv", keep_default_na=False)
    table = score_agents.build_table(boot, offer, speed, agents, score_agents.load_config())
    table.to_csv(out / "rankings.csv", index_label="agent_id")
    stress = pd.DataFrame({label: stress_test.run_one(42, strength, oracle)
                           for label, strength, oracle in stress_test.SCENARIOS}).T
    stress.to_csv(out / "stress_test.csv", index_label="scenario")
    return out
