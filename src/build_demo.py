"""Phase 8: build everything the dashboard needs from fake data, and save the public demo snapshot.

Generated data is never committed (data/fake/ is git-ignored). The hosted demo (Streamlit Community
Cloud) still needs something to show the moment someone clicks the link, so we commit two small
FAKE files in demo/: rankings.csv and trends.csv. The app reads them only when data/fake/ has no
scores and no private folder is set.

Same steps and seed as the README "Run it" section, in order:
generate_fake_data, price_model, speed_and_pricing, bootstrap_ranges (1,000 resamples), offer_metrics,
score_agents, trends. The bootstrap is seeded, so the snapshot repeats exactly.

Why not a faster rebuild on the server? With 200 resamples instead of 1,000, one agent's call moves
across the line (7 "clearly above" instead of 6). Resampling noise at the edge would make the demo
disagree with the README. The snapshot keeps them identical (checked in tests/test_phase8.py).

Run:  python src/build_demo.py            (builds data/fake/ with 1,000 resamples and writes demo/, about 3 minutes)
      python src/build_demo.py 200        (quick build of data/fake/ only; does not touch demo/)
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEMO = ROOT / "demo"
SNAPSHOT_FILES = ["rankings.csv", "trends.csv"]
FULL_RESAMPLES = 1000


def choose_data(env=None, root=ROOT):
    """Which folder the dashboard reads, and whether it is the public demo snapshot.
    A private folder (AGENT_INSIGHTS_DATA) always wins and never falls back to the demo, so a missing
    private file can't quietly show fake numbers to a brokerage."""
    env = os.environ if env is None else env
    if env.get("AGENT_INSIGHTS_DATA"):
        return Path(env["AGENT_INSIGHTS_DATA"]), False
    fake = Path(root) / "data" / "fake"
    if (fake / "rankings.csv").exists():
        return fake, False
    return Path(root) / "demo", True


def build(n_resamples=FULL_RESAMPLES, quiet=True):
    """Write data/fake/ and every scored file the dashboard reads. Returns the data folder."""
    import agent_model
    subprocess.run([sys.executable, str(ROOT / "src" / "generate_fake_data.py")], cwd=ROOT, check=True,
                   capture_output=quiet)

    import bootstrap_ranges
    import offer_metrics
    import price_model
    import score_agents
    import speed_and_pricing
    import trends

    steps = [price_model.main, speed_and_pricing.main, lambda: bootstrap_ranges.main(n_resamples),
             offer_metrics.main, score_agents.main, trends.main]
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        for step in steps:
            step()
    return agent_model.DATA


def snapshot(data_folder):
    """Copy the two files the dashboard reads into demo/. Fake data only."""
    DEMO.mkdir(exist_ok=True)
    for name in SNAPSHOT_FILES:
        shutil.copy(Path(data_folder) / name, DEMO / name)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else FULL_RESAMPLES
    print(f"Building fake data and scores with {n} bootstrap resamples...")
    folder = build(n, quiet=False)
    if n == FULL_RESAMPLES:
        snapshot(folder)
        print(f"Saved the demo snapshot to {DEMO}")
    else:
        print("Quick build: demo/ left unchanged (the snapshot always uses 1,000 resamples).")
