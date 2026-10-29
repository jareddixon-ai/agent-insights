"""Phase 8 tests: public demo snapshot, which folder the dashboard reads, the CI workflow and badge,
and the pitch-kit number sources (every number the kit quotes must still appear in the README or
the data dictionary)."""

import re
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import build_demo

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "coaching_app.py"
DEMO = ROOT / "demo"


def test_demo_snapshot_matches_readme_calls():
    """The committed demo must show the same calls the README and pitch kit quote."""
    t = pd.read_csv(DEMO / "rankings.csv", index_col="agent_id")
    counts = t["index_group"].value_counts().to_dict()
    assert len(t) == 150
    assert counts == {"Not clearly different from average": 135, "Clearly below average": 8,
                      "Clearly above average": 6, "No data": 1}
    a133 = t.loc["A133"]   # the agent the demo script walks through
    assert (round(a133["seller_low"], 3), round(a133["seller_high"], 3), a133["seller_n"]) == (0.047, 0.107, 62)
    assert a133["index_group"] == "Clearly above average"
    assert set(t["buyer_group"]) <= {"Context only", "No data"}
    assert set(t["offer_win_group"]) <= {"Context only", "No data"}


def test_demo_snapshot_is_the_seed_42_pipeline(scored_dir):
    """Point scores don't depend on the bootstrap, so they must equal a fresh build of seed 42.
    That proves the snapshot is the fake pipeline's output, not edited numbers."""
    demo = pd.read_csv(DEMO / "rankings.csv", index_col="agent_id").sort_index()
    fresh = pd.read_csv(scored_dir / "rankings.csv", index_col="agent_id").sort_index().round(4)
    for col in ["seller_score", "seller_n", "buyer_score", "buyer_n"]:
        pd.testing.assert_series_equal(demo[col], fresh[col], check_names=False, atol=1e-4)
    trends = pd.read_csv(DEMO / "trends.csv")
    assert {"agent_id", "low", "high", "small_sample"} <= set(trends.columns)


def test_demo_holds_no_answer_key_or_personal_columns():
    """The public snapshot only has what the dashboard reads: no true skills, no addresses, no names."""
    for name in build_demo.SNAPSHOT_FILES:
        cols = pd.read_csv(DEMO / name, nrows=1).columns.str.lower()
        assert not any(w in c for c in cols for w in ["true", "address", "name", "email", "phone", "note"]), name
    assert sorted(p.name for p in DEMO.iterdir()) == sorted(build_demo.SNAPSHOT_FILES)


def test_choose_data(tmp_path):
    # 1. A private folder always wins, even if it is empty: never fall back to fake numbers.
    folder, is_demo = build_demo.choose_data({"AGENT_INSIGHTS_DATA": str(tmp_path / "private")}, root=tmp_path)
    assert folder == tmp_path / "private" and not is_demo
    # 2. No scores in data/fake: the demo snapshot.
    folder, is_demo = build_demo.choose_data({}, root=tmp_path)
    assert folder == tmp_path / "demo" and is_demo
    # 3. Scores in data/fake: use them.
    (tmp_path / "data" / "fake").mkdir(parents=True)
    (tmp_path / "data" / "fake" / "rankings.csv").write_text("agent_id\n")
    folder, is_demo = build_demo.choose_data({}, root=tmp_path)
    assert folder == tmp_path / "data" / "fake" and not is_demo


def test_app_opens_on_demo_snapshot(monkeypatch):
    """With no data/fake/ scores (as on the hosted demo), the app reads demo/ and says it is a snapshot."""
    monkeypatch.delenv("AGENT_INSIGHTS_DATA", raising=False)
    monkeypatch.setattr(build_demo, "choose_data", lambda env=None, root=None: (DEMO, True))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Public demo" in i.value for i in at.info)
    assert not any("rankings.csv not found" in e.value for e in at.error)


def test_ci_workflow_and_badge():
    wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    for needed in ["python-version: \"3.12\"", "pip install -r requirements.txt", "pytest", "contents: read"]:
        assert needed in wf, needed
    assert "secrets." not in wf                      # the tests need no secrets
    readme = (ROOT / "README.md").read_text()
    assert "actions/workflows/tests.yml/badge.svg" in readme


SOURCES = (ROOT / "README.md").read_text() + (ROOT / "docs" / "data_dictionary.md").read_text()


def pitch_rows():
    text = (ROOT / "docs" / "pitch_numbers.md").read_text()
    rows = [line for line in text.splitlines() if line.startswith("| ") and "`" in line]
    return [(row.split("|")[1].strip(), re.findall(r"`([^`]+)`", row.split("|")[3])) for row in rows]


def test_pitch_numbers_have_sources():
    rows = pitch_rows()
    assert len(rows) >= 15
    for number, phrases in rows:
        assert phrases, number
        for phrase in phrases:
            assert phrase in SOURCES, f"{number}: '{phrase}' is no longer in the README or data dictionary"
