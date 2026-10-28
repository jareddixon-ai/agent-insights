"""Phase 4 tests: validation numbers and charts, and the Streamlit app reading rankings.csv."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import validate

APP = Path(__file__).resolve().parent.parent / "app" / "coaching_app.py"


@pytest.fixture(scope="module")
def validation(scored_dir, tmp_path_factory):
    docs = tmp_path_factory.mktemp("docs")
    stats, deals, worlds, split = validate.main(data=scored_dir, docs=docs, seeds=(42,))
    return docs, stats, deals, worlds, split


# ---- 4-0 to 4-3 Validation ----
def test_charts_are_written(validation):
    docs = validation[0]
    for name in ["validation.png", "accuracy_by_deals.png", "stress_test.png", "hidden_quality.png"]:
        assert (docs / name).stat().st_size > 10_000, name


def test_model_finds_skill_and_buyer_side_is_weaker(validation):
    stats = validation[1]
    assert stats["seller"]["rank_agreement_shrunk"] > 0.6
    # Phase 4.5: with California tax values the buyer side is far weaker (about 0.05 to 0.3).
    assert stats["buyer"]["rank_agreement_shrunk"] < stats["seller"]["rank_agreement_shrunk"] - 0.3
    assert stats["seller"]["coverage_used"] > 0.9
    assert stats["seller"]["clear_calls_right"] >= 0.9


def test_shrinkage_helps_thin_records(validation):
    for g in validation[2].values():
        thin = g.loc[["1-4", "5-9"]]
        assert (thin["error_pts_shrunk"] < thin["error_pts_raw"]).all()
        assert g.loc["1-4", "share_called_clearly"] == 0   # never call an agent with 1-4 deals


def test_hidden_quality_explains_the_misses(validation):
    stats = validation[1]
    assert stats["seller"]["miss_vs_hidden_corr"] > 0.5     # hidden value inflates seller scores
    assert stats["buyer"]["miss_vs_hidden_corr"] < -0.5     # and makes buyer agents look like they overpaid


def test_no_answer_key_check_runs(validation):
    split = validation[4]
    assert split["reliability"].between(-1, 1).all()
    assert split["implied_accuracy"].between(0, 1).all()


# ---- 4-4 Streamlit app ----
def test_app_shows_an_agent_from_rankings(scored_dir, monkeypatch):
    monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(scored_dir))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Agent coaching view"
    assert at.metric[0].label == "Coaching index"
    first = at.sidebar.selectbox[0].value
    at.sidebar.selectbox[0].select(at.sidebar.selectbox[0].options[-1]).run()   # pick another agent
    assert not at.exception and at.sidebar.selectbox[0].value != first
    rankings = pd.read_csv(scored_dir / "rankings.csv", index_col="agent_id")
    shown = at.sidebar.selectbox[0].value
    assert at.metric[0].value == f"{rankings.loc[shown, 'coaching_index']:+.2f}"


def test_app_explains_missing_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "rankings.csv not found" in at.error[0].value
