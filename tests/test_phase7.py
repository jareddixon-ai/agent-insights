"""Phase 7 tests: private-copy mode (column mapping, readiness check, scoring with no answer key),
privacy guarantees, and the dashboard guard rails (explore toggle, 10-peer minimum)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import bootstrap_ranges as br
import make_fake_export
import run_private
from readiness_check import ROOT, UnsafeFolder, check, check_folder_safe, load_mapping

APP = ROOT / "app" / "coaching_app.py"
DEMO = ROOT / "config" / "column_mapping_demo.json"
# Every kind of made-up personal detail in the fake export. None may ever be printed or stored.
PERSONAL = ["Example Lane", "Pat Example", "@example.com", "805-555-01", "Fake note", "Agent Example",
            "agents.example.com"]
SCHEMA = set(run_private.CONTEXT) | {"listing_id", "zip", "beds", "baths", "sqft", "lot_sqft", "year_built",
                                     "assessed_value", "list_date", "sold_date", "sold_price", "listing_agent_id",
                                     "buyer_agent_id", "prior_sale_date", "prior_sale_price", "value_estimate",
                                     "condition_rating", "agent_id", "office", "team", "years_licensed",
                                     "offer_price", "outcome", "financing", "down_payment_pct",
                                     "inspection_contingency", "appraisal_contingency", "loan_contingency",
                                     "days_to_close"}


@pytest.fixture(scope="module")
def identity():
    return load_mapping()


@pytest.fixture(scope="module")
def export_dir(data_dir, tmp_path_factory):
    return make_fake_export.main(data=data_dir, out=tmp_path_factory.mktemp("export"))


@pytest.fixture(scope="module")
def private_run(export_dir):
    """Run the whole private flow once on the messy export (10 resamples: fast), capturing what it prints."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_private.main(["--mapping", str(DEMO), "--data", str(export_dir), "--resamples", "10"])
    return export_dir, buf.getvalue()


def copy_fake(data_dir, tmp_path, name="copy", **edits):
    """A copy of the fake data folder with some sales columns changed."""
    out = tmp_path / name
    out.mkdir()
    for name in ["sales.csv", "agents.csv", "offers.csv"]:
        shutil.copy(data_dir / name, out / name)
    sales = pd.read_csv(out / "sales.csv", keep_default_na=False)
    for col, values in edits.items():
        sales[col] = values
    sales.to_csv(out / "sales.csv", index=False)
    return out


# ---- column mapping ----
def test_identity_mapping_reproduces_the_pipeline_exactly(identity, data_dir):
    """Going through the mapping and cleaning changes nothing on clean data: same scores."""
    r = check(identity, str(data_dir))
    assert r.ready and set(r.tables) == {"sales", "agents", "offers"}
    via_mapping, _ = br.shrunk_scores(r.tables["sales"])
    direct, _ = br.shrunk_scores(pd.read_csv(data_dir / "sales.csv", keep_default_na=False))
    np.testing.assert_allclose(via_mapping["score"], direct["score"].reindex(via_mapping.index), atol=1e-10)


def test_demo_mapping_matches_the_export_headers(export_dir):
    mapping = load_mapping(DEMO)
    header = set(pd.read_csv(export_dir / mapping["sales"]["file"], nrows=0).columns)
    assert {v for v in mapping["sales"]["columns"].values() if v} <= header
    assert all(k in mapping["sales"]["columns"] for k in load_mapping()["sales"]["columns"])   # same keys as ours


# ---- readiness check ----
def test_readiness_on_messy_export_says_what_is_missing_and_what_it_turns_off(export_dir):
    r = check(load_mapping(DEMO), str(export_dir))
    text = r.text()
    assert r.ready
    assert r.turned_off == {"buyer_scores", "offer_score"}
    assert "50 duplicates" in text and "200 unsold" in text                    # cleaned, and said so
    assert "5 whose header looks like personal details" in text                # counted, never read
    assert "No last sale date and price" in text
    assert "No independent value estimate" in text and "buyer scores stay context only" in text
    assert "No offer data: no offer score" in text
    assert "'years_licensed'" in text
    assert "not clearly different from average" in text
    assert len(r.tables["sales"]) == 4800 and set(r.tables["sales"].columns) <= SCHEMA


def test_missing_required_column_blocks_scoring(identity, data_dir):
    mapping = json.loads(json.dumps(identity))
    mapping["sales"]["columns"]["zip"] = None
    r = check(mapping, str(data_dir))
    assert not r.ready and any("zip" in b for b in r.blockers)


def test_offers_need_every_term(identity, data_dir):
    mapping = json.loads(json.dumps(identity))
    mapping["offers"]["columns"]["financing"] = None
    r = check(mapping, str(data_dir))
    assert r.ready and "offer_score" in r.turned_off and "offers" not in r.tables


def test_agent_ids_that_look_like_names_block_and_are_not_printed(identity, data_dir, tmp_path):
    sales = pd.read_csv(data_dir / "sales.csv", keep_default_na=False)
    names = "Jordan Sample" + sales["listing_agent_id"].str[1:]
    folder = copy_fake(data_dir, tmp_path, listing_agent_id=names)
    r = check(identity, str(folder))
    assert not r.ready and any("may be names or emails" in b for b in r.blockers)
    assert "Jordan Sample" not in r.text()


def test_value_estimate_leak_warning(identity, data_dir, tmp_path):
    rng = np.random.default_rng(0)
    sales = pd.read_csv(data_dir / "sales.csv", keep_default_na=False)
    true_value = pd.read_csv(data_dir / "true_home_values.csv").set_index("listing_id")["true_value"]
    mapping = json.loads(json.dumps(identity))
    mapping["sales"]["columns"]["value_estimate"] = "value_estimate"
    leaky = copy_fake(data_dir, tmp_path, "leaky", value_estimate=sales["sold_price"] * (1 + rng.normal(0, 0.005, len(sales))))
    r = check(mapping, str(leaky))
    assert any("suspiciously close" in w for w in r.watch)                   # like a loan appraisal
    assert "buyer_scores" not in r.turned_off and any("context only" in w for w in r.watch)
    honest = copy_fake(data_dir, tmp_path, "honest", value_estimate=true_value.reindex(sales["listing_id"]).to_numpy()
                       * (1 + rng.normal(0, 0.04, len(sales))))
    r = check(mapping, str(honest))
    assert not any("suspiciously close" in w for w in r.watch)


# ---- folder safety ----
def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_refuses_a_data_folder_that_git_would_publish(tmp_path, identity):
    repo = tmp_path / "repo"
    (repo / "private").mkdir(parents=True)
    (repo / "public").mkdir()
    git(repo, "init", "-q")
    (repo / ".gitignore").write_text("private/\n")
    assert "git-ignored" in check_folder_safe(repo / "private", repo)
    with pytest.raises(UnsafeFolder):
        check_folder_safe(repo / "public", repo)
    with pytest.raises(UnsafeFolder):                                          # refused before reading anything
        check(identity, str(repo / "public"), repo=repo)
    (repo / "private" / "sales.csv").write_text("x\n")                        # a file someone force-added
    git(repo, "add", "-f", "private/sales.csv")
    with pytest.raises(UnsafeFolder):
        check_folder_safe(repo / "private", repo)
    plain = tmp_path / "not_a_repo"                                           # can't confirm it's ignored
    (plain / "data").mkdir(parents=True)
    with pytest.raises(UnsafeFolder):
        check_folder_safe(plain / "data", plain)
    assert check_folder_safe(tmp_path / "elsewhere", repo) == "outside the repo folder"


def test_this_repo_ignores_data_private_and_not_docs():
    assert "git-ignored" in check_folder_safe(ROOT / "data" / "private")
    with pytest.raises(UnsafeFolder):
        check_folder_safe(ROOT / "docs")


# ---- scoring a private copy ----
def test_private_run_never_prints_or_stores_personal_details(private_run):
    folder, printed = private_run
    assert "Coaching index:" in printed
    for marker in PERSONAL:
        assert marker not in printed
    assert not re.search(r"\bA\d{3}\b|\bL\d{5}\b", printed)                  # no agent or listing IDs either
    for sub in ["standard", "scored"]:
        for f in (folder / sub).glob("*.csv"):
            text = f.read_text()
            for marker in PERSONAL:
                assert marker not in text, f"{f.name} stores a personal detail"
    for f in (folder / "standard").glob("*.csv"):
        assert set(pd.read_csv(f, nrows=0).columns) <= SCHEMA, f"{f.name} stores a column we don't use"


def test_private_run_scores_like_the_pipeline(private_run, data_dir):
    folder, _ = private_run
    r = pd.read_csv(folder / "scored" / "rankings.csv", index_col="agent_id")
    assert len(r) == 150 and r["offer_win_score"].isna().all()               # no offers file: no offer score
    assert set(r["buyer_group"].dropna()) <= {"Context only", "No data"}
    truth = pd.read_csv(data_dir / "true_skills.csv").set_index("agent_id")["seller_premium"]
    assert r["seller_score"].rank().corr(truth.reindex(r.index).rank()) > 0.5
    t = pd.read_csv(folder / "scored" / "trends.csv")
    assert set(t["kind"]) == {"year", "half"}


def test_private_scores_open_in_the_dashboard(private_run, monkeypatch):
    folder, _ = private_run
    monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(folder / "scored"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 150
    assert any(m.label == "Offer score" and m.value == "n/a" for m in at.tabs[1].metric)


# ---- dashboard guard rails ----
@pytest.fixture
def app(scored_dir, monkeypatch):
    monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(scored_dir))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    return at


def test_tested_weights_are_the_default_view(app):
    assert not [s for s in app.slider if str(s.key).startswith("w_")]        # no weight sliders until "Explore"
    assert app.toggle(key="explore").value is False
    assert any("tested settings" in c.value for c in app.tabs[0].caption)
    tested = app.dataframe[0].value["Coaching index"].tolist()

    app.toggle(key="explore").set_value(True).run()
    assert app.slider(key="w_buyer").value == 0.0
    assert any("Exploring" in c.value for c in app.tabs[0].caption)
    app.slider(key="w_buyer").set_value(0.5).run()
    assert app.dataframe[0].value["Coaching index"].tolist() != tested and len(app.warning) == 1

    app.toggle(key="explore").set_value(False).run()                         # off again: back to tested weights
    assert app.dataframe[0].value["Coaching index"].tolist() == tested and len(app.warning) == 0


def small_brokerage(scored_dir, tmp_path, n_agents):
    folder = tmp_path / f"small_{n_agents}"
    folder.mkdir()
    r = pd.read_csv(scored_dir / "rankings.csv", index_col="agent_id")
    keep = r[r["seller_score"].notna()].index[:n_agents]
    r.loc[keep].to_csv(folder / "rankings.csv")
    t = pd.read_csv(scored_dir / "trends.csv")
    t[t["agent_id"].isin(keep)].to_csv(folder / "trends.csv", index=False)
    return folder


def test_peer_chart_needs_10_peers(scored_dir, tmp_path, monkeypatch):
    for n_agents, hidden in [(10, True), (11, False)]:                       # 9 peers: hidden, 10 peers: shown
        monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(small_brokerage(scored_dir, tmp_path, n_agents)))
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        assert not at.exception
        caps = " ".join(c.value for c in at.tabs[1].caption)
        assert ("Peer comparison hidden: fewer than 10" in caps) is hidden
        assert ("Grey ticks" in caps) is not hidden


def test_min_peers_setting_can_not_go_below_10():
    cfg = json.loads((ROOT / "config" / "scoring.json").read_text())
    assert cfg["min_peers"] >= 10
    assert "max(int(cfg.get(\"min_peers\", 10)), 10)" in APP.read_text()


# ---- docs ----
def test_readme_has_ethics_and_private_copy_sections():
    readme = (ROOT / "README.md").read_text()
    for heading in ["## Ethics", "## Private copy"]:
        assert heading in readme
    for phrase in ["not for firing", "Ranges measure luck, not bias", "see their own card",
                   "readiness_check.py", "run_private.py", "column_mapping.json"]:
        assert phrase in readme, phrase
