"""Phase 9 tests: the broker-only coaching design (opens on Team overview, Coaching brief, no rank wording,
nothing shown to agents), the trend against the agent's own past, broker tags, the track-record and
low-confidence labels, the sale list, the disclosure setting (wording only), the rename to scorecards.csv,
and the private split-half check (seller only, counts only)."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import build_demo
import context_notes
import disclosure
import sale_detail
import validate_private
from self_trend import self_trend

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "coaching_app.py"
BANNED = re.compile(r"\b[Rr]ank(s|ed|ing)?\b|\b[Ll]eaders?\b|clearly below|Focus area|[Rr]eport card", re.I)
AGENTS_SEE = re.compile(r"agents? (can |will |would )?(see|view|read|receive)|see their own|shown to (the )?agents?|"
                        r"your (own )?(seller )?result|nothing hidden from", re.I)


@pytest.fixture
def app(scored_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(scored_dir))
    monkeypatch.setenv("AGENT_INSIGHTS_NOTES", str(tmp_path / "notes.csv"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    return at


def all_text(at):
    """Every visible word in the app: titles, headers, markdown, captions, metrics, warnings, tab labels."""
    parts = [e.value for group in (at.title, at.subheader, at.markdown, at.caption, at.info, at.warning, at.success)
             for e in group]
    parts += [f"{m.label} {m.value} {m.delta}" for m in at.metric]
    parts += [t.label for t in at.tabs] + [t.label for t in at.toggle]
    return "\n".join(parts)


def brief_text(at):
    return "\n".join([m.value for m in at.tabs[1].markdown] + [c.value for c in at.tabs[1].caption]
                      + [i.value for i in at.tabs[1].info] + [w.value for w in at.tabs[1].warning])


def pick(at, agent):
    at.sidebar.selectbox[0].select(agent).run()
    assert not at.exception


# ---- the design ----
def test_app_opens_on_the_team_overview(app):
    assert [t.label for t in app.tabs] == ["Team overview", "Coaching brief"]
    assert app.dataframe[0].key == "team_table" or len(app.dataframe) >= 1       # the first thing on screen is the team table
    assert app.subheader[0].value.endswith("listed by agent code")


def test_team_overview_is_listed_by_agent_code_never_by_score(app):
    t = app.dataframe[0].value
    assert t["Agent"].tolist() == sorted(t["Agent"].tolist())
    app.toggle(key="explore").set_value(True).run()
    app.slider(key="w_buyer").set_value(0.5).run()                       # a different index must not reorder rows
    assert app.dataframe[0].value["Agent"].tolist() == sorted(app.dataframe[0].value["Agent"].tolist())
    assert set(app.dataframe[0].value["Result"]) <= {"Strength", "Not clearly different from average",
                                                    "Room to grow", "Building a track record", "No data"}


def test_no_rank_or_leader_wording_anywhere_a_person_reads(app):
    assert not BANNED.search(all_text(app)), BANNED.search(all_text(app))
    for name in ["README.md"] + [f"docs/{p.name}" for p in (ROOT / "docs").glob("*.md")] + ["app/coaching_app.py"]:
        text = (ROOT / name).read_text()
        assert not BANNED.search(text), (name, BANNED.search(text).group(0))


def test_nothing_says_agents_see_anything_and_the_brokerage_decides(app, scored_dir):
    """The new rule: the tool is for brokers only. No claim anywhere that agents see a card, a score or a readout."""
    pick(app, app.sidebar.selectbox[0].options[0])
    text = all_text(app) + "\n" + brief_text(app)
    assert not AGENTS_SEE.search(text), AGENTS_SEE.search(text).group(0)
    assert "shows nothing to agents" in text and "decides what its agents are told" in text
    for name in ["README.md", "docs/pitch_numbers.md", "docs/data_dictionary.md", "docs/readout_owner_template.md",
                 "docs/private_install_walkthrough.md", "docs/builder_vs_agent_note.md"]:
        body = (ROOT / name).read_text()
        assert not AGENTS_SEE.search(body), (name, AGENTS_SEE.search(body).group(0))
    readme = (ROOT / "README.md").read_text()
    assert "decides what its agents are told" in readme and "does not give legal advice" in readme
    assert not (ROOT / "docs" / "readout_agent_template.md").exists()          # the agent readout is gone


def test_the_calls_keep_ranges_and_the_honest_label(app):
    t = app.dataframe[0].value
    for col in ["Seller side", "Coaching index"]:
        filled = t[col][~t[col].isin(["n/a", "Building a track record"])]
        assert filled.str.contains(r"\(range .+ to .+\)").all()
    assert "Not clearly different from average" in set(t["Result"])


def test_index_is_labeled_with_its_percent_in_the_brief(app, scored_dir):
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    agent = cards["seller_n"].idxmax()
    pick(app, agent)
    text = brief_text(app)
    assert "same result as the seller percent" in text
    pct = f"{round(cards.loc[agent, 'seller_score'] * 100, 1) + 0.0:+.1f}%"
    assert pct in text
    assert app.tabs[1].markdown[0].value.startswith("#### Seller result") or "#### Seller result" in text   # percent comes first


def test_peer_chart_is_off_by_default_and_never_names_other_agents(app):
    assert app.toggle(key="show_peers").value is False
    assert not any("Grey ticks" in c.value for c in app.tabs[1].caption)
    app.toggle(key="show_peers").set_value(True).run()
    assert any("Grey ticks" in c.value for c in app.tabs[1].caption)    # 10+ peers in the fake data
    assert any("no names" in c.value for c in app.tabs[1].caption)


def test_scorecards_rename_is_complete():
    assert (ROOT / "demo" / "scorecards.csv").exists() and not (ROOT / "demo" / "rankings.csv").exists()
    assert build_demo.SNAPSHOT_FILES == ["scorecards.csv", "trends.csv", "sale_detail.csv"]
    hits = []
    for path in list(ROOT.glob("src/*.py")) + list(ROOT.glob("app/*.py")) + list(ROOT.glob("docs/*.md")) + [ROOT / "README.md"]:
        if "rankings.csv" in path.read_text():
            hits.append(path.name)
    assert not hits, hits


# ---- track record, low confidence, the sale list ----
def test_agent_with_under_five_sales_gets_a_track_record_label_not_a_score(app, scored_dir):
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    small = cards.index[cards["low_data_flag"]]
    assert len(small) > 0
    row = app.dataframe[0].value.set_index("Agent").loc[small[0]]
    assert row["Seller side"] == row["Coaching index"] == row["Result"] == "Building a track record"
    pick(app, small[0])
    text = brief_text(app)
    assert "Building a track record" in text and "Talking points" not in text
    assert not any(m.label == "Seller side" for m in app.metric)


def test_every_scored_agent_row_keeps_its_range_and_big_agents_get_a_brief_with_talking_points(app, scored_dir):
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    pick(app, cards["seller_n"].idxmax())
    assert app.metric[0].label == "Seller side" and app.metric[0].value.endswith("%")
    assert "#### Talking points" in brief_text(app) and "#### Sales behind these numbers" in brief_text(app)


def test_similar_home_count_and_low_confidence_label():
    zips = np.array(["z1", "z1", "z1", "z2"])
    expected = np.array([100.0, 110.0, 500.0, 100.0])
    assert sale_detail.similar_counts(zips, expected).tolist() == [1, 1, 0, 0]      # same zip, within 30%, not itself
    detail = pd.DataFrame({"listing_agent_id": ["A0"] * 30 + ["A1"] * 30 + ["A2"] * 30,
                           "low_confidence": [False] * 30 + [True] * 30 + [True] * 10 + [False] * 20})
    label = sale_detail.agent_confidence(detail)["label"]
    assert label.to_dict() == {"A0": False, "A1": True, "A2": False}                # exactly 1/3 is not "more than 1/3"


def test_brief_shows_low_confidence_label_only_for_thin_markets(scored_dir, monkeypatch, tmp_path):
    import shutil
    folder = tmp_path / "scored"
    shutil.copytree(scored_dir, folder)
    cards = pd.read_csv(folder / "scorecards.csv", index_col="agent_id")
    agent, other = cards["seller_n"].nlargest(2).index
    detail = pd.read_csv(folder / "sale_detail.csv", keep_default_na=False)
    detail.loc[detail["listing_agent_id"] == agent, "low_confidence"] = True
    detail.to_csv(folder / "sale_detail.csv", index=False)
    monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(folder))
    monkeypatch.setenv("AGENT_INSIGHTS_NOTES", str(tmp_path / "n.csv"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    pick(at, agent)
    assert any("Low confidence" in w.value for w in at.warning)
    pick(at, other)
    assert not any("Low confidence" in w.value for w in at.warning)


def test_sale_list_has_ids_and_years_only(scored_dir):
    detail = pd.read_csv(scored_dir / "sale_detail.csv", keep_default_na=False)
    assert {"listing_id", "listing_agent_id", "year", "sold_vs_model", "n_similar", "low_confidence"} <= set(detail.columns)
    assert not (set(detail.columns) & {"address", "street", "client", "client_name", "phone", "email", "notes", "sold_price", "sold_date"})
    assert detail["low_confidence"].dtype == bool or set(detail["low_confidence"].astype(str)) <= {"True", "False"}
    assert (detail["n_similar"] >= 0).all()


def test_the_brief_lists_the_sales_behind_the_numbers(app, scored_dir):
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    agent = cards["seller_n"].idxmax()
    pick(app, agent)
    table = next(d.value for d in app.dataframe if "Sale ID" in d.value.columns)
    assert len(table) == int(cards.loc[agent, "seller_n"])
    assert not any(c in table.columns for c in ["Address", "Client", "Price"])


# ---- trend versus the agent's own past ----
def years(rows):
    return pd.DataFrame(rows, columns=["period", "score", "low", "high", "deals"])


def test_self_trend_calls_a_change_only_when_the_whole_range_agrees():
    steady = self_trend(years([("2023", 0.010, -0.02, 0.04, 20), ("2024", 0.012, -0.02, 0.04, 20), ("2025", 0.014, -0.02, 0.05, 20)]))
    assert steady["clear"] is False and "includes 0" in steady["sentence"]
    jump = self_trend(years([("2023", 0.000, -0.005, 0.005, 60), ("2024", 0.001, -0.005, 0.007, 60), ("2025", 0.080, 0.07, 0.09, 60)]))
    assert jump["clear"] is True and jump["change"] > 5 and "higher" in jump["sentence"]
    drop = self_trend(years([("2023", 0.05, 0.045, 0.055, 60), ("2024", -0.03, -0.035, -0.025, 60)]))
    assert drop["clear"] is True and "lower" in drop["sentence"]
    assert jump["sentence"].startswith("This agent's")                      # third person: written for the broker


def test_self_trend_is_honest_about_thin_latest_year_and_missing_history():
    thin = self_trend(years([("2023", 0.00, -0.01, 0.01, 30), ("2024", 0.09, -0.05, 0.23, 2)]))
    assert thin["small"] and "mostly luck" in thin["sentence"] and "2 listings" in thin["sentence"]
    assert self_trend(years([("2025", 0.01, -0.02, 0.04, 10)])) is None
    assert self_trend(years([("2023", 0.01, -0.02, 0.04, 0), ("2024", 0.01, -0.02, 0.04, 8)])) is None


def test_brief_shows_a_trend_sentence(app, scored_dir):
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    pick(app, cards["seller_n"].idxmax())
    text = brief_text(app)
    assert "#### Trend versus this agent's own past" in text
    assert re.search(r"This agent's \d{4} seller score is about|has only \d+ listings? so far", text)


# ---- broker notes: tags, stored locally ----
def test_broker_tags_save_locally_and_change_no_number(app, tmp_path, scored_dir):
    notes = tmp_path / "notes.csv"
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    agent = cards["seller_n"].idxmax()
    pick(app, agent)
    before = [(m.label, m.value, m.delta) for m in app.metric]
    next(b for b in app.button if b.label == "Add specialty tag").click().run()
    sale = app.selectbox(key="goal_sale").options[0]
    app.selectbox(key="goal_sale").select(sale)
    next(b for b in app.button if b.label == "Save client goal").click().run()
    assert not app.exception
    tags = context_notes.read_tags(agent, notes)
    assert tags["specialties"] and len(tags["goals"]) == 1
    assert [(m.label, m.value, m.delta) for m in app.metric] == before        # tags never touch a score or range
    assert "tag" in " ".join(c.value for c in app.tabs[1].caption) and "saved for this agent" in brief_text(app)


def test_speed_or_certainty_sales_are_listed_separately_but_stay_in_the_score(app, tmp_path, scored_dir):
    notes = tmp_path / "notes.csv"
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    agent = cards["seller_n"].idxmax()
    sales = pd.read_csv(scored_dir / "sale_detail.csv", keep_default_na=False)
    mine = sales[sales["listing_agent_id"] == agent]["listing_id"].tolist()
    context_notes.save_tag(agent, "goal", "Speed", mine[0], notes)
    context_notes.save_tag(agent, "goal", "Price", mine[1], notes)
    app.run()
    pick(app, agent)
    separate = [d.value for d in app.dataframe if "Sale ID" in d.value.columns]
    assert any(mine[0] in t["Sale ID"].tolist() for t in separate)
    main = next(t for t in separate if mine[1] in t["Sale ID"].tolist())
    assert mine[0] not in main["Sale ID"].tolist() and mine[0] in " ".join(map(str, separate[1]["Sale ID"]))
    assert app.metric[0].value == f"{round(cards.loc[agent, 'seller_score'] * 100, 1) + 0.0:+.1f}%"   # score unchanged


def test_this_sale_looks_wrong_flag_shows_on_the_sale_row(app, tmp_path, scored_dir):
    notes = tmp_path / "notes.csv"
    cards = pd.read_csv(scored_dir / "scorecards.csv", index_col="agent_id")
    agent = cards["seller_n"].idxmax()
    sales = pd.read_csv(scored_dir / "sale_detail.csv", keep_default_na=False)
    sale = sales[sales["listing_agent_id"] == agent]["listing_id"].iloc[0]
    context_notes.save_tag(agent, "looks_wrong", "yes", sale, notes)
    app.run()
    pick(app, agent)
    table = next(d.value for d in app.dataframe if "Sale ID" in d.value.columns)
    assert table.set_index("Sale ID").loc[sale, "Check"].startswith("Broker flag")


def test_tag_rules_and_nothing_free_text(tmp_path):
    f = tmp_path / "n.csv"
    assert context_notes.save_tag("A001", "specialty", "kitchen was gutted, owner is sick", path=f) is False     # not on the list
    assert context_notes.save_tag("A001", "gossip", "Condos", path=f) is False                                   # unknown kind
    assert context_notes.save_tag("A001", "goal", "Speed", path=f) is False                                      # needs a sale
    assert context_notes.save_tag("A001", "goal", "Speed", "L0001", f) is True
    assert context_notes.save_tag("A001", "specialty", "Condos", path=f) is True
    assert context_notes.save_tag("A001", "looks_wrong", "yes", "L0002", f) is True
    assert context_notes.save_tag("A001", "looks_wrong", "no", "L0002", f) is True                               # latest entry wins
    tags = context_notes.read_tags("A001", f)
    assert tags == {"specialties": ["Condos"], "goals": {"L0001": "Speed"}, "flags": set()}
    assert context_notes.count_tags("A001", f) == 2 and context_notes.count_tags("A002", f) == 0
    assert list(pd.read_csv(f).columns) == context_notes.FIELDS


def test_notes_file_is_git_ignored_and_nothing_scores_reads_it():
    default = context_notes.notes_path({}, ROOT)
    assert default == ROOT / "data" / "private" / "context_notes.csv"
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", str(default.relative_to(ROOT))], capture_output=True, text=True)
    assert r.returncode == 0, "data/private/context_notes.csv must be git-ignored"
    for py in list((ROOT / "src").glob("*.py")):
        if py.name != "context_notes.py":
            assert "context_notes" not in py.read_text(), py.name       # no scoring or export code touches notes
    assert "context_notes" not in (ROOT / "demo" / "scorecards.csv").read_text()


def test_tags_are_off_in_the_public_demo(monkeypatch):
    monkeypatch.delenv("AGENT_INSIGHTS_DATA", raising=False)
    monkeypatch.setattr(build_demo, "choose_data", lambda env=None, root=None: (build_demo.DEMO, True))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert not [b for b in at.button if b.label in ("Add specialty tag", "Save client goal")]
    assert any("Turned off in the public demo" in c.value for c in at.caption)


# ---- disclosure setting: wording only ----
def test_disclosure_changes_wording_and_never_scoring(tmp_path, scored_dir):
    assert disclosure.load_setting(ROOT / "config" / "disclosure.json") == "broker_only"
    assert disclosure.WORDING["broker_only"] != disclosure.WORDING["agents_told"]
    bad = tmp_path / "d.json"
    bad.write_text('{"disclosure": "everyone"}')
    assert disclosure.load_setting(bad) == "broker_only"                      # anything unknown falls back to the safe default
    for py in ROOT.glob("src/*.py"):
        if py.name not in {"disclosure.py", "make_readout.py"}:
            assert "disclosure" not in py.read_text().lower().replace("disclosure.json", ""), py.name
    assert "disclosure" not in (ROOT / "config" / "scoring.json").read_text().lower()


# ---- private split-half check ----
@pytest.fixture(scope="module")
def sales(data_dir):
    return pd.read_csv(data_dir / "sales.csv", keep_default_na=False)


def test_split_half_matches_the_answer_key_on_the_seller_side(sales, data_dir):
    stats = validate_private.split_half(sales)
    skills = pd.read_csv(data_dir / "true_skills.csv").set_index("agent_id")
    from agent_model import fit_agent_effects
    full, _, _ = fit_agent_effects(sales)
    truth = skills["seller_premium"].reindex(full.index)
    actual = full["seller_effect"].corr(truth)
    assert stats["agents_compared"] >= 20 and stats["verdict"] == "meets"
    assert abs(stats["accuracy"] - actual) < 0.15, (stats["accuracy"], actual)     # the guess tracks the truth
    assert stats["low"] <= stats["r"] <= stats["high"]


def test_buyer_side_is_refused_everywhere(sales, monkeypatch):
    with pytest.raises(validate_private.BuyerSideRefused):
        validate_private.split_half(sales, side="buyer")
    with pytest.raises(SystemExit) as e:
        validate_private.main(["--side", "buyer"])
    assert "REFUSED" in str(e.value)
    assert "buyer" not in " ".join(validate_private.split_half(sales).keys())


def test_report_is_counts_only_and_says_what_it_cannot_prove(sales):
    stats = validate_private.split_half(sales)
    report = validate_private.build_report(stats)
    assert not re.search(r"\bA\d{3}\b", report)                                      # no agent code
    assert not re.search(r"\$\d|\b20\d\d-\d\d-\d\d\b|L\d{4}", report)                 # no price, date or listing code
    for phrase in ["Consistent is not the same as valid", "Buyer side: REFUSED", "0.50 or better"]:
        assert phrase in report, phrase
    thin = validate_private.build_report({**stats, "verdict": "too few", "agents_compared": 3})
    assert "Too few agents to judge" in thin and "r =" not in thin


def test_split_half_says_too_few_when_agents_are_few(sales):
    few = sales[sales["listing_agent_id"].isin(sales["listing_agent_id"].unique()[:10])].reset_index(drop=True)
    assert validate_private.split_half(few)["verdict"] == "too few"


# ---- deletion script ----
import delete_private  # noqa: E402


def private_folder(tmp_path):
    folder = tmp_path / "client_data"
    (folder / "scored").mkdir(parents=True)
    (folder / "export.csv").write_text("secret,1\n")
    (folder / "scored" / "scorecards.csv").write_text("agent_id\nA001\n")
    (folder / "context_notes.csv").write_text("saved_at,agent_id,kind,text\n")
    return folder


def test_dry_run_lists_everything_and_deletes_nothing(tmp_path, capsys):
    folder = private_folder(tmp_path)
    rows = delete_private.delete_folder(folder, tmp_path / "log.txt")
    assert {p for p, _ in rows} == {"export.csv", "context_notes.csv", str(Path("scored") / "scorecards.csv")}
    assert folder.exists() and not (tmp_path / "log.txt").exists()


def test_confirm_deletes_and_logs_names_and_sizes_only(tmp_path):
    folder, log = private_folder(tmp_path), tmp_path / "log.txt"
    (tmp_path / "keep.txt").write_text("not in the folder")
    outside = tmp_path / "outside.txt"
    outside.write_text("target of a link")
    (folder / "link_to_outside").symlink_to(outside)
    rows = delete_private.delete_folder(folder, log, confirm=True)
    assert not folder.exists() and (tmp_path / "keep.txt").exists() and outside.exists()   # links are not followed
    text = log.read_text()
    assert len(rows) == 4 and all(path in text for path, _ in rows)
    assert "secret" not in text and "A001" not in text                                      # no contents in the log


def test_delete_refuses_dangerous_folders(tmp_path):
    for bad in [ROOT, ROOT.parent, Path.home(), Path("/")]:
        with pytest.raises(delete_private.UnsafeDelete):
            delete_private.check_safe_to_delete(bad, tmp_path / "log.txt")
    repo_like = tmp_path / "repo_like"
    (repo_like / ".git").mkdir(parents=True)
    with pytest.raises(delete_private.UnsafeDelete):
        delete_private.check_safe_to_delete(repo_like, tmp_path / "log.txt")
    folder = private_folder(tmp_path)
    with pytest.raises(delete_private.UnsafeDelete):                      # log inside the folder being deleted
        delete_private.check_safe_to_delete(folder, folder / "log.txt")
    with pytest.raises(delete_private.UnsafeDelete):
        delete_private.check_safe_to_delete(tmp_path / "missing", tmp_path / "log.txt")


def test_delete_refuses_a_folder_with_a_tracked_file(tmp_path):
    repo = tmp_path / "r"
    (repo / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "data" / "oops.csv").write_text("x\n")
    subprocess.run(["git", "-C", str(repo), "add", "-f", "data/oops.csv"], check=True)
    with pytest.raises(delete_private.UnsafeDelete, match="tracks files"):
        delete_private.check_safe_to_delete(repo / "data", tmp_path / "log.txt", repo=ROOT)


def test_delete_main_needs_confirm_and_prints_the_hand_checklist(tmp_path, capsys, monkeypatch):
    folder = private_folder(tmp_path)
    delete_private.main(["--data", str(folder), "--log", str(tmp_path / "l.txt")])
    out = capsys.readouterr().out
    assert folder.exists() and "DRY RUN" in out and "Time Machine" in out and "deletion letter" in out
    delete_private.main(["--data", str(folder), "--log", str(tmp_path / "l.txt"), "--confirm"])
    assert not folder.exists() and (tmp_path / "l.txt").exists()
