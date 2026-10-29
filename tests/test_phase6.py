"""Phase 6 tests: the dashboard (leader view and agent report card), trends, context ranges and
the coaching-point rules."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

import score_agents
from coaching_points import coaching_points

APP = Path(__file__).resolve().parent.parent / "app" / "coaching_app.py"
NOT_CLEAR = "Not clearly different from average"


@pytest.fixture(scope="module")
def cfg():
    return score_agents.load_config()


@pytest.fixture(scope="module")
def rankings(scored_dir):
    return pd.read_csv(scored_dir / "rankings.csv", index_col="agent_id")


@pytest.fixture
def app(scored_dir, monkeypatch):
    monkeypatch.setenv("AGENT_INSIGHTS_DATA", str(scored_dir))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    return at


def card_text(at):
    """All the words on the report card tab."""
    tab = at.tabs[1]
    parts = [m.value for m in tab.markdown] + [c.value for c in tab.caption] + [s.value for s in tab.subheader]
    parts += [f"{m.label} {m.value}" for m in tab.metric]
    return "\n".join(parts)


def chart_values(element):
    """Every value in a chart's embedded data tables."""
    out = []
    for ds in element.proto.datasets:
        df = pa.ipc.open_stream(ds.data.data).read_all().to_pandas()
        out += [str(c) for c in df.columns] + [str(v) for v in df.to_numpy().ravel()]
    return out


# ---- Scoring pieces ----
def test_apply_weights_reproduces_the_saved_index(rankings, cfg):
    again = score_agents.apply_weights(rankings, cfg["weights"])
    for col in ["coaching_index", "index_low", "index_high"]:
        np.testing.assert_allclose(again[col], rankings[col], rtol=1e-9, equal_nan=True)
    assert (again["index_group"] == rankings["index_group"]).all()
    turned_on = score_agents.apply_weights(rankings, {"seller": 0.5, "offer_win": 0.2})
    assert (turned_on["offer_win_group"] != "Context only").any()
    assert not np.allclose(turned_on["coaching_index"].fillna(0), rankings["coaching_index"].fillna(0))


def test_trends_have_ranges_and_small_sample_flags(scored_dir, cfg):
    t = pd.read_csv(scored_dir / "trends.csv")
    assert set(t["kind"]) == {"year", "half"}
    assert ((t["low"] <= t["score"]) & (t["score"] <= t["high"])).all()
    assert (t["small_sample"] == (t["deals"] < cfg["small_sample"])).all()
    assert ((t["high"] - t["low"]) >= 2 * 1.96 * cfg["bias_allowance"] - 1e-9).all()   # never narrower than the table
    half = t[t["kind"] == "half"]
    assert half["small_sample"].mean() > 0.3          # splitting time makes samples small: the trap is real
    skills = pd.read_csv(scored_dir / "true_skills.csv").set_index("agent_id")["seller_premium"]
    truth = (skills - skills.mean()).reindex(t["agent_id"]).to_numpy()
    assert ((truth >= t["low"]) & (truth <= t["high"])).mean() > 0.9   # constant true skill caught 90%+


def test_context_numbers_have_ranges(rankings):
    has = rankings["list_vs_value"].notna()
    r = rankings[has]
    for col, lo, hi in [("speed_vs_similar", "speed_low", "speed_high"),
                        ("list_vs_value", "list_vs_value_low", "list_vs_value_high"),
                        ("share_with_price_cut", "cut_low", "cut_high")]:
        assert ((r[lo] <= r[col] + 1e-9) & (r[col] <= r[hi] + 1e-9)).all(), col
    assert r["cut_low"].ge(0).all() and r["cut_high"].le(1).all()


# ---- Coaching-point rules ----
def one_sentence(text):
    """No sentence break inside (a period, ? or ! followed by a space and a capital letter)."""
    return not re.search(r"[.?!]\s+[A-Z]", text) and text.endswith((".", "?"))


def test_coaching_points_follow_the_rules_for_every_agent(rankings, cfg):
    for agent, row in rankings.iterrows():
        points = coaching_points(row, cfg["weights"], cfg["validation"], cfg["small_sample"])
        assert 2 <= len(points) <= 3, agent
        for label, text in points:
            assert one_sentence(text), text
            if text.startswith(("Strength", "Focus area")):
                part = "seller"   # the only part that counts and passed validation
                assert cfg["validation"][part]["passed"] and cfg["weights"][part] > 0
                side = 1 if text.startswith("Strength") else -1
                assert (row["seller_low"] > 0) if side > 0 else (row["seller_high"] < 0), (agent, text)
            if "buyers paid" in text or "offers' chance" in text:
                assert text.startswith("Worth a look"), text    # context-only parts never get a verdict


def test_coaching_points_on_made_up_rows(cfg):
    base = {f"{p}_{c}": np.nan for p in ["seller", "buyer", "offer_win"] for c in ["score", "low", "high"]}
    base.update({"seller_n": 30, "speed_vs_similar": 0.0, "speed_low": -0.1, "speed_high": 0.1,
                 "list_vs_value": 0.0, "list_vs_value_low": -0.02, "list_vs_value_high": 0.02,
                 "offers_attracted": 0.0, "offers_attracted_low": -0.1, "offers_attracted_high": 0.1})
    w, v = cfg["weights"], cfg["validation"]

    strong = pd.Series({**base, "seller_score": 0.03, "seller_low": 0.01, "seller_high": 0.05})
    assert coaching_points(strong, w, v)[0][1].startswith("Strength")

    unclear = pd.Series({**base, "seller_score": 0.03, "seller_low": -0.01, "seller_high": 0.07,
                         "buyer_score": 0.04, "buyer_low": 0.01, "buyer_high": 0.07})
    texts = [t for _, t in coaching_points(unclear, w, v)]
    assert "not clearly different from average" in texts[0]          # the counted part leads, honestly
    assert not any(t.startswith(("Strength", "Focus area")) for t in texts)
    assert any(t.startswith("Worth a look") and "buyers paid" in t for t in texts)   # clear, but context only

    # If a leader's config ever turned the buyer score on WITHOUT it passing validation, still no verdict.
    texts = [t for _, t in coaching_points(unclear, {"seller": 1, "buyer": 0.3}, v)]
    assert not any(t.startswith("Strength") for t in texts)


# ---- Leader view ----
def test_office_and_team_filters(app, rankings):
    assert len(app.dataframe[0].value) == len(rankings)
    app.multiselect(key="office").set_value(["Office A"]).run()
    shown = app.dataframe[0].value
    assert len(shown) == (rankings["office"] == "Office A").sum() and set(shown["Office"]) == {"Office A"}
    app.multiselect(key="team").set_value(["Team 3"]).run()
    shown = app.dataframe[0].value
    assert set(shown["Team"]) <= {"Team 3"} and set(shown["Office"]) <= {"Office A"}
    expected = ((rankings["office"] == "Office A") & (rankings["team"] == "Team 3")).sum()
    assert len(shown) == expected


def test_weight_sliders_warn_on_failed_parts(app, cfg):
    app.toggle(key="explore").set_value(True).run()                    # sliders live behind "Explore" (Phase 7)
    assert [s.value for s in [app.slider(key=f"w_{p}") for p in ["seller", "buyer", "offer_win"]]] == \
        [cfg["weights"]["seller"], cfg["weights"]["buyer"], cfg["weights"]["offer_win"]]
    assert len(app.warning) == 0                                       # tested settings: no warning
    before = app.dataframe[0].value["Coaching index"].tolist()
    card_before = app.metric[0].value

    app.slider(key="w_buyer").set_value(0.3).run()
    warnings = [w.value for w in app.warning]
    assert any("Buyer side" in w and "did not pass validation" in w and cfg["validation"]["buyer"]["why"] in w
               for w in warnings)
    app.slider(key="w_offer_win").set_value(0.2).run()
    assert sum("did not pass validation" in w.value for w in app.warning) == 2
    assert app.dataframe[0].value["Coaching index"].tolist() != before   # the index really reblends
    assert app.metric[0].value == card_before                          # the report card keeps the tested settings

    for p in ["seller", "buyer", "offer_win"]:
        app.slider(key=f"w_{p}").set_value(0.0)
    app.run()
    assert not app.exception and "All weights are 0" in app.info[0].value


def test_every_leader_number_has_a_range(app):
    t = app.dataframe[0].value
    for col in ["Coaching index", "Seller side", "Buyer side", "Offer score"]:
        filled = t[col][t[col] != "n/a"]
        assert filled.str.contains(r"\(range .+ to .+\)").all(), col


def test_trend_by_year_and_half_year_flags_small_samples(app):
    leader = app.tabs[0]
    assert leader.get("vega_lite_chart")                                # team chart and trend chart
    app.radio(key="trend_by").set_value("Half-year").run()
    caps = " ".join(c.value for c in app.tabs[0].caption)
    assert "under 5 listings (hollow dots)" in caps


# ---- Report card ----
def test_report_card_points_and_offer_meanings(app, rankings):
    agent = rankings["seller_n"].idxmax()
    app.sidebar.selectbox[0].select(agent).run()
    text = card_text(app)
    bullets = [m.value for m in app.tabs[1].markdown if m.value.startswith("- ")]
    points = [b for b in bullets if not b.startswith("- Days on market")
              and not b.startswith("- Typical") and not b.startswith("- Share")]
    assert 2 <= len(points) <= 3
    assert "Offer score" in text and "3-offer contest" in text          # one-sentence meaning shown
    assert "Offers attracted" in text and "similar list price" in text


def test_report_card_shows_no_other_agent_names(app, rankings):
    agent = rankings["seller_n"].idxmax()
    app.sidebar.selectbox[0].select(agent).run()
    others = set(rankings.index) - {agent}
    words = set(re.findall(r"A\d{3}", card_text(app)))
    assert not (words & others)
    for chart in app.tabs[1].get("vega_lite_chart"):
        values = chart_values(chart)
        assert not (set(values) & others), "a chart on the report card carries another agent's id"
        assert "agent_id" not in values


def test_not_clearly_different_wording(app, rankings):
    unclear = rankings[(rankings["seller_low"] <= 0) & (rankings["seller_high"] >= 0)].index[0]
    app.sidebar.selectbox[0].select(unclear).run()
    text = card_text(app)
    assert NOT_CLEAR in text                                           # index and seller part
    assert "not clearly different from average yet" in text           # first coaching point
    assert "Strength" not in text and "Focus area" not in text
    leader = app.dataframe[0].value
    assert (leader.loc[leader["Agent"] == unclear, "Result"] == NOT_CLEAR).all()
