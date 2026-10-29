"""Coaching dashboard for agent-insights. Run with:  streamlit run app/coaching_app.py
Reads data/fake/rankings.csv (src/score_agents.py) and data/fake/trends.csv (src/trends.py). Fake data only.

Two views:
  Leader view   : office and team filters, everyone with ranges, one agent's trend. Tested weights by
                  default; weight sliders only behind an "Explore" toggle.
  Report card   : one agent against anonymous peers (shown only with 10+ peers), 2 or 3 coaching points.

A private copy points at its own scored folder with AGENT_INSIGHTS_DATA (src/run_private.py prints the command).
With no scores in data/fake/ and no private folder set (the hosted public demo), it reads the committed
fake snapshot in demo/ (src/build_demo.py).
"""

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from coaching_points import coaching_points  # noqa: E402
from score_agents import apply_weights  # noqa: E402
from build_demo import choose_data  # noqa: E402


DATA, IS_DEMO = choose_data()
CONFIG = ROOT / "config" / "scoring.json"
NOT_CLEAR = "Not clearly different from average"
GROUPS = ["Clearly above average", NOT_CLEAR, "Clearly below average"]
GROUP_COLORS = ["#2a78d6", "#898781", "#eb6834"]   # color follows the group, never the rank
NO_TEAM = "No team"
PARTS = {  # name in rankings.csv -> (label, one-sentence meaning, how to show the number)
    "seller": ("Seller side", "How much more or less this agent's listings sold for than similar homes would predict.", "pct"),
    "buyer": ("Buyer side", "How much less or more this agent's buyers paid than similar homes would predict.", "pct"),
    "offer_win": ("Offer score", "At the same price and terms on the same home, how much more or less often this agent's "
                  "offers win, in percentage points of win chance in a typical 3-offer contest.", "pts"),
}
ATTRACT_MEANING = ("How many more or fewer offers this agent's listings drew than similar listings with a similar "
                   "list price (so listing low on purpose doesn't count as skill).")


# ---------- formatting ----------
def fmt(x, kind):
    if pd.isna(x):
        return "n/a"
    if kind == "idx":
        return f"{x + 0.0:+.2f}"
    if kind == "share":
        return f"{x:.0%}"
    x = round(x * 100, 1 if kind == "pct" else 0) + 0.0   # + 0.0 turns -0.0 into 0.0
    return f"{x:+.1f}%" if kind == "pct" else f"{x:+.0f} pts"


def with_range(x, low, high, kind):
    """'+1.2% (range -0.4% to +2.8%)': every number travels with its 95% range."""
    if pd.isna(x):
        return "n/a"
    return f"{fmt(x, kind)} (range {fmt(low, kind)} to {fmt(high, kind)})"


# ---------- data ----------
@st.cache_data
def load(path):
    table = pd.read_csv(Path(path) / "rankings.csv", index_col="agent_id")
    table["team"] = table["team"].fillna(NO_TEAM).replace("", NO_TEAM)
    trends_file = Path(path) / "trends.csv"
    trends = pd.read_csv(trends_file) if trends_file.exists() else None
    return table, trends


def load_config():
    return json.loads(CONFIG.read_text())


def trend_chart(t):
    """Seller score per period with its range. Hollow dots = small samples."""
    d = t.drop(columns=["agent_id", "kind"]).assign(score_pct=t["score"] * 100, low_pct=t["low"] * 100, high_pct=t["high"] * 100,
                 sample=t["small_sample"].map({True: "Small sample (under 5 listings)", False: "5+ listings"}))
    x = alt.X("period:O", title=None, sort=sorted(d["period"].unique()), axis=alt.Axis(labelAngle=0))
    base = alt.Chart(d).encode(x=x)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="#898781").encode(y="y:Q")
    ranges = base.mark_rule(strokeWidth=2, opacity=0.5, color="#2a78d6").encode(
        y=alt.Y("low_pct:Q", title="Seller score, % vs average agent that period"), y2="high_pct:Q")
    dots = base.mark_point(size=70, color="#2a78d6", strokeWidth=2, filled=True, stroke="#2a78d6").encode(
        y="score_pct:Q",
        fillOpacity=alt.condition("datum.small_sample", alt.value(0), alt.value(1)),
        shape=alt.Shape("sample:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["period", alt.Tooltip("score_pct:Q", title="score %", format="+.1f"),
                 alt.Tooltip("low_pct:Q", title="range low %", format="+.1f"),
                 alt.Tooltip("high_pct:Q", title="range high %", format="+.1f"), "deals", "sample"])
    return (zero + ranges + dots).properties(height=280)


def show_trend(trends, agent, kind, key):
    if trends is None:
        st.info("trends.csv not found. Run src/trends.py to see scores over time.")
        return
    t = trends[(trends["agent_id"] == agent) & (trends["kind"] == kind)]
    if t.empty:
        st.write("No listings on record, so there is no trend.")
        return
    st.altair_chart(trend_chart(t), width="stretch", key=key)
    small = int(t["small_sample"].sum())
    clear_any = ((t["low"] > 0) | (t["high"] < 0)).sum()
    st.caption(f"{small} of {len(t)} periods have under 5 listings (hollow dots): read those as noise, not change. "
               f"{'No period' if clear_any == 0 else f'{clear_any} period(s)'} sit clearly away from average. "
               "Each period is compared with the average agent in that same period, after adjusting for the home "
               "and the market, so a hot spring lifts everyone, not one agent. A big jump after a small period is "
               "usually luck evening out, not a change in skill.")


# ---------- page ----------
st.set_page_config(page_title="agent-insights", layout="wide")
st.title("Agent coaching view")
st.caption("FAKE DATA. Every score compares the agent with the average agent at this brokerage, after "
           "adjusting for the home, zip and month. Each number has a 95% range (where the true value very likely "
           "sits): if the range includes 0, we can't tell this agent from average yet. That is an honest, common answer.")

if IS_DEMO:
    st.info("Public demo: a saved snapshot of the fake data (150 made-up agents, seed 42, 1,000 bootstrap "
            "resamples). No real agent, client or brokerage is in it.")

if not (DATA / "rankings.csv").exists():
    st.error("rankings.csv not found. Run the pipeline first (README, 'Run it'), ending with score_agents.py.")
    st.stop()

cfg = load_config()
MIN_PEERS = max(int(cfg.get("min_peers", 10)), 10)   # never fewer than 10, whatever the config says
full, trends = load(str(DATA))
ids = sorted(full.index)
asked = st.query_params.get("agent")   # ?agent=A133 opens that agent's card directly
agent = st.sidebar.selectbox("Agent (for the report card)", ids, index=ids.index(asked) if asked in ids else 0)
leader, card = st.tabs(["Leader view", "Agent report card"])

# ================= Leader view =================
with leader:
    f1, f2 = st.columns(2)
    offices = sorted(full["office"].unique())
    teams = sorted(full["team"].unique(), key=lambda t: (t == NO_TEAM, t))
    pick_offices = f1.multiselect("Office", offices, default=offices, key="office")
    pick_teams = f2.multiselect("Team", teams, default=teams, key="team")

    # Guard rail (Phase 7): the tested weights are the default view. Sliders appear only after a
    # leader turns on "explore", so nobody ranks people on an untested part by accident.
    tested = {name: float(cfg["weights"].get(name, 0)) for name in PARTS}
    explore = st.toggle("Explore other weights (not tested)", value=False, key="explore")
    if explore:
        st.markdown("**How much each part counts in the coaching index** (the index blends the parts; "
                    "defaults are the tested settings)")
        st.caption("**Exploring.** Numbers shown with other weights are for discussion, not for decisions "
                   "about people. The report card always keeps the tested settings. Turn exploring off to go back.")
        w_cols = st.columns(3)
        weights = {name: w_cols[i].slider(PARTS[name][0], 0.0, 1.0, tested[name], 0.1, key=f"w_{name}")
                   for i, name in enumerate(PARTS)}
    else:
        weights = tested
        st.caption("Coaching index uses the tested settings. Turn on 'Explore' to try other weights.")
    for name, w in weights.items():
        status = cfg.get("validation", {}).get(name, {})
        if w > 0 and not status.get("passed", False):
            st.warning(f"{PARTS[name][0]} is turned on but did not pass validation. {status.get('why', '')}")

    if sum(weights.values()) == 0:
        st.info("All weights are 0, so there is no coaching index. Move at least one slider above 0.")
        table = apply_weights(full, {"seller": 1.0})   # part ranges still shown; index hidden below
        table[["coaching_index", "index_low", "index_high"]] = float("nan")
        table["index_group"] = "No data"
    else:
        # Weights are applied to the WHOLE brokerage first, then filtered: every score still compares
        # the agent with the average agent at the brokerage, not with the filtered group.
        table = apply_weights(full, weights)
    view = table[table["office"].isin(pick_offices) & table["team"].isin(pick_teams)]
    total = sum(weights.values())
    st.caption("Weights used: " + ", ".join(f"{PARTS[n][0]} {w / total:.0%}" for n, w in weights.items() if w > 0)
               if total else "Weights used: none")

    min_deals = st.slider("Hide agents with fewer deals than", 0, 40, 0, key="min_deals")
    view = view[(view["seller_n"] + view["buyer_n"]) >= min_deals]
    st.subheader(f"{len(view)} agents shown")
    scored = view[view["coaching_index"].notna()]
    counts = scored["index_group"].value_counts()
    st.write(" · ".join(f"{g}: {counts.get(g, 0)}" for g in GROUPS))

    if not scored.empty:
        chart_data = scored.reset_index()[["agent_id", "office", "team", "coaching_index", "index_low",
                                           "index_high", "index_group", "seller_n"]]
        base = alt.Chart(chart_data).encode(
            y=alt.Y("agent_id:N", sort=alt.EncodingSortField("coaching_index", order="descending"), axis=None),
            color=alt.Color("index_group:N", title=None, scale=alt.Scale(domain=GROUPS, range=GROUP_COLORS),
                            legend=alt.Legend(orient="top")))
        chart = (
            base.mark_rule(strokeWidth=2, opacity=0.5).encode(x="index_low:Q", x2="index_high:Q")
            + base.mark_point(filled=True, size=40).encode(
                x=alt.X("coaching_index:Q", title="Coaching index (0 = average agent)"),
                tooltip=["agent_id", "office", "team", alt.Tooltip("seller_n", title="listings"),
                         alt.Tooltip("coaching_index:Q", format="+.2f"),
                         alt.Tooltip("index_low:Q", format="+.2f"), alt.Tooltip("index_high:Q", format="+.2f")])
        ).properties(height=max(300, 6 * len(chart_data)))
        st.altair_chart(chart, width="stretch", key="team_chart")

    shown = pd.DataFrame({
        "Agent": view.index,
        "Office": view["office"].to_numpy(),
        "Team": view["team"].to_numpy(),
        "Coaching index": [with_range(r.coaching_index, r.index_low, r.index_high, "idx") for r in view.itertuples()],
        "Result": view["index_group"].to_numpy(),
        **{PARTS[n][0]: [with_range(getattr(r, f"{n}_score"), getattr(r, f"{n}_low"), getattr(r, f"{n}_high"),
                                    PARTS[n][2]) for r in view.itertuples()] for n in PARTS},
        "Listings": view["seller_n"].to_numpy(),
        "Buyer deals": view["buyer_n"].to_numpy(),
        "Small sample": view["low_data_flag"].map({True: "Under 5 deals: first look only", False: ""}).to_numpy(),
    })
    st.dataframe(shown, hide_index=True, width="stretch", key="leader_table")
    st.caption("1 index point = about one typical step between agents. Parts with weight 0 are 'Context only': "
               "shown, never called clearly above or below.")

    st.subheader("Seller score over time")
    with_listings = sorted(view.index[view["seller_n"] > 0])
    if with_listings:
        t1, t2 = st.columns([2, 1])
        trend_agent = t1.selectbox("Agent", with_listings, key="trend_agent")
        by = t2.radio("By", ["Year", "Half-year"], horizontal=True, key="trend_by")
        show_trend(trends, trend_agent, "year" if by == "Year" else "half", key="leader_trend")
    else:
        st.write("No agents with listings match these filters.")

# ================= Agent report card =================
with card:
    row = apply_weights(full, cfg["weights"]).loc[agent]   # the card always uses the tested settings
    st.subheader(f"Report card: agent {agent} · {row['office']}")
    st.caption("This card always uses the tested settings, whatever the leader view's sliders say. "
               "Other agents appear only as anonymous dots.")
    if row["low_data_flag"]:
        st.warning("Fewer than 5 deals on record. Treat every number here as a first look, not a verdict.")

    if pd.isna(row["coaching_index"]):
        st.metric("Coaching index", "n/a")
        st.write("**No data.** No listings on record yet, so there is no index.")
    else:
        st.metric("Coaching index", fmt(row["coaching_index"], "idx"),
                  f"range {fmt(row['index_low'], 'idx')} to {fmt(row['index_high'], 'idx')}", delta_color="off", delta_arrow="off")
        st.write(f"**{row['index_group']}.** +1 means about one typical step better than the average agent.")

    st.markdown("#### Coaching points")
    for _, text in coaching_points(row, cfg["weights"], cfg.get("validation", {}), cfg["small_sample"]):
        st.markdown(f"- {text}")

    st.markdown("#### Scores against anonymous peers")
    cols = st.columns(3)
    for col, (name, (label, meaning, kind)) in zip(cols, PARTS.items()):
        n = int(row[f"{name}_n"])
        if pd.isna(row[f"{name}_score"]):
            col.write(f"**{label}:** no deals on record.")
            continue
        col.metric(label, fmt(row[f"{name}_score"], kind),
                   f"range {fmt(row[f'{name}_low'], kind)} to {fmt(row[f'{name}_high'], kind)}", delta_color="off", delta_arrow="off")
        unit = ("offer" if name == "offer_win" else "deal") + ("" if n == 1 else "s")
        col.caption(f"{meaning} **{row[f'{name}_group']}** ({n} {unit}).")
        peers = full.drop(index=agent)[f"{name}_score"].dropna() * 100    # values only: no names, no ids
        if len(peers) < MIN_PEERS:
            # A distribution of a handful of values can point to individuals (Phase 7 guard rail).
            col.caption(f"Peer comparison hidden: fewer than {MIN_PEERS} other agents have this score, so a "
                        "chart could point to individuals.")
            continue
        dots = pd.DataFrame({"score": peers.to_numpy(), "who": "Peers (anonymous)"})
        you = pd.DataFrame({"score": [row[f"{name}_score"] * 100], "low": [row[f"{name}_low"] * 100],
                            "high": [row[f"{name}_high"] * 100], "who": ["This agent"]})
        color = alt.Color("who:N", title=None, legend=None,
                          scale=alt.Scale(domain=["Peers (anonymous)", "This agent"], range=["#c3c2bd", "#2a78d6"]))
        peer_chart = (
            alt.Chart(dots).mark_tick(thickness=1, size=18, opacity=0.6).encode(
                x=alt.X("score:Q", title="% vs average" if kind == "pct" else "pts vs average"), color=color)
            + alt.Chart(you).mark_rule(strokeWidth=3).encode(x="low:Q", x2="high:Q", color=color)
            + alt.Chart(you).mark_point(filled=True, size=120).encode(x="score:Q", color=color)
        ).properties(height=70)
        col.altair_chart(peer_chart, width="stretch", key=f"peer_{name}")
        col.caption(f"Grey ticks: {len(peers)} other agents, no names. Blue: this agent and their range.")
    if (pd.Series([row[f"{p}_group"] for p in PARTS]) == "Context only").any():
        st.caption("Context only = shown for coaching conversations, not in the index and never called clearly "
                   "above or below average, because it did not pass validation yet.")

    st.markdown("#### Offers")
    o1, o2 = st.columns(2)
    o1.metric("Offer score", fmt(row["offer_win_score"], "pts"),
              f"range {fmt(row['offer_win_low'], 'pts')} to {fmt(row['offer_win_high'], 'pts')}", delta_color="off", delta_arrow="off")
    o1.caption(PARTS["offer_win"][1] + " Context only for now.")
    o2.metric("Offers attracted", fmt(row["offers_attracted"], "pct"),
              f"range {fmt(row['offers_attracted_low'], 'pct')} to {fmt(row['offers_attracted_high'], 'pct')}",
              delta_color="off", delta_arrow="off")
    o2.caption(ATTRACT_MEANING + " Context only.")

    with st.expander("Context, not scored: speed and pricing"):
        if pd.notna(row["speed_vs_similar"]):
            st.markdown(
                f"- Days on market vs similar homes: {with_range(row['speed_vs_similar'], row['speed_low'], row['speed_high'], 'pct')}\n"
                f"- Typical first list price vs model value: {with_range(row['list_vs_value'], row['list_vs_value_low'], row['list_vs_value_high'], 'pct')}\n"
                f"- Share of listings with a price cut: {with_range(row['share_with_price_cut'], row['cut_low'], row['cut_high'], 'share')}")
        else:
            st.write("No listings on record.")
        st.caption("The agent chooses the list price, so these describe style, not skill. Listing high goes with "
                   "slower sales, but listing close to value does not predict better results.")

    st.markdown("#### Seller score over time")
    card_by = st.radio("By", ["Year", "Half-year"], horizontal=True, key="card_by")
    show_trend(trends, agent, "year" if card_by == "Year" else "half", key="card_trend")
