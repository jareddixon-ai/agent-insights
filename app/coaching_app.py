"""Coaching dashboard for agent-insights, for BROKERS only. Run with:  streamlit run app/coaching_app.py
Reads scorecards.csv (src/score_agents.py), trends.csv (src/trends.py) and sale_detail.csv (src/sale_detail.py)
from data/fake/, a private scored folder, or the committed demo/ snapshot. Fake data only in this repo.

Two tabs. The app opens on the Team overview:
  Team overview  : for a broker team. Office and team filters, every agent code with ranges, sorted by agent
                   code (never by score). Tested weights by default; sliders only behind "Explore".
  Coaching brief : one page a broker opens before a one-on-one. The seller result as a percent with its range,
                   talking points, trend against the agent's own past, the sales behind every number, and broker
                   notes (tags only, saved to a local git-ignored file). An agent with under 5 sales gets
                   "building a track record" instead of a score.

This tool shows nothing to agents. The brokerage decides what its agents are told.
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
from context_notes import GOALS, SPECIALTIES, count_tags, read_tags, save_tag  # noqa: E402
from sale_detail import LOW_SHARE, MIN_SIMILAR  # noqa: E402
from self_trend import self_trend  # noqa: E402


DATA, IS_DEMO = choose_data()
CONFIG = ROOT / "config" / "scoring.json"
NOT_CLEAR = "Not clearly different from average"
GROUPS = ["Strength", NOT_CLEAR, "Room to grow"]
GROUP_COLORS = ["#0369A1", "#94A3B8", "#C2410C"]   # color follows the call, never an order
NO_TEAM = "No team"
PARTS = {  # name in scorecards.csv -> (label, one-sentence meaning, how to show the number)
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
    table = pd.read_csv(Path(path) / "scorecards.csv", index_col="agent_id")
    table["team"] = table["team"].fillna(NO_TEAM).replace("", NO_TEAM)
    trends_file, detail_file = Path(path) / "trends.csv", Path(path) / "sale_detail.csv"
    trends = pd.read_csv(trends_file) if trends_file.exists() else None
    detail = pd.read_csv(detail_file, keep_default_na=False) if detail_file.exists() else None
    return table, trends, detail


def load_config():
    return json.loads(CONFIG.read_text())


def trend_chart(t):
    """Seller score per period with its range. Hollow dots = small samples."""
    d = t.drop(columns=["agent_id", "kind"]).assign(score_pct=t["score"] * 100, low_pct=t["low"] * 100, high_pct=t["high"] * 100,
                 sample=t["small_sample"].map({True: "Small sample (under 5 listings)", False: "5+ listings"}))
    x = alt.X("period:O", title=None, sort=sorted(d["period"].unique()), axis=alt.Axis(labelAngle=0))
    base = alt.Chart(d).encode(x=x)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="#898781").encode(y="y:Q")
    ranges = base.mark_rule(strokeWidth=2, opacity=0.5, color="#0369A1").encode(
        y=alt.Y("low_pct:Q", title="Seller score, % vs average agent that period"), y2="high_pct:Q")
    dots = base.mark_point(size=70, color="#0369A1", strokeWidth=2, filled=True, stroke="#0369A1").encode(
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
TRACK_RECORD = "Building a track record"


def part_cell(r, name, kind):
    """A table cell: the number with its range, or the track-record label for an agent with under 5 sales."""
    if r.low_data_flag:
        return TRACK_RECORD
    return with_range(getattr(r, f"{name}_score"), getattr(r, f"{name}_low"), getattr(r, f"{name}_high"), kind)


def peer_chart(row, name, kind, peers):
    dots = pd.DataFrame({"score": peers.to_numpy(), "who": "Other agents (anonymous)"})
    you = pd.DataFrame({"score": [row[f"{name}_score"] * 100], "low": [row[f"{name}_low"] * 100],
                        "high": [row[f"{name}_high"] * 100], "who": ["This agent"]})
    color = alt.Color("who:N", title=None, legend=None,
                      scale=alt.Scale(domain=["Other agents (anonymous)", "This agent"], range=["#c3c2bd", "#0369A1"]))
    return (
        alt.Chart(dots).mark_tick(thickness=1, size=18, opacity=0.6).encode(
            x=alt.X("score:Q", title="% vs average" if kind == "pct" else "pts vs average"), color=color)
        + alt.Chart(you).mark_rule(strokeWidth=3).encode(x="low:Q", x2="high:Q", color=color)
        + alt.Chart(you).mark_point(filled=True, size=120).encode(x="score:Q", color=color)
    ).properties(height=70)


def show_part(col, row, name, label, meaning, kind, show_peers):
    n = int(row[f"{name}_n"])
    if pd.isna(row[f"{name}_score"]):
        col.write(f"**{label}:** no deals on record.")
        return
    col.metric(label, fmt(row[f"{name}_score"], kind),
               f"range {fmt(row[f'{name}_low'], kind)} to {fmt(row[f'{name}_high'], kind)}", delta_color="off", delta_arrow="off")
    unit = ("offer" if name == "offer_win" else "deal") + ("" if n == 1 else "s")
    col.caption(f"{meaning} **{row[f'{name}_group']}** ({n} {unit}).")
    if not show_peers:
        return
    peers = full.drop(index=row.name)[f"{name}_score"].dropna() * 100    # values only: no names, no ids
    if len(peers) < MIN_PEERS:
        # A distribution of a handful of values can point to individuals (Phase 7 guard rail).
        col.caption(f"Comparison hidden: fewer than {MIN_PEERS} other agents have this score, so a "
                    "chart could point to individuals.")
        return
    col.altair_chart(peer_chart(row, name, kind, peers), width="stretch", key=f"peer_{name}")
    col.caption(f"Grey ticks: {len(peers)} other agents, no names. Blue: this agent and their range.")


def sales_table(df, goals, flags):
    """One row per sale: sale ID and year only (no address, no names), with the broker's tags beside it."""
    return pd.DataFrame({
        "Sale ID": df["listing_id"].to_numpy(),
        "Year": df["year"].to_numpy(),
        "Sold vs model estimate": [fmt(x, "pct") for x in df["sold_vs_model"]],
        "Similar homes in the data": df["n_similar"].to_numpy(),
        "Confidence": df["low_confidence"].map({True: "Low confidence", False: ""}).to_numpy(),
        "Client goal (broker tag)": [goals.get(s, "") for s in df["listing_id"]],
        "Check": ["Broker flag: this sale looks wrong" if s in flags else "" for s in df["listing_id"]],
    })


st.set_page_config(page_title="agent-insights", layout="wide")

# Look and feel: a few CSS rules, static text only (no data is ever put into this block).
# Contrast follows the design notes (body text 4.5:1 or better), keyboard focus is visible, and motion is
# switched off for people who ask their computer for reduced motion.
st.markdown("""<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1 {font-weight: 700; letter-spacing: -0.02em; font-size: 2rem;}
h4 {margin-top: 1.2rem; letter-spacing: -0.01em;}
[data-testid="stMetricValue"] {font-size: 2.3rem; font-variant-numeric: tabular-nums;}
[data-baseweb="tab"] {font-weight: 600;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #FFFFFF;}
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible {outline: 2px solid #0369A1; outline-offset: 2px;}
@media (prefers-reduced-motion: reduce) {* {transition: none !important; animation: none !important;}}
</style>""", unsafe_allow_html=True)
st.title("Agent coaching view")
st.caption("FAKE DATA. For the broker only. Every score compares the agent with the average agent at this brokerage, "
           "after adjusting for the home, zip and month. Each number has a 95% range (where the true value very likely "
           "sits): if the range includes 0, we can't tell this agent from average yet. That is an honest, common answer.")

if IS_DEMO:
    st.info("Public demo: a saved snapshot of the fake data (150 made-up agents, seed 42, 1,000 bootstrap "
            "resamples). No real agent, client or brokerage is in it.")

if not (DATA / "scorecards.csv").exists():
    st.error("scorecards.csv not found. Run the pipeline first (README, 'Run it'), ending with score_agents.py.")
    st.stop()

cfg = load_config()
MIN_PEERS = max(int(cfg.get("min_peers", 10)), 10)   # never fewer than 10, whatever the config says
full, trends, detail = load(str(DATA))
ids = sorted(full.index)
asked = st.query_params.get("agent")   # ?agent=A133 picks that agent for the coaching brief
agent = st.sidebar.selectbox("Agent code (for the coaching brief)", ids, index=ids.index(asked) if asked in ids else 0)
team, brief = st.tabs(["Team overview", "Coaching brief"])   # the app opens on the first tab: the team overview

# ================= Team overview =================
with team:
    with st.container(border=True):
        f1, f2 = st.columns(2)
        offices = sorted(full["office"].unique())
        teams = sorted(full["team"].unique(), key=lambda t: (t == NO_TEAM, t))
        pick_offices = f1.multiselect("Office", offices, default=offices, key="office")
        pick_teams = f2.multiselect("Team", teams, default=teams, key="team")

    # Guard rail (Phase 7): the tested weights are the default view. Sliders appear only after someone
    # turns on "explore", so nobody judges people on an untested part by accident.
    tested = {name: float(cfg["weights"].get(name, 0)) for name in PARTS}
    explore = st.toggle("Explore other weights (not tested)", value=False, key="explore")
    if explore:
        st.markdown("**How much each part counts in the coaching index** (the index blends the parts; "
                    "defaults are the tested settings)")
        st.caption("**Exploring.** Numbers shown with other weights are for discussion, not for decisions "
                   "about people. The coaching brief always keeps the tested settings. Turn exploring off to go back.")
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
    view = view[(view["seller_n"] + view["buyer_n"]) >= min_deals].sort_index()   # always by agent code
    st.subheader(f"{len(view)} agents shown, listed by agent code")
    scored = view[view["coaching_index"].notna()]
    counts = scored["index_group"].value_counts()
    st.write(" · ".join(f"{g}: {counts.get(g, 0)}" for g in GROUPS))
    building = int(view["low_data_flag"].sum())
    if building:
        st.caption(f"{building} agent{'s' if building != 1 else ''} with under {cfg['small_sample']} sales show "
                   f"'{TRACK_RECORD}' instead of a score, and are left off the chart.")

    plotted = scored[~scored["low_data_flag"]]
    if not plotted.empty:
        chart_data = plotted.reset_index()[["agent_id", "office", "team", "coaching_index", "index_low",
                                            "index_high", "index_group", "seller_n"]]
        base = alt.Chart(chart_data).encode(
            y=alt.Y("agent_id:N", sort="ascending", axis=None),   # agent code order, never score order
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
        PARTS["seller"][0]: [part_cell(r, "seller", "pct") for r in view.itertuples()],
        "Coaching index": [TRACK_RECORD if r.low_data_flag else with_range(r.coaching_index, r.index_low, r.index_high, "idx")
                           for r in view.itertuples()],
        "Result": [TRACK_RECORD if low else g for low, g in zip(view["low_data_flag"], view["index_group"])],
        **{PARTS[n][0]: [part_cell(r, n, PARTS[n][2]) for r in view.itertuples()] for n in PARTS if n != "seller"},
        "Listings": view["seller_n"].to_numpy(),
        "Buyer deals": view["buyer_n"].to_numpy(),
    })
    st.dataframe(shown, hide_index=True, width="stretch", key="team_table")
    st.caption("1 index point = about one typical step between agents. Parts with weight 0 are 'Context only': "
               "shown, never called a Strength or Room to grow. Pick an agent in the sidebar and open the "
               "Coaching brief tab for a one-page brief before a one-on-one.")

    st.subheader("Seller score over time")
    with_listings = sorted(view.index[view["seller_n"] > 0])
    if with_listings:
        t1, t2 = st.columns([2, 1])
        trend_agent = t1.selectbox("Agent", with_listings, key="trend_agent")
        by = t2.radio("By", ["Year", "Half-year"], horizontal=True, key="trend_by")
        show_trend(trends, trend_agent, "year" if by == "Year" else "half", key="team_trend")
    else:
        st.write("No agents with listings match these filters.")

# ================= Coaching brief =================
with brief:
    row = apply_weights(full, cfg["weights"]).loc[agent]   # the brief always uses the tested settings
    st.subheader(f"Coaching brief: agent {agent} · {row['office']}")
    st.caption("One page to open before a one-on-one. For the broker only: this tool shows nothing to agents, and "
               "the brokerage decides what its agents are told. The brief always uses the tested settings, "
               "whatever the Team overview's Explore sliders say.")

    own = detail[detail["listing_agent_id"] == agent] if detail is not None else None
    bought = detail[detail["buyer_agent_id"] == agent] if detail is not None else None
    tags = read_tags(agent) if not IS_DEMO else {"specialties": [], "goals": {}, "flags": set()}
    if tags["specialties"]:
        st.markdown("**Specialty tags (broker):** " + ", ".join(tags["specialties"]))

    with st.container(border=True):
        st.markdown("#### Seller result")
        s_label, s_meaning, s_kind = PARTS["seller"]
        show_peers = st.toggle("Compare with other agents, anonymously (optional)", value=False, key="show_peers",
                               help="Grey ticks with no names, shown only when at least 10 other agents have the score.")
        n_total = int(row["seller_n"] + row["buyer_n"])
        if row["low_data_flag"]:
            st.info(f"**{TRACK_RECORD}.** {n_total} sale{'s' if n_total != 1 else ''} on record "
                    f"({int(row['seller_n'])} as listing agent, {int(row['buyer_n'])} for a buyer). Scores need at least "
                    f"{cfg['small_sample']} sales before they mean much, so none is shown. The sales are listed below.")
        elif pd.isna(row["seller_score"]):
            st.write("**No data.** No listings on record yet, so there is no seller result.")
        else:
            top = st.columns([2, 1])
            show_part(top[0], row, "seller", s_label, s_meaning, s_kind, show_peers)
            if own is not None and len(own) and (own["low_confidence"].mean() > LOW_SHARE):
                low_n = int(own["low_confidence"].sum())
                st.warning(f"**Low confidence.** {low_n} of this agent's {len(own)} listings sit in thin parts of the market "
                           f"(fewer than {MIN_SIMILAR} similar sales in the data), so the estimate of what those homes should "
                           "sell for is less certain, and the range above may not fully show it.")

    if not row["low_data_flag"]:
        st.markdown("#### Talking points")
        for _, text in coaching_points(row, cfg["weights"], cfg.get("validation", {}), cfg["small_sample"]):
            st.markdown(f"- {text}")

        st.markdown("#### Other parts")
        cols = st.columns(2)
        for col, name in zip(cols, ["buyer", "offer_win"]):
            label, meaning, kind = PARTS[name]
            show_part(col, row, name, label, meaning, kind, show_peers)
        if (pd.Series([row[f"{p}_group"] for p in PARTS]) == "Context only").any():
            st.caption("Context only = shown for coaching conversations, not in the coaching index and never called a "
                       "Strength or Room to grow, because it did not pass validation yet.")
        o1, _ = st.columns(2)
        o1.metric("Offers attracted", fmt(row["offers_attracted"], "pct"),
                  f"range {fmt(row['offers_attracted_low'], 'pct')} to {fmt(row['offers_attracted_high'], 'pct')}",
                  delta_color="off", delta_arrow="off")
        o1.caption(ATTRACT_MEANING + " Context only.")

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

        st.markdown("#### Trend versus this agent's own past")
        st.caption("Each year is measured against the average agent in that same year, so a hot market lifts everyone "
                   "and never shows up as a change in one agent.")
        own_years = trends[(trends["agent_id"] == agent) & (trends["kind"] == "year")] if trends is not None else None
        summary = self_trend(own_years, cfg["small_sample"]) if own_years is not None else None
        st.write(summary["sentence"] if summary else "Not enough years of listings yet to compare this agent with their own past.")
        card_by = st.radio("By", ["Year", "Half-year"], horizontal=True, key="card_by")
        show_trend(trends, agent, "year" if card_by == "Year" else "half", key="card_trend")

    st.markdown("#### Sales behind these numbers")
    if detail is None:
        st.caption("No sale list in this data folder. Run src/sale_detail.py (the pipeline does it) to list the sales.")
    else:
        st.caption("Sale IDs and the year only: no addresses, no names. 'Sold vs model estimate' is how far the price sat "
                   "from what a model that never saw the sale expected. It explains the score; it does not replace it.")
        speedy = own["listing_id"].map(tags["goals"]).isin(["Speed", "Certainty"])
        main_sales, goal_sales = own[~speedy], own[speedy]
        if own.empty:
            st.write("No listings on record.")
        else:
            st.markdown(f"**Listings ({len(main_sales)})**")
            st.dataframe(sales_table(main_sales, tags["goals"], tags["flags"]), hide_index=True, width="stretch", key="sales_main")
        if len(goal_sales):
            avg_goal, avg_rest = goal_sales["sold_vs_model"].mean(), main_sales["sold_vs_model"].mean() if len(main_sales) else float("nan")
            st.markdown(f"**Listings where the client's goal was speed or certainty ({len(goal_sales)}), shown separately**")
            st.dataframe(sales_table(goal_sales, tags["goals"], tags["flags"]), hide_index=True, width="stretch", key="sales_goal")
            st.caption("A client who wants a fast or sure sale gives up some price on purpose. These sales are still inside the "
                       f"seller result above, which is not recalculated. Average gap to the model estimate: {fmt(avg_goal, 'pct')} "
                       f"on these {len(goal_sales)} versus {fmt(avg_rest, 'pct')} on the others (plain averages, not scores).")
        if len(bought):
            st.markdown(f"**Sales where this agent represented the buyer ({len(bought)}), context only**")
            st.dataframe(sales_table(bought, tags["goals"], tags["flags"]), hide_index=True, width="stretch", key="sales_buyer")
            st.caption("Buyer scores did not pass validation, so these are listed for conversation, not judged.")
        if tags["flags"]:
            st.caption(f"{len(tags['flags'])} sale(s) flagged by the broker as looking wrong: check the source data before "
                       "using any call that depends on them.")

    st.markdown("#### Coaching index")
    if row["low_data_flag"] or pd.isna(row["coaching_index"]):
        st.write(f"{TRACK_RECORD if row['low_data_flag'] else 'No coaching index yet: no listings on record.'}")
    else:
        seller_only = all(w == 0 for n, w in cfg["weights"].items() if n != "seller")
        same = (f" Today only the seller side counts, so this is the same result as the seller percent "
                f"({fmt(row['seller_score'], 'pct')}), shown in typical steps between agents instead of percent."
                if seller_only else "")
        st.markdown(f"**{fmt(row['coaching_index'], 'idx')}** (range {fmt(row['index_low'], 'idx')} to "
                    f"{fmt(row['index_high'], 'idx')}), call: **{row['index_group']}**.{same} +1 means about one typical "
                    "step better than the average agent.")

    st.markdown("#### Broker notes")
    if IS_DEMO:
        st.caption("Turned off in the public demo: nothing chosen here would be stored.")
    else:
        st.caption("Tags, not free text, so no client names or gossip can end up in the file. Saved only in a local file on this "
                   "computer (git-ignored, never in the repo). They change no score, range or call. They only label sales "
                   "here and list speed or certainty sales separately. Never printed or logged.")
        sale_ids = sorted(own["listing_id"]) if own is not None and len(own) else []
        with st.form("tag_specialty", clear_on_submit=True):
            spec = st.selectbox("Specialty tag", SPECIALTIES)
            if st.form_submit_button("Add specialty tag"):
                if save_tag(agent, "specialty", spec):
                    st.rerun()
        if sale_ids:
            with st.form("tag_goal", clear_on_submit=True):
                g1, g2 = st.columns(2)
                sale = g1.selectbox("Sale", sale_ids, key="goal_sale")
                goal = g2.selectbox("What the client cared about most", GOALS)
                if st.form_submit_button("Save client goal"):
                    if save_tag(agent, "goal", goal, sale_id=sale):
                        st.rerun()
            with st.form("tag_wrong", clear_on_submit=True):
                wrong = st.selectbox("Sale", sale_ids, key="wrong_sale")
                if st.form_submit_button("Flag: this sale looks wrong"):
                    if save_tag(agent, "looks_wrong", "yes", sale_id=wrong):
                        st.rerun()
        n_tags = count_tags(agent)
        if n_tags:
            st.caption(f"{n_tags} tag{'s' if n_tags != 1 else ''} saved for this agent.")
