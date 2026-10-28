"""Coaching view for agent-insights. Run with:  streamlit run app/coaching_app.py
Reads data/fake/rankings.csv (built by src/score_agents.py). Fake data only.

A private copy can point at its own folder by setting AGENT_INSIGHTS_DATA (for example data/private).
"""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AGENT_INSIGHTS_DATA", ROOT / "data" / "fake"))
GROUPS = ["Clearly above average", "Not clearly different from average", "Clearly below average"]
GROUP_COLORS = ["#2a78d6", "#898781", "#eb6834"]   # color follows the group, never the rank
PARTS = {  # name in rankings.csv -> (label, one-sentence meaning, how to show the number)
    "seller": ("Seller side", "Sells homes for this much above or below what the home predicts.", "pct"),
    "buyer": ("Buyer side", "Buys homes for clients this much below or above what the home predicts.", "pct"),
    "offer_win": ("Offers that win", "On the same home, at the same price and terms, how much more often this "
                  "agent's offers win (win-chance points in a typical 3-offer contest).", "pts"),
}


@st.cache_data
def load(path):
    return pd.read_csv(Path(path) / "rankings.csv", index_col="agent_id")


def fmt(x, kind):
    if pd.isna(x):
        return "n/a"
    x = round(x * 100, 1 if kind == "pct" else 0) + 0.0   # + 0.0 turns -0.0 into 0.0
    return f"{x:+.1f}%" if kind == "pct" else f"{x:+.0f} pts"


st.set_page_config(page_title="agent-insights", layout="wide")
st.title("Agent coaching view")
st.caption("FAKE DATA. Every score compares the agent with the average agent at this brokerage, after "
           "adjusting for the home, zip and month. Each score has a 95% range: if the range includes 0, "
           "we can't tell this agent from average yet. That is an honest, common answer.")

if not (DATA / "rankings.csv").exists():
    st.error("rankings.csv not found. Run the pipeline first (README, 'Run it'), ending with score_agents.py.")
    st.stop()

table = load(str(DATA))
agent = st.sidebar.selectbox("Agent", sorted(table.index))
min_deals = st.sidebar.slider("In the team chart, hide agents with fewer deals than", 0, 40, 0)

# ---- One agent ----
row = table.loc[agent]
st.subheader(f"Agent {agent}  ·  {row['office']}")
if pd.isna(row["coaching_index"]):
    st.metric("Coaching index", "n/a")
    st.write("**No data.** No listings on record yet, so there is no index. Context scores are below.")
else:
    st.metric("Coaching index", f"{row['coaching_index']:+.2f}",
              f"range {row['index_low']:+.2f} to {row['index_high']:+.2f}", delta_color="off")
    st.write(f"**{row['index_group']}.** The index is built from the scored parts below. +1 means about one "
             "typical step better than the average agent.")
if row["low_data_flag"]:
    st.warning("Fewer than 5 deals on record. Treat every number here as a first look, not a verdict.")

cols = st.columns(3)
for col, (part, (label, meaning, kind)) in zip(cols, PARTS.items()):
    n = int(row[f"{part}_n"])
    if pd.isna(row[f"{part}_score"]):
        col.write(f"**{label}:** no deals on record.")
        continue
    col.metric(label, fmt(row[f"{part}_score"], kind),
               f"range {fmt(row[f'{part}_low'], kind)} to {fmt(row[f'{part}_high'], kind)}", delta_color="off")
    unit = ("offer" if part == "offer_win" else "deal") + ("" if n == 1 else "s")
    col.caption(f"{meaning} {row[f'{part}_group']} ({n} {unit}).")
if (table[[f"{p}_group" for p in PARTS]] == "Context only").any().any():
    st.caption("Context only = shown for coaching conversations, not in the index and never called "
               "clearly above or below average. Buyer scores need an independent value estimate of each home "
               "to be trusted. The offer score already allows for price and terms, but its ranges are still a "
               "little too narrow in testing, so it waits for one more check.")

with st.expander("Context, not scored: speed, pricing style and offers attracted"):
    st.write(f"Days on market vs similar homes: {fmt(row['speed_vs_similar'], 'pct')}. "
             f"Typical first list price vs model value: {fmt(row['list_vs_value'], 'pct')}. "
             f"Share of listings with a price cut: {row['share_with_price_cut']:.0%}. "
             f"Offers drawn vs similar listings at a similar list price: {fmt(row['offers_attracted'], 'pct')} "
             f"(range {fmt(row['offers_attracted_low'], 'pct')} to {fmt(row['offers_attracted_high'], 'pct')})."
             if pd.notna(row["speed_vs_similar"]) else "No listings on record.")
    st.caption("The agent chooses the list price, so these describe style, not skill. "
               "Listing high goes with slower sales, but listing close to value does not predict better results. "
               "Listing low draws more offers, so offers attracted is compared with listings priced alike.")

# ---- Whole team ----
st.subheader("Everyone, with ranges")
view = table[table["coaching_index"].notna()].assign(deals=table["seller_n"] + table["buyer_n"]).reset_index()
view = view[view["deals"] >= min_deals]
base = alt.Chart(view).encode(
    y=alt.Y("agent_id:N", sort="-x", axis=None),
    color=alt.Color("index_group:N", title=None, scale=alt.Scale(domain=GROUPS, range=GROUP_COLORS),
                    legend=alt.Legend(orient="top")))
chart = (
    base.mark_rule(strokeWidth=2, opacity=0.5).encode(x="index_low:Q", x2="index_high:Q")
    + base.mark_point(filled=True, size=40).encode(
        x=alt.X("coaching_index:Q", title="Coaching index (0 = average agent)"),
        tooltip=["agent_id", "office", "deals", alt.Tooltip("coaching_index:Q", format="+.2f"),
                 alt.Tooltip("index_low:Q", format="+.2f"), alt.Tooltip("index_high:Q", format="+.2f")])
).properties(height=max(300, 6 * len(view)))
st.altair_chart(chart, width="stretch")
counts = view["index_group"].value_counts()
st.write(" · ".join(f"{g}: {counts.get(g, 0)}" for g in GROUPS))
