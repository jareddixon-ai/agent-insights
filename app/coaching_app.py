"""Coaching view for agent-insights. Run with:  streamlit run app/coaching_app.py
Shows each agent's scores with an honest range. Fake data only."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data" / "fake"
NEEDED = ["seller_scores.csv", "buyer_scores.csv", "offer_win_rates.csv"]


@st.cache_data
def load():
    seller = pd.read_csv(DATA / "seller_scores.csv", index_col="agent_id")
    buyer = pd.read_csv(DATA / "buyer_scores.csv", index_col="agent_id")
    wins = pd.read_csv(DATA / "offer_win_rates.csv", index_col="agent_id")
    # Buyer score is a price effect (lower price = better), so flip it into "savings".
    buyer = buyer.assign(score=-buyer["score"], low=-buyer["high"], high=-buyer["low"])
    return seller, buyer, wins


def verdict(row):
    if row["low"] > 0:
        return "Clearly above average"
    if row["high"] < 0:
        return "Clearly below average"
    return "Not clearly different from average"


def pct(x):
    return f"{x * 100:+.1f}%"


st.set_page_config(page_title="agent-insights", layout="wide")
st.title("Agent coaching view")
st.caption("FAKE DATA. Scores are percent differences versus the average agent, after adjusting for the "
           "home, zip, and month. A range that includes 0 means we cannot tell this agent from average yet.")

if not all((DATA / f).exists() for f in NEEDED):
    st.error("Scores not found. Run the pipeline first: see the README, 'Run it' section.")
    st.stop()

seller, buyer, wins = load()
seller["verdict"] = seller.apply(verdict, axis=1)

agent = st.sidebar.selectbox("Agent", sorted(seller.index))
min_deals = st.sidebar.slider("Hide agents with fewer listings than", 0, 30, 0)

st.subheader(f"Agent {agent}")
c1, c2, c3 = st.columns(3)
s = seller.loc[agent]
c1.metric("Seller side: sold vs expected", pct(s["score"]), f"range {pct(s['low'])} to {pct(s['high'])}", delta_color="off")
c1.write(f"{s['verdict']} ({int(s['deals'])} listings)")
if agent in buyer.index:
    b = buyer.loc[agent]
    c2.metric("Buyer side: savings vs expected", pct(b["score"]), f"range {pct(b['low'])} to {pct(b['high'])}", delta_color="off")
    c2.write(f"{verdict(b)} ({int(b['deals'])} purchases)")
else:
    c2.write("No buyer-side deals for this agent.")
if agent in wins.index:
    w = wins.loc[agent]
    c3.metric("Offers that win", f"{w['win_rate']:.0%}", f"range {w['low']:.0%} to {w['high']:.0%}", delta_color="off")
    c3.write(f"{int(w['wins'])} of {int(w['offers'])} offers won (raw {w['raw_rate']:.0%})")
else:
    c3.write("No offers on record.")

st.subheader("Everyone, with ranges")
view = seller[seller["deals"] >= min_deals].reset_index()
view["order"] = view["score"].rank(method="first")
base = alt.Chart(view).encode(y=alt.Y("agent_id:N", sort="-x", axis=None))
chart = (
    base.mark_rule().encode(x="low:Q", x2="high:Q", color="verdict:N")
    + base.mark_point(filled=True).encode(x=alt.X("score:Q", title="Seller score (fraction, 0.01 = 1%)"),
                                          color="verdict:N", tooltip=["agent_id", "deals", "score", "low", "high"])
).properties(height=max(300, 7 * len(view)))
st.altair_chart(chart, width="stretch")
counts = view["verdict"].value_counts()
st.write(", ".join(f"{v}: {n}" for v, n in counts.items()))
