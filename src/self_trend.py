"""Phase 9: "this agent versus their own past", from the per-year rows of trends.csv.

One agent's seller score per year is already measured against the average agent in that same year, so a
hot market lifts everyone and never shows up as a change. This compares the agent's latest year with the
deal-weighted average of their own earlier years, and gives a range for the difference.

The range is deliberately wide: the two yearly ranges are combined as if their errors were independent,
and each already includes the bias allowance, so the shared bias is counted twice. A change that survives
this is solid. One that doesn't is most often luck evening out.
"""

import numpy as np
import pandas as pd

Z95 = 1.96


def self_trend(t, small_sample=5):
    """t: one agent's rows from trends.csv with kind == 'year'. Returns a dict (or None when there is
    nothing to compare): latest, change (in points of percent), low, high, clear, small, sentence."""
    t = t[t["deals"] > 0].sort_values("period")
    if len(t) < 2:
        return None
    latest, earlier = t.iloc[-1], t.iloc[:-1]
    w = earlier["deals"] / earlier["deals"].sum()
    past = float((w * earlier["score"]).sum())
    se_past = float(np.sqrt(((w * (earlier["high"] - earlier["low"]) / (2 * Z95)) ** 2).sum()))
    se_now = float((latest["high"] - latest["low"]) / (2 * Z95))
    change = float(latest["score"]) - past
    half = Z95 * float(np.sqrt(se_past**2 + se_now**2))
    low, high = (change - half) * 100, (change + half) * 100
    out = {"period": latest["period"], "change": change * 100, "low": low, "high": high,
           "clear": bool(low > 0 or high < 0), "small": bool(latest["deals"] < small_sample),
           "first_period": earlier["period"].iloc[0]}

    def pts(x):
        return f"{round(x, 1) + 0.0:+.1f} points"
    span = f"{pts(low)} to {pts(high)}"
    if out["small"]:
        n = int(latest["deals"])
        out["sentence"] = (f"This agent's latest year ({latest['period']}) has only {n} listing{'s' if n != 1 else ''} so far, "
                           "so any change from their past is mostly luck until more sales close.")
    elif out["clear"]:
        direction = "higher" if change > 0 else "lower"
        out["sentence"] = (f"This agent's {latest['period']} seller score is about {abs(round(change * 100, 1)):.1f} points {direction} than "
                           f"their own earlier years (range {span}), and even the end of the range nearest 0 agrees.")
    else:
        direction = "higher" if change > 0 else "lower"
        out["sentence"] = (f"This agent's {latest['period']} seller score is about {abs(round(change * 100, 1)):.1f} points {direction} than "
                           f"their own earlier years, but the range ({span}) includes 0, so we can't say their results changed.")
    return out
