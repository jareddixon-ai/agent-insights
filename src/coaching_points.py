"""Step 6-2: two or three coaching points for one agent, written from their row in rankings.csv.

Rules (each one is tested in tests/test_phase6.py):
  1. A strength or a focus area is only named when the 95% range supports it (the whole range
     is above 0, or the whole range is below 0), and only for a part that passed validation
     and counts in the coaching index (weight above 0 in config/scoring.json).
  2. Context-only numbers (buyer score, offer score, speed, list price vs value, offers attracted)
     are worded as "Worth a look", never as a verdict, and only raised when their range is clearly
     away from 0. Otherwise there is nothing to look at.
  3. Every point is one sentence an agent would understand.
If fewer than 2 points qualify, honest fallbacks fill in ("not clearly different from average
yet", or "few listings, so luck still moves the score"). Never more than 3.
"""

import pandas as pd

MAX_POINTS = 3


def pct(x):
    return f"{abs(x) * 100:.1f}%"


def signed(x, kind="pct"):
    if pd.isna(x):
        return "n/a"
    if kind == "pts":
        return f"{round(x * 100) + 0:+.0f} pts"
    return f"{round(x * 100, 1) + 0.0:+.1f}%"


def clear(row, name):
    """+1 if the whole range is above 0, -1 if below, 0 if it includes 0 or is missing."""
    low, high = row.get(f"{name}_low"), row.get(f"{name}_high")
    if pd.isna(low) or pd.isna(high):
        return 0
    return 1 if low > 0 else (-1 if high < 0 else 0)


def strength_of_signal(row, name, score_col):
    """How far the score sits from 0, in half-widths of its range (bigger = clearer)."""
    half = (row[f"{name}_high"] - row[f"{name}_low"]) / 2
    return abs(row[score_col]) / half if half > 0 else 0


# Context-only items: (name used for the range columns, score column, sentence when above 0, below 0).
CONTEXT = [
    ("speed", "speed_vs_similar",
     "Worth a look: your listings took about {v} longer to sell than similar homes; is that a pricing choice or something to change?",
     "Worth a look: your listings sold about {v} faster than similar homes; worth checking that a quick sale didn't leave money on the table."),
    ("list_vs_value", "list_vs_value",
     "Worth a look: you usually list about {v} above our estimate of the home's value, which in this data goes with slower sales and price cuts, not higher final prices.",
     "Worth a look: you usually list about {v} below our estimate of the home's value, which draws more offers but in this data does not raise the final price."),
    ("offers_attracted", "offers_attracted",
     "Worth a look: your listings drew about {v} more offers than similar listings priced alike; worth sharing how you market them.",
     "Worth a look: your listings drew about {v} fewer offers than similar listings priced alike; worth reviewing how they are marketed."),
]
PART_CONTEXT = {
    "buyer": ("Worth a look: your buyers paid about {v} less than the model expected, but buyer scores aren't reliable yet, so treat it as a conversation starter.",
              "Worth a look: your buyers paid about {v} more than the model expected, but buyer scores aren't reliable yet, so review a few of those deals together first."),
    "offer_win": ("Worth a look: at the same price and terms, your offers' chance of winning a typical 3-offer contest is about {v} higher than average; worth asking what you do differently.",
                  "Worth a look: at the same price and terms, your offers' chance of winning a typical 3-offer contest is about {v} lower than average; worth reviewing how your offers are put together."),
}
VERDICT = {
    "seller": ("Strength: your listings sold for about {v} more than similar homes would predict, and even the low end of your range is above the average agent.",
               "Focus area: your listings sold for about {v} less than similar homes would predict, and even the high end of your range is below the average agent."),
    "buyer": ("Strength: your buyers paid about {v} less than similar homes would predict, and even the low end of your range is above the average agent.",
              "Focus area: your buyers paid about {v} more than similar homes would predict, and even the high end of your range is below the average agent."),
    "offer_win": ("Strength: at the same price and terms, your offers' chance of winning a typical 3-offer contest is about {v} higher than average, and the whole range agrees.",
                  "Focus area: at the same price and terms, your offers' chance of winning a typical 3-offer contest is about {v} lower than average, and the whole range agrees."),
}


def value_text(name, x):
    return f"{abs(round(x * 100)):.0f} percentage points" if name == "offer_win" else pct(x)


def coaching_points(row, weights, validation, small_sample=5):
    """row: one agent's row from rankings.csv (a pandas Series). Returns a list of (label, sentence)."""
    verdicts, context = [], []
    for name in ["seller", "buyer", "offer_win"]:
        score = row.get(f"{name}_score")
        if pd.isna(score):
            continue
        counted = weights.get(name, 0) > 0 and validation.get(name, {}).get("passed", False)
        side = clear(row, name)
        if side == 0:
            continue
        v = value_text(name, score)
        if counted:
            label = "Strength" if side > 0 else "Focus area"
            verdicts.append((strength_of_signal(row, name, f"{name}_score"), label,
                             VERDICT[name][0 if side > 0 else 1].format(v=v)))
        elif name in PART_CONTEXT:
            context.append((strength_of_signal(row, name, f"{name}_score"), "Worth a look",
                            PART_CONTEXT[name][0 if side > 0 else 1].format(v=v)))
    for name, col, up, down in CONTEXT:
        if pd.isna(row.get(col)) or clear(row, name) == 0:
            continue
        context.append((strength_of_signal(row, name, col), "Worth a look",
                        (up if row[col] > 0 else down).format(v=pct(row[col]))))

    # The main (counted) result comes first, even when the honest answer is "not clearly different":
    # a context-only item should never take the headline from the part that actually counts.
    n = int(row.get("seller_n", 0) or 0)
    headline = []
    if n == 0:
        headline.append(("Note", "No listings on record yet, so there is no seller score; the other numbers here are context only."))
    elif clear(row, "seller") == 0 and weights.get("seller", 0) > 0:
        headline.append(("Not clearly different", f"Your seller result ({signed(row['seller_score'])}) is not clearly "
                         f"different from average yet, because its range ({signed(row['seller_low'])} to "
                         f"{signed(row['seller_high'])}) includes 0."))
    points = [(lab, txt) for _, lab, txt in sorted(verdicts, reverse=True)] + headline
    points += [(lab, txt) for _, lab, txt in sorted(context, reverse=True)]
    points = points[:MAX_POINTS]

    # Honest fallbacks so there are always at least 2 points.
    fallbacks = []
    if 0 < n < 4 * small_sample:
        fallbacks.append(("Note", f"With {n} listing{'s' if n != 1 else ''}, luck still moves your score a lot, "
                                  f"so each new sale will make the picture sharper."))
    fallbacks.append(("Note", "Nothing else stands out clearly from average right now, which is the most common and honest result."))
    for fb in fallbacks:
        if len(points) >= 2:
            break
        points.append(fb)
    return points[:MAX_POINTS]
