"""Phase 5 follow-up (backlog item 2): is the offer score's lean toward thrifty clients real,
or a side effect of how the fake losing offers are made?

The lean: agents whose clients pay less (higher buyer savings) get slightly HIGHER fair offer
scores (rank agreement +0.10 to +0.16), even though thrift and win skill are unrelated.

Suspect: in the standard fake data the winner and sold price come first, and losing offers
are drawn afterwards to sit around the winner's price (make_offers). Each loser's price gap
includes the WINNER's buyer savings, so when a thrifty agent wins, the rivals look like they
bid more. The contest model sees "won while cheaper" and credits it to the agent.

Test: a side world with full contests. Same listings, same bidders (same agents, same terms),
but every bidder's price is drawn the same way (their agent's thrift + spread), and the
winner is whoever has the strongest offer (price + terms + win skill + seller's taste). The
seller's-taste noise is sized only to match the standard data's "highest price wins" share
(set before looking at the lean). If the lean disappears here, it comes from the fake data,
not from the score.

Result (Oct 28, seeds 42-46, 3 kinds of world): the lean is 0.10 / 0.12 / 0.16 with the
standard fake offers and 0.01 / 0.00 / 0.01 with full contests. Ranges catch the truth
94-96% in full contests with no widening (87-88% standard). Win rank 0.84-0.87 (0.89
standard): full contests have more luck in who wins, so fewer agents are clear calls.
So both the lean and the narrow ranges come from how the fake losers are made.
"""

import numpy as np
import pandas as pd

import generate_fake_data as gen
import offer_metrics as om
from agent_model import DATA
from offer_calibrate import CACHE, WORLDS, grade

SEEDS = [42, 43, 44, 45, 46]
DOCS = DATA.parent.parent / "docs"
TASTE = [0.0, 0.003, 0.006, 0.01, 0.015]   # seller's taste noise (Gumbel scale, share of price) to pick from


def observed_terms(o):
    """Strength from the terms a seller can see (the generator's made-up term values)."""
    v, loan = gen.TERM_VALUE, (o["financing"] == "loan").to_numpy()
    return (v["cash"] * ~loan + v["per_10_pts_down"] * np.where(loan, (o["down_payment_pct"] - 20) / 10, 0)
            + v["inspection"] * o["inspection_contingency"] + v["appraisal"] * o["appraisal_contingency"]
            + v["loan"] * o["loan_contingency"] + v["per_day_over_21"] * (o["days_to_close"] - 21)).to_numpy()


def highest_price_wins(o):
    top = o.groupby("listing_id")["offer_price"].transform("max") == o["offer_price"]
    multi = o["competing_offers"] > 1
    return (top & (o["outcome"] == "won"))[multi].sum() / o.loc[multi, "listing_id"].nunique()


def full_contests(offers, sales, skills, seed, taste):
    """Same bidders and terms; prices redrawn; the strongest offer wins."""
    rng = np.random.default_rng([seed, 9])
    sk = skills.set_index("agent_id")
    agent = offers["buyer_agent_id"].replace("", np.nan)
    save = sk["buyer_savings"].reindex(agent).fillna(0).to_numpy()
    win = sk["win_skill"].reindex(agent).fillna(0).to_numpy()
    ref = sales.set_index("listing_id")["sold_price"].reindex(offers["listing_id"]).to_numpy()
    gap = -save + rng.normal(0.0, gen.BID_SPREAD, len(offers))
    strength = (gap + observed_terms(offers) + rng.normal(0.0, 0.006, len(offers))   # 0.006: letters etc., as in the generator
                + gen.WIN_SKILL_VALUE * win + rng.gumbel(0.0, taste, len(offers)))   # taste: seller's own pick
    o = offers.assign(offer_price=(ref * np.exp(gap)).round(-3), strength=strength)
    best = o.groupby("listing_id")["strength"].transform("max")
    o["outcome"] = np.where(o["strength"] == best, "won", "lost")
    return o.drop(columns="strength")


def pick_taste(seed=42, strength=0.4):
    t = gen.generate(seed=seed, selection_strength=strength, quiet=True)
    target = highest_price_wins(t["offers"])
    got = {s: highest_price_wins(full_contests(t["offers"], t["sales"], t["skills"], seed, s)) for s in TASTE}
    return min(got, key=lambda s: abs(got[s] - target)), target, got


def run(seeds=SEEDS, worlds=WORLDS, taste=None):
    from offer_calibrate import agent_rows
    if taste is None:
        taste, _, _ = pick_taste()
    rows, shares = [], []
    for strength, world in worlds.items():
        for seed in seeds:
            t = gen.generate(seed=seed, selection_strength=strength, quiet=True)
            full = full_contests(t["offers"], t["sales"], t["skills"], seed, taste)
            for version, offers in [("standard", t["offers"]), ("full contests", full)]:
                name = f"{seed}_{world}.csv" if version == "standard" else f"full_{taste}_{seed}_{world}.csv"
                cached = CACHE / name   # standard worlds were already fitted by offer_calibrate
                if not cached.exists():
                    agent_rows(seed, strength, offers, t["skills"]).reset_index(names="agent_id").to_csv(cached, index=False)
                fitted = pd.read_csv(cached, index_col="agent_id")
                rows.append(fitted.assign(version=version))
                shares.append({"version": version, "world": world, "seed": seed,
                               "highest_price_wins": highest_price_wins(offers)})
            print(f"  seed {seed}, {world} world done", flush=True)
    d = pd.concat(rows).reset_index(names="agent_id")
    graded = pd.concat({v: grade(g) for v, g in d.groupby("version", sort=False)}, names=["version"])
    extra = pd.DataFrame(shares).groupby(["version", "world"], sort=False)[["highest_price_wins"]].mean()
    return graded.join(extra), taste


def chart(cal, lean, path):
    """Two panels: does widening fix coverage, and does the lean survive full contests?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from validate import AQUA, BLUE, INK2, ORANGE

    worlds = list(WORLDS.values())
    x, w = np.arange(3), 0.26
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    bars = [("Standard offers,\nfresh seeds, before", cal.loc["confirm_before"], "#b9b8ad"),
            ("Standard offers,\nfresh seeds, widened", cal.loc["confirm_after"], BLUE),
            ("Full contests, not widened", lean.loc["full contests"], AQUA)]
    for i, (label, t, color) in enumerate(bars):
        v = t.reindex(worlds)["coverage"].to_numpy() * 100
        a1.bar(x + (i - 1) * w, v, w, color=color, label=label.replace("\n", " "))
        for xi, vi in zip(x + (i - 1) * w, v):
            a1.text(xi, vi + 0.3, f"{vi:.1f}", ha="center", fontsize=7.5, color=INK2)
    a1.axhline(90, color=ORANGE, lw=1.2, ls="--")
    a1.text(1.5, 88.9, "rule:\n90%", color=ORANGE, fontsize=8, ha="center", va="top")
    a1.set_ylim(80, 100)
    a1.set_xticks(x, [f"good agents get\n{k} homes" if k != "random" else "homes at\nrandom" for k in worlds], fontsize=8.5)
    a1.set_ylabel("% of 95% ranges that catch the truth")
    a1.set_title("Offer score ranges")
    a1.legend(loc="upper left", fontsize=7.5)

    for i, (label, t, color) in enumerate([("Standard offers (fresh seeds)", cal.loc["confirm_after"], BLUE),
                                           ("Full contests", lean.loc["full contests"], AQUA)]):
        v = t.reindex(worlds)["savings_rank"].to_numpy()
        a2.bar(x + (i - 0.5) * w, v, w, color=color, label=label)
        for xi, vi in zip(x + (i - 0.5) * w, v):
            a2.text(xi, vi + 0.005, f"{vi:+.2f}", ha="center", fontsize=8, color=INK2)
    a2.axhspan(-0.15, 0.15, color="#c3c2b7", alpha=0.25, lw=0)
    a2.text(0.5, 0.138, "allowed: within 0.15", color=INK2, fontsize=8, ha="center")
    a2.axhline(0, color="#c3c2b7", lw=1)
    a2.set_ylim(-0.05, 0.25)
    a2.set_xticks(x, [f"good agents get\n{k} homes" if k != "random" else "homes at\nrandom" for k in worlds], fontsize=8.5)
    a2.set_ylabel("Rank agreement with buyer savings")
    a2.set_title("Lean toward clients who pay less (want 0)")
    a2.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Offer score follow-up: widening the ranges, and where the lean comes from",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main():
    pd.set_option("display.width", 200)
    taste, target, got = pick_taste()
    print(f"Standard data: highest price wins {target:.0%} of contests. Full contests by taste noise:",
          {k: round(v, 3) for k, v in got.items()}, "-> chosen", taste)
    out, _ = run(taste=taste)
    print("\nOFFER SCORE: standard fake offers vs full contests (average of 5 seeds per kind of world)")
    print(out.round(3).to_string())
    out.round(4).to_csv(DATA / "offer_lean_test.csv")
    cal = pd.read_csv(DATA / "offer_calibration.csv", index_col=["step", "world"])   # from offer_calibrate.py
    DOCS.mkdir(exist_ok=True)
    chart(cal, out.round(4), DOCS / "offer_followup.png")   # same numbers as the saved CSV
    print("Saved offer_lean_test.csv and docs/offer_followup.png")


if __name__ == "__main__":
    main()
