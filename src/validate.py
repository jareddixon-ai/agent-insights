"""Phase 4: validation. Does the model find the skill we hid in the fake data, and where does it fail?

The answer key (true_skills.csv, true_home_values.csv) only exists for fake data, so this
is a fake-data-only check. A real brokerage can never run it, which is why we do it here:
it is the evidence that the method works before anyone trusts it on real data.

Charts go to docs/ (committed, shown in the README). Numbers go to data/fake/.
  4-0  validation.png          estimated vs true skill, seller and buyer, with 95% ranges
  4-1  accuracy_by_deals.png   error with shrinkage on vs off, and range coverage, by number of deals
  4-2  stress_test.png         naive vs model when selection bias points up, nowhere, or down
  4-3  hidden_quality.png      why the buyer side fails: score error follows home value the model can't see

Every score is shown as "higher = better for the client": seller = % sold above what the
home predicts, buyer = % bought below what the home predicts. 0 = the average agent.

Phase 4.5: the fake data now uses California (Prop 13) assessed values, and the table's
ranges include a bias allowance. "Used" below means exactly the range the coaching table uses.

Run order: the full pipeline (see README), then this file (about 15 seconds).
Needs bootstrap_scores.csv and stress_test.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from agent_model import DATA, build_features
from score_agents import load_config, table_range

DOCS = Path(__file__).resolve().parent.parent / "docs"
BUCKETS = [0, 4, 9, 19, 49, np.inf]            # number-of-deals groups
BUCKET_LABELS = ["1-4", "5-9", "10-19", "20-49", "50+"]
RANGES = {"formula": "Formula range", "boot": "Bootstrap range", "used": "Used in the table (+ bias allowance)"}

# Colors: a tested colorblind-safe set, plus quiet gray ink for text, grid and axes.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,   # ships with matplotlib: same charts on every computer
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 9,
})


# ---------- Build the comparison tables ----------

def side_table(boot, truth, side, cfg=None):
    """One row per agent: shrunk score, raw score, three 95% ranges, and the TRUE skill.
    truth is in price units (seller: + is good, buyer: - is good); we flip the buyer side
    so both read 'higher = better for the client'."""
    t = boot.xs(side, level="side").copy()
    t["boot_low"] = t["boot_low"].fillna(t["formula_low"])
    t["boot_high"] = t["boot_high"].fillna(t["formula_high"])
    t["used_low"], t["used_high"] = table_range(t, cfg or load_config())
    t["true"] = (truth - truth.mean()).reindex(t.index)   # centered: 0 = average agent
    if side == "buyer":
        for col in ["raw", "score", "true"]:
            t[col] = -t[col]
        for m in RANGES:
            t[f"{m}_low"], t[f"{m}_high"] = -t[f"{m}_high"], -t[f"{m}_low"]
    t["bucket"] = pd.cut(t["deals"], BUCKETS, labels=BUCKET_LABELS)
    return t


def caught(t, method):
    """True where the 95% range contains the true skill. Should happen about 95% of the time."""
    return (t["true"] >= t[f"{method}_low"]) & (t["true"] <= t[f"{method}_high"])


def summarize(t):
    """Headline numbers for one side."""
    rank = lambda col: t[col].rank().corr(t["true"].rank())   # Spearman: does the ORDER match?
    mae = lambda col: (t[col] - t["true"]).abs().mean() * 100
    out = {
        "agents": len(t),
        "correlation": t["score"].corr(t["true"]),           # Pearson: do the VALUES line up?
        "rank_agreement_shrunk": rank("score"),
        "rank_agreement_raw": rank("raw"),
        "error_pts_shrunk": mae("score"),
        "error_pts_raw": mae("raw"),
        "called_clearly_different": called(t).sum(),
        # Of the agents called clearly above or below average, how many are on the right side of 0?
        "clear_calls_right": (np.sign(t["score"]) == np.sign(t["true"]))[called(t)].mean(),
    }
    for m in RANGES:
        out[f"coverage_{m}"] = caught(t, m).mean()
    return out


def called(t):
    """True where the range the table uses excludes 0: a 'clearly above/below average' call."""
    return (t["used_low"] > 0) | (t["used_high"] < 0)


def by_deals(t):
    """Accuracy and range coverage for agents with few vs many deals."""
    g = t.assign(err_raw=(t["raw"] - t["true"]).abs() * 100, err_shrunk=(t["score"] - t["true"]).abs() * 100,
                 called=called(t),
                 **{f"cov_{m}": caught(t, m) for m in RANGES})
    return g.groupby("bucket", observed=False).agg(
        agents=("true", "size"), error_pts_raw=("err_raw", "mean"), error_pts_shrunk=("err_shrunk", "mean"),
        coverage_formula=("cov_formula", "mean"), coverage_boot=("cov_boot", "mean"),
        coverage_used=("cov_used", "mean"), share_called_clearly=("called", "mean"))


def hidden_value(sales, true_values):
    """How much value each home has that the model CAN'T see (fake data only).
    Fit the model's own home features to the TRUE value; what's left over is hidden quality
    (view, condition, finishes beyond what the assessed value catches). In % (log points)."""
    y = np.log(true_values.set_index("listing_id")["true_value"].reindex(sales["listing_id"]).to_numpy())
    X = build_features(sales)
    return pd.Series(y - LinearRegression().fit(X, y).predict(X), index=sales.index)


def coverage_across_worlds(seeds=(42, 43, 44, 45, 46), allowance=None):
    """Is the busy-agent coverage drop real or one unlucky draw? Rebuild the fake data in 5 worlds,
    with and without selection bias, and check how often the formula range catches the truth.
    If coverage falls for busy agents ONLY when selection bias is on, the ranges are missing a bias:
    more deals shrink the range, but they don't shrink a push from hidden home quality.
    The second row per side adds the bias allowance from config/scoring.json (the 4.5 fix)."""
    import generate_fake_data as gen
    from agent_model import fit_agent_effects
    from shrinkage import shrink
    allowance = load_config().get("bias_allowance", 0) if allowance is None else allowance
    rows = []
    for seed in seeds:
        for strength, world in [(0.4, "Selection bias on (standard)"), (0.0, "Homes assigned at random")]:
            fake = gen.generate(seed=seed, selection_strength=strength, quiet=True)
            effects, _, resid_sd = fit_agent_effects(fake["sales"])
            skills = fake["skills"].set_index("agent_id")
            for side, col, n, truth in [("seller", "seller_effect", "listings", skills["seller_premium"]),
                                        ("buyer", "buyer_effect", "buyer_deals", -skills["buyer_savings"])]:
                tab = shrink(effects[col], effects[n], resid_sd)
                tr = (truth - truth.mean()).reindex(tab.index)
                size = pd.cut(tab["deals"], [0, 19, 49, np.inf], labels=["under 20 deals", "20-49", "50+"])
                half = (tab["high"] - tab["low"]) / 2
                wide = np.sqrt(half**2 + (1.96 * allowance) ** 2)
                for rng_name, lo, hi in [("formula", tab["low"], tab["high"]),
                                         ("formula + bias allowance", tab["score"] - wide, tab["score"] + wide)]:
                    hit = (tr >= lo) & (tr <= hi)
                    for bucket, share in hit.groupby(size, observed=False).mean().items():
                        rows.append({"world": world, "side": side, "range": rng_name, "deals": bucket, "caught": share})
    keys = ["world", "side", "range", "deals"]
    out = pd.DataFrame(rows).groupby(keys, observed=False, sort=False)["caught"].mean()
    return out.unstack("deals")


def time_split_check(seeds=(42, 43, 44, 45, 46)):
    """How would a real brokerage check accuracy with NO answer key? Score every agent twice:
    once on the first half of the sales (by date) and once on the second half. If scores are
    mostly skill, the two halves agree. The agreement (r) tells you how reliable the scores are:
      reliability of the full record = 2r / (1 + r)   (Spearman-Brown: twice the data, less noise)
      implied correlation with true skill = square root of that reliability
    Here we CAN compare that guess with the answer key. The catch: a bias that repeats in both
    halves (hidden home quality) looks just like skill, so the check can be fooled.
    Reliability (consistent) is not the same as validity (correct)."""
    import generate_fake_data as gen
    from agent_model import fit_agent_effects
    rows = []
    for seed in seeds:
        for strength, world in [(0.4, "Selection bias on (standard)"), (0.0, "Homes assigned at random")]:
            fake = gen.generate(seed=seed, selection_strength=strength, quiet=True)
            sales, skills = fake["sales"], fake["skills"].set_index("agent_id")
            sold = pd.to_datetime(sales["sold_date"])
            early = sold <= sold.median()
            full, _, _ = fit_agent_effects(sales)
            first, _, _ = fit_agent_effects(sales[early].reset_index(drop=True))
            second, _, _ = fit_agent_effects(sales[~early].reset_index(drop=True))
            for side, col, truth in [("seller", "seller_effect", skills["seller_premium"]),
                                     ("buyer", "buyer_effect", -skills["buyer_savings"])]:
                halves = pd.concat([first[col], second[col]], axis=1).dropna()
                r = halves.iloc[:, 0].corr(halves.iloc[:, 1])
                reliability = 2 * r / (1 + r)
                est = full[col].dropna()
                rows.append({"world": world, "side": side, "halves_agree": r, "reliability": reliability,
                             "implied_accuracy": np.sqrt(max(reliability, 0)),
                             "actual_accuracy": est.corr((truth - truth.mean()).reindex(est.index))})
    return pd.DataFrame(rows).groupby(["world", "side"], sort=False).mean()


# ---------- Charts ----------

def pct_axis(ax, which="both"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f}%" if v else "0%")
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def chart_validation(tables, stats, path):
    """4-0: each dot is an agent. On the dashed line = estimate matches the truth exactly."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (side, t) in zip(axes, tables.items()):
        hit = caught(t, "used")
        for mask, color, label in [(hit, BLUE, f"Range caught the truth ({hit.sum()})"),
                                   (~hit, ORANGE, f"Range missed ({(~hit).sum()})")]:
            d = t[mask]
            ax.vlines(d["true"], d["used_low"], d["used_high"], color=color, alpha=0.30, linewidth=1.2)
            ax.scatter(d["true"], d["score"], s=22, color=color, edgecolor=SURFACE, linewidth=0.6,
                       label=label, zorder=3)
        lim = max(t["true"].abs().max(), t["score"].abs().max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], ls="--", color=INK2, lw=1, zorder=2, label="Perfect estimate")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(min(-lim, t["used_low"].min()), max(lim, t["used_high"].max()))
        pct_axis(ax)
        s = stats[side]
        name = "Seller side: sold above expected" if side == "seller" else "Buyer side: bought below expected"
        ax.set_title(name, pad=22)
        ax.text(0, 1.015, f"Rank agreement {s['rank_agreement_shrunk']:.2f}  |  correlation "
                f"{s['correlation']:.2f}  |  {len(t)} agents", transform=ax.transAxes, color=INK2,
                fontsize=9, va="bottom")
        ax.set_xlabel("True skill (hidden answer key)")
        ax.set_ylabel("Estimated skill, with 95% range")
        ax.legend(loc="upper left")
    fig.suptitle("Estimated vs true agent skill (fake data)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", y=1.03)
    fig.text(0.01, -0.03, "Shrinkage pulls thin records toward 0 on purpose, so the dots sit flatter than the "
             "dashed line. Ranges are the ones the table uses: the wider of the formula and bootstrap ranges,\n"
             "plus the bias allowance. California (Prop 13) assessed values.", color=INK2, fontsize=8.5)
    save(fig, path)


def chart_by_deals(tables_by_deals, path):
    """4-1: top row = error with shrinkage off vs on; bottom row = how often the 95% range caught the truth."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    top = max(g["error_pts_raw"].max() for g in tables_by_deals.values()) * 1.08
    x = np.arange(len(BUCKET_LABELS))
    for col, (side, g) in enumerate(tables_by_deals.items()):
        ax = axes[0, col]
        w = 0.38
        ax.bar(x - w / 2 - 0.01, g["error_pts_raw"], w, color=ORANGE, label="No shrinkage")
        ax.bar(x + w / 2 + 0.01, g["error_pts_shrunk"], w, color=BLUE, label="With shrinkage")
        ax.set_ylim(0, top)   # same scale on both sides so they compare fairly
        ax.set_title(f"{side.capitalize()} side: average miss")
        ax.set_ylabel("Average miss (percentage points)")
        ax.legend(loc="upper right")
        ax = axes[1, col]
        ax.axhline(0.95, ls="--", color=INK2, lw=1)
        ax.text(len(x) - 0.6, 0.955, "95% target", color=INK2, fontsize=8.5, ha="right", va="bottom")
        for m, color in zip(RANGES, [BLUE, ORANGE, AQUA]):
            ax.plot(x, g[f"coverage_{m}"], marker="o", ms=6, lw=2, color=color, label=RANGES[m])
        ax.set_ylim(0.3, 1.04)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{side.capitalize()} side: did the 95% range catch the truth?")
        ax.set_ylabel("Share of agents caught")
        ax.set_xticks(x, [f"{b}\n({n} agents)" for b, n in zip(BUCKET_LABELS, g["agents"])])
        ax.set_xlabel("Number of deals")
        ax.legend(loc="lower right")
    fig.suptitle("Accuracy by number of deals (fake data)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, path)


def chart_stress(stress, path):
    """4-2: rank agreement with true skill (1 = same order, 0 = unrelated, -1 = reversed)."""
    worlds = ["Better agents get\npricier homes\n(standard)", "Homes assigned\nat random",
              "Better agents get\ncheaper homes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [3, 4]})
    x = np.arange(3)
    w = 0.38
    for off, col, color, label in [(-w / 2 - 0.01, "naive_seller", ORANGE, "Naive: average sold price"),
                                   (w / 2 + 0.01, "model_seller", BLUE, "Model")]:
        vals = stress[col].iloc[:3].to_numpy()
        a1.bar(x + off, vals, w, color=color, label=label)
        for xi, v in zip(x + off, vals):
            a1.text(xi, v + (0.03 if v >= 0 else -0.03), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK2)
    a1.axhline(0, color=AXIS, lw=1)
    a1.set_ylim(-1, 1.05)
    a1.set_xticks(x, worlds, fontsize=8.5)
    a1.set_ylabel("Rank agreement with true skill")
    a1.set_title("Seller side: naive vs model")
    a1.legend(loc="lower left")

    labels = worlds + ["Standard world,\nmodel sees TRUE\nhome value"]
    vals = stress["model_buyer"].to_numpy()
    xb = np.arange(len(vals))
    a2.bar(xb, vals, 0.55, color=BLUE)
    for xi, v in zip(xb, vals):
        a2.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    a2.axvline(2.5, color=AXIS, lw=1, ls=":")
    a2.set_ylim(0, 1.05)
    a2.set_xticks(xb, labels, fontsize=8.5)
    a2.set_title("Buyer side: model only")
    fig.suptitle("Stress test: change who gets which homes (average of 5 fake worlds)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, path)


def chart_hidden(tables, hidden_by_agent, path):
    """4-3: each dot is an agent. x = hidden home value in their deals; y = how far their raw score missed."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (side, t) in zip(axes, tables.items()):
        h = hidden_by_agent[side].reindex(t.index)
        err = t["raw"] - t["true"]
        ax.axhline(0, color=AXIS, lw=1)
        ax.axvline(0, color=AXIS, lw=1)
        ax.scatter(h, err, s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.set_title(f"{side.capitalize()} side", pad=22)
        ax.text(0, 1.015, f"correlation {err.corr(h):+.2f}", transform=ax.transAxes, color=INK2, fontsize=9,
                va="bottom")
        ax.set_xlabel("Hidden value in the agent's homes (average)")
        pct_axis(ax)
    axes[0].set_ylabel("Score miss (estimate minus truth)")
    fig.suptitle("Most of each agent's miss is home quality the model can't see (fake data)", x=0.01,
                 ha="left", fontsize=13, fontweight="bold", y=1.03)
    skill_link = np.mean([hidden_by_agent[s].reindex(t.index).corr(t["true"]) for s, t in tables.items()])
    fig.text(0.01, -0.12, "Raw scores, before shrinkage. Hidden value = what the home was truly worth beyond what "
             "the model predicts from its features and assessed value.\nSeller side: hidden value inflates the "
             "score. Buyer side: it makes the agent look like they paid more. Much of this is luck, which the ranges\n"
             "allow for. The problem is the part that isn't luck: good agents get homes with more hidden value "
             f"(correlation about {skill_link:+.1f} with true skill).", color=INK2, fontsize=8.5)
    save(fig, path)


# ---------- Run ----------

def main(data=DATA, docs=DOCS, seeds=(42, 43, 44, 45, 46)):
    data, docs = Path(data), Path(docs)
    docs.mkdir(exist_ok=True)
    boot = pd.read_csv(data / "bootstrap_scores.csv", index_col=["side", "agent_id"])
    skills = pd.read_csv(data / "true_skills.csv").set_index("agent_id")
    sales = pd.read_csv(data / "sales.csv", keep_default_na=False)
    stress = pd.read_csv(data / "stress_test.csv", index_col="scenario")

    tables = {"seller": side_table(boot, skills["seller_premium"], "seller"),
              "buyer": side_table(boot, -skills["buyer_savings"], "buyer")}
    stats = {side: summarize(t) for side, t in tables.items()}
    deals = {side: by_deals(t) for side, t in tables.items()}

    sales["hidden"] = hidden_value(sales, pd.read_csv(data / "true_home_values.csv"))
    hidden = {"seller": sales.groupby("listing_agent_id")["hidden"].mean(),
              "buyer": sales[sales["buyer_agent_id"] != ""].groupby("buyer_agent_id")["hidden"].mean()}
    for side, t in tables.items():
        h = hidden[side].reindex(t.index)
        stats[side]["miss_vs_hidden_corr"] = (t["raw"] - t["true"]).corr(h)
        stats[side]["hidden_vs_true_skill_corr"] = h.corr(t["true"])   # + = flatters good agents

    chart_validation(tables, stats, docs / "validation.png")
    chart_by_deals(deals, docs / "accuracy_by_deals.png")
    chart_stress(stress, docs / "stress_test.png")
    chart_hidden(tables, hidden, docs / "hidden_quality.png")

    worlds = coverage_across_worlds(seeds)
    worlds.round(3).to_csv(data / "coverage_across_worlds.csv")
    split = time_split_check(seeds)
    split.round(3).to_csv(data / "time_split_check.csv")

    summary = pd.DataFrame(stats).T
    summary.round(3).to_csv(data / "validation_summary.csv", index_label="side")
    pd.concat(deals, names=["side", "deals"]).round(3).to_csv(data / "validation_by_deals.csv")

    pd.set_option("display.width", 140)
    print("VALIDATION vs ANSWER KEY (fake data only)")
    print(summary.round(2).to_string())
    for side, g in deals.items():
        print(f"\n{side.upper()} side by number of deals")
        print(g.round(2).to_string())
    print("\nRANGE COVERAGE ACROSS 5 FAKE WORLDS (should be about 0.95)")
    print(worlds.round(2).to_string())
    print("\nNO-ANSWER-KEY CHECK: first half vs second half of the sales (raw scores, 5 fake worlds)")
    print(split.round(2).to_string())
    print(f"\nSaved 4 charts in {docs.name}/ and validation_summary.csv, validation_by_deals.csv,"
          " coverage_across_worlds.csv, time_split_check.csv in data/fake/")
    return stats, deals, worlds, split


if __name__ == "__main__":
    main()
