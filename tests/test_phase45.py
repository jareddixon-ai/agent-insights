"""Phase 4.5 tests: California tax values (Prop 13), sturdier shrinkage, bias allowance, context-only parts."""

import numpy as np
import pandas as pd

import generate_fake_data as gen
import sale_history_test
import score_agents
from agent_model import fit_agent_effects
from shrinkage import shrink, spread_between_agents


# ---- Prop 13 fake data ----
def test_prop13_assessed_value_follows_the_rule():
    t = gen.generate(quiet=True)
    s = t["sales"]
    years = (pd.to_datetime(s["list_date"]) - pd.to_datetime(s["prior_sale_date"])).dt.days / 365.25
    expected = s["prior_sale_price"] * 1.02 ** np.floor(years)
    assert ((s["assessed_value"] / expected - 1).abs() < 0.01).all()          # last sale price, up 2% a year
    ratio = s["assessed_value"] / t["homes"]["true_value"].to_numpy()
    assert ratio.median() < 0.85 and ratio[years >= 20].median() < 0.6        # long-held homes lag far behind


def test_market_style_keeps_phase4_data_and_other_columns_match():
    ca, old = gen.generate(quiet=True), gen.generate(assessed_style="market", quiet=True)
    assert "prior_sale_date" not in old["sales"] and "prior_sale_date" in ca["sales"]
    same = [c for c in old["sales"].columns if c != "assessed_value"]
    pd.testing.assert_frame_equal(ca["sales"][same], old["sales"][same])        # only the tax records change
    pd.testing.assert_frame_equal(ca["offers"], old["offers"])


def test_sale_history_removes_estate_specialist_bias():
    out = sale_history_test.run(seeds=[42])
    without, with_history = out.iloc[1], out.iloc[2]
    assert without["estate_specialist_bias_pts"] > 1.0                       # flattered when the model can't see it
    assert abs(with_history["estate_specialist_bias_pts"]) < 0.5


# ---- Sturdier spread estimate ----
def test_spread_between_agents_recovers_a_known_spread():
    rng = np.random.default_rng(0)
    deals = rng.integers(1, 80, 400)
    noise_var = pd.Series(0.08**2 / deals)
    effect = pd.Series(rng.normal(0, 0.01, 400) + rng.normal(0, np.sqrt(noise_var)))
    assert 0.007 < np.sqrt(spread_between_agents(effect, noise_var)) < 0.013


def test_buyer_spread_no_longer_collapses_to_zero():
    t = gen.generate(seed=42, selection_strength=0.0, quiet=True)        # the world that broke in Phase 4
    effects, _, resid_sd = fit_agent_effects(t["sales"])
    buyer = shrink(effects["buyer_effect"], effects["buyer_deals"], resid_sd)
    assert buyer["weight"].median() > 0.1


# ---- Bias allowance and context-only parts ----
def test_bias_allowance_widens_every_range():
    cfg = {"range": "formula", "bias_allowance": 0.008}
    t = pd.DataFrame({"score": [0.01, 0.0], "formula_low": [0.009, -0.03], "formula_high": [0.011, 0.03],
                      "boot_low": [np.nan] * 2, "boot_high": [np.nan] * 2})
    low, high = score_agents.table_range(t, cfg)
    assert (t["score"] - low >= 1.96 * 0.008 - 1e-9).all() and (high - t["score"] >= 1.96 * 0.008 - 1e-9).all()
    assert high[1] - t["score"][1] < 0.035                              # a wide range grows only a little (3.0 to 3.4 pts)


def test_weight_zero_parts_are_context_only(scored_dir):
    rankings = pd.read_csv(scored_dir / "rankings.csv", index_col="agent_id")
    assert set(rankings["buyer_group"]) <= {"Context only", "No data"}
    assert set(rankings["offer_win_group"]) <= {"Context only", "No data"}
    assert "Context only" not in set(rankings["seller_group"])
