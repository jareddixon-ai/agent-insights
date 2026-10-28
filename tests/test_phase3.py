"""Phase 3 tests: stress test, price model, speed and pricing, bootstrap, coaching index."""

import numpy as np
import pandas as pd

import generate_fake_data as gen
import price_model
import stress_test


def read(data_dir, name, **kw):
    return pd.read_csv(data_dir / name, keep_default_na=False, **kw)


# ---- 3-0 Stress test ----
def test_naive_collapses_but_model_holds_when_bias_flips():
    r = stress_test.run_one(seed=42, strength=-0.4, oracle=False)
    assert r["naive_seller"] < 0          # average price now points the WRONG way
    # Phase 4.5: with California tax values the model weakens too (about 0.3), but it still
    # points the right way and beats average price by a wide margin.
    assert r["model_seller"] > 0.15 and r["model_seller"] > r["naive_seller"] + 0.5


def test_generator_settings_change_only_what_they_should():
    a = gen.generate(quiet=True)
    b = gen.generate(selection_strength=0.0, quiet=True)
    pd.testing.assert_frame_equal(a["skills"], b["skills"])   # same agents and skills
    assert not a["sales"]["listing_agent_id"].equals(b["sales"]["listing_agent_id"])


# ---- 3-1 Price model ----
def test_price_model_is_honest_and_accurate(data_dir):
    pred = price_model.cross_validated_predictions(read(data_dir, "sales.csv"))
    assert len(pred) == 5000 and pred["predicted_price"].gt(0).all()
    assert price_model.error_summary(pred)["median_error"] < 0.10


# ---- 3-2 Speed and pricing ----
def test_speed_and_pricing_describe_style_not_skill(data_dir):
    import speed_and_pricing
    sales, skills = read(data_dir, "sales.csv"), read(data_dir, "true_skills.csv").set_index("agent_id")
    t = speed_and_pricing.speed_and_pricing(sales, price_model.cross_validated_predictions(sales))
    assert t["list_vs_value"].corr(t["speed_vs_similar"], method="spearman") > 0.3   # list high -> slow
    miss_vs_skill = t["typical_list_miss"].corr(skills["seller_premium"].reindex(t.index), method="spearman")
    assert abs(miss_vs_skill) < 0.3   # pricing "accuracy" is not a skill measure


# ---- 3-3 Bootstrap ----
def test_bootstrap_ranges_and_small_sample_flag(data_dir):
    import bootstrap_ranges as br
    sales = read(data_dir, "sales.csv")
    full, _ = br.shrunk_scores(sales)
    draws, _ = br.bootstrap(sales, n_resamples=20)          # a few resamples keep the test fast
    t = br.summarize(full, draws)
    busy = t[t["deals"] >= 20]
    assert (busy["boot_high"] > busy["boot_low"]).all()
    assert t["under_5_deals"].equals(t["deals"] < 5)
    assert np.allclose(t["fixed_k_score"], t["deals"] / (t["deals"] + 10) * t["raw"])


# ---- 3-4 Coaching index ----
def test_coaching_table_is_honest(data_dir):
    import bootstrap_ranges as br
    import offer_metrics
    import score_agents
    import speed_and_pricing
    sales = read(data_dir, "sales.csv")
    full_s, full_b = br.shrunk_scores(sales)
    ds, db = br.bootstrap(sales, n_resamples=20)
    boot = pd.concat({"seller": br.summarize(full_s, ds), "buyer": br.summarize(full_b, db)},
                     names=["side", "agent_id"])
    pred = price_model.cross_validated_predictions(sales)
    offer = offer_metrics.offer_table(sales, read(data_dir, "offers.csv"), pred, penalty=0.25)
    speed = speed_and_pricing.speed_and_pricing(sales, pred)
    cfg = score_agents.load_config()
    t = score_agents.build_table(boot, offer, speed, read(data_dir, "agents.csv"), cfg)

    assert "Not clearly different from average" in set(t["index_group"])   # the honest middle exists
    t = t[t["coaching_index"].notna()]                                       # agents with no listings have no index
    assert (t["index_low"] <= t["coaching_index"]).all() and (t["coaching_index"] <= t["index_high"]).all()
    clear_above = t["index_group"] == "Clearly above average"
    assert (t.loc[clear_above, "index_low"] > 0).all()
    skills = read(data_dir, "true_skills.csv").set_index("agent_id").reindex(t.index)
    assert t["coaching_index"].rank().corr(skills["seller_premium"].rank()) > 0.4
