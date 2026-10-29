"""Phase 5 follow-up tests: offer-score range widening and the full-contest side world."""

import numpy as np
import pandas as pd
import pytest

import generate_fake_data as gen
import offer_calibrate as oc
import offer_lean_test as lt
import offer_metrics as om


@pytest.fixture(scope="module")
def world():
    return gen.generate(quiet=True)


def test_allowance_widens_narrow_ranges_most():
    se = pd.Series([0.1, 1.0])
    wide = om.widened_se(se, per_pct=1.0, setting={"form": "allowance", "value": 0.3})
    assert (wide >= se).all() and wide[0] / se[0] > wide[1] / se[1]   # busy agents (small error) grow most
    same = om.widened_se(se, per_pct=1.0, setting={"form": "multiplier", "value": 1.0})
    pd.testing.assert_series_equal(same, se)


def test_offer_score_uses_the_configured_widening(world):
    plain = om.fair_offer_scores(world["offers"], penalty=0.25, widen={"form": "multiplier", "value": 1.0})
    wide = om.fair_offer_scores(world["offers"], penalty=0.25)          # reads offer_range from scoring.json
    assert om.offer_range_setting()["form"] in {"multiplier", "allowance"}
    pd.testing.assert_series_equal(plain["score"], wide["score"])      # widening never moves the score
    assert ((wide["high"] - wide["low"]) >= (plain["high"] - plain["low"]) - 1e-12).all()


def test_calibration_picks_smallest_setting_that_reaches_coverage():
    rng = np.random.default_rng(0)
    rows = []
    for world in oc.WORLDS.values():
        se = rng.uniform(0.1, 0.5, 2000)
        truth = rng.normal(0, 1, 2000)
        rows.append(pd.DataFrame({"seed": 1, "world": world, "se": se, "truth": truth,
                                  "est": truth + rng.normal(0, 1.3 * se),   # errors 30% bigger than claimed
                                  "score": truth, "win_skill": truth, "buyer_savings": rng.normal(0, 1, 2000)}))
    form, value, best = oc.calibrate(pd.concat(rows))
    # 90% coverage (the rule) with errors 1.3x too big needs 1.645 x 1.3 / 1.96 = about 1.09
    assert 1.05 <= best["multiplier"][0] <= 1.15
    assert (oc.grade(pd.concat(rows), "multiplier", best["multiplier"][0])["coverage"] >= 0.9).all()


def test_full_contests_keep_bidders_and_pick_one_winner(world):
    o = world["offers"]
    full = lt.full_contests(o, world["sales"], world["skills"], 42, taste=0.006)
    assert (full["outcome"] == "won").groupby(full["listing_id"]).sum().eq(1).all()
    same = ["offer_id", "listing_id", "buyer_agent_id", "financing", "down_payment_pct", "days_to_close"]
    pd.testing.assert_frame_equal(full[same], o[same])
    assert 0.7 < lt.highest_price_wins(full) < 0.9                      # sized to look like the standard data
