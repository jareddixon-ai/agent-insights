"""Phase 5 tests: condition data what-if, offers that work like real ones, fair offer metrics,
offer PDFs and the extractor that drops names."""

import numpy as np
import pandas as pd
import pytest

import condition_test
import extract_offers
import generate_fake_data as gen
import make_offer_pdfs
import offer_metrics as om
from agent_model import fit_agent_effects


@pytest.fixture(scope="module")
def world():
    return gen.generate(quiet=True)


# ---- 5-0 Condition data ----
def test_condition_data_only_adds_a_column(world):
    s = world["sales"]
    rated = gen.add_condition_rating(s, world["homes"], world["skills"], 0.8)
    valued = gen.add_value_estimate(s, world["homes"], 0.03)
    pd.testing.assert_frame_equal(rated.drop(columns="condition_rating"), s)
    pd.testing.assert_frame_equal(valued.drop(columns="value_estimate"), s)
    assert set(rated["condition_rating"]) <= {1, 2, 3, 4, 5}
    assert 0.65 < np.corrcoef(rated["condition_rating"], world["homes"]["quality"])[0, 1] < 0.85
    worth = world["homes"]["true_value"].to_numpy() * gen.market_factor(s["list_date"])
    assert 0.025 < np.std(np.log(valued["value_estimate"] / worth)) < 0.035


def test_value_estimate_fixes_buyer_side_but_leaky_one_hides_differences():
    out = condition_test.run(seeds=[42], worlds={0.4: "standard"},
                             versions=[("none", 0.0, False), ("estimate", 0.03, False), ("estimate", 0.02, True)])
    none, good, leaky = out.iloc[0], out.iloc[1], out.iloc[2]
    assert good["buyer_rank"] > none["buyer_rank"] + 0.3        # an independent estimate rescues the buyer side
    assert leaky["seller_clear"] < good["seller_clear"] / 5     # anchored to the sale price: everyone looks average


def test_listing_agent_rating_soaks_up_skill(world):
    s, sk = world["sales"], world["skills"].set_index("agent_id")
    truth = sk["seller_premium"]
    def seller_rank(sales):
        e = fit_agent_effects(sales)[0]["seller_effect"].dropna()
        return e.rank().corr(truth.reindex(e.index).rank())
    honest = gen.add_condition_rating(s, world["homes"], world["skills"], 0.8)
    puffed = gen.add_condition_rating(s, world["homes"], world["skills"], 0.8, puffery=0.5)
    assert seller_rank(puffed) < seller_rank(honest) - 0.1


# ---- 5-1 Offers ----
def test_offers_have_terms_and_stronger_terms_explain_upsets(world):
    o, s = world["offers"], world["sales"].set_index("listing_id")
    assert set(o["financing"]) == {"cash", "loan"}
    assert (o.loc[o["financing"] == "cash", "down_payment_pct"] == 100).all()
    assert (o.loc[o["financing"] == "cash", "loan_contingency"] == 0).all()
    final, first = s["final_list_price"].reindex(o["listing_id"]).to_numpy(), s["original_list_price"].reindex(o["listing_id"]).to_numpy()
    assert ((o["list_price_at_offer"] >= final - 1000) & (o["list_price_at_offer"] <= first + 1000)).all()
    assert (o.groupby("listing_id")["competing_offers"].transform("size") == o["competing_offers"]).all()
    # When a losing offer had a HIGHER price, the winner usually had stronger terms.
    v, loan = gen.TERM_VALUE, o["financing"] == "loan"
    terms = (v["cash"] * ~loan + v["per_10_pts_down"] * np.where(loan, (o["down_payment_pct"] - 20) / 10, 0)
             + v["inspection"] * o["inspection_contingency"] + v["appraisal"] * o["appraisal_contingency"]
             + v["loan"] * o["loan_contingency"] + v["per_day_over_21"] * (o["days_to_close"] - 21))
    o = o.assign(terms=terms)
    w = o[o["outcome"] == "won"].set_index("listing_id")[["offer_price", "terms"]]
    lost = o[o["outcome"] == "lost"].join(w, on="listing_id", rsuffix="_won")
    upset = lost[lost["offer_price"] > lost["offer_price_won"]]
    assert len(upset) > 100 and (upset["terms_won"] > upset["terms"]).mean() > 0.65


# ---- 5-2 Fair offer metrics ----
def test_fair_offer_score_finds_win_skill_without_rewarding_overpaying(world):
    from offer_win_rates import win_rates
    o, sk = world["offers"], world["skills"].set_index("agent_id")
    fair = om.fair_offer_scores(o, penalty=0.25)
    raw = win_rates(o)["win_rate"]
    r = lambda x, col: x.rank().corr(sk[col].reindex(x.index).rank())
    assert r(fair["score"], "win_skill") > 0.8 and r(raw, "win_skill") < 0.65
    assert r(raw, "buyer_savings") < -0.2                 # raw win rate rewards clients who pay more
    assert abs(r(fair["score"], "buyer_savings")) < 0.2
    values = om.term_values(fair.attrs["terms"])           # the model recovers what terms are worth
    assert 0.7 < values["cash"] < 1.5 and -0.9 < values["inspection"] < -0.3
    assert (fair["low"] <= fair["score"]).all() and (fair["score"] <= fair["high"]).all()


def test_contest_model_penalty_is_picked_by_cross_validation(world):
    d = om.contest_data(world["offers"])
    best, scores = om.pick_penalty(d, penalties=[0.25, 8], folds=3)
    assert best == 0.25 and scores[0.25] > scores[8]      # heavy shrinkage predicts held-out winners worse


def test_offers_attracted_fair_version_tracks_seller_skill(world):
    import price_model
    s, sk = world["sales"], world["skills"].set_index("agent_id")
    t = om.offers_attracted(s, world["offers"], price_model.cross_validated_predictions(s))
    x = t["attract_fair_score"]
    assert x.rank().corr(sk["seller_premium"].reindex(x.index).rank()) > 0.6
    assert (t["attract_fair_low"] <= x).all() and (x <= t["attract_fair_high"]).all()


# ---- 5-3 PDFs and extractor ----
@pytest.fixture(scope="module")
def pdfs(world, tmp_path_factory):
    folder = tmp_path_factory.mktemp("pdfs")
    ids = make_offer_pdfs.make_pdfs(world["offers"], world["sales"], n=8, out_dir=folder)
    return folder, ids


def test_extractor_reads_every_number_back(world, pdfs):
    folder, ids = pdfs
    table, failures = extract_offers.extract_folder(folder)
    assert len(failures) == 0 and sorted(table["offer_id"]) == sorted(ids)
    truth = world["offers"].set_index("offer_id").loc[table["offer_id"], extract_offers.FIELDS[1:]].reset_index()
    assert (table.astype(str).to_numpy() == truth[extract_offers.FIELDS].astype(str).to_numpy()).all()


def test_extractor_stores_no_names_emails_or_phones(pdfs, tmp_path):
    import pdfplumber
    folder, _ = pdfs
    table, _ = extract_offers.extract_folder(folder)
    stored = table.to_csv(index=False).lower()
    for path in folder.glob("*.pdf"):
        text = pdfplumber.open(path).pages[0].extract_text()
        line = next(l for l in text.splitlines() if l.startswith(("Buyer(s):", "Buyer name(s):")))
        for secret in extract_offers.private_bits(text, [line.split(":", 1)[1]]):
            assert secret.lower() not in stored
    assert not set(table.columns) & {"buyer_name", "email", "phone", "note"}


def test_extractor_refuses_a_row_that_would_store_a_name(tmp_path):
    # A form where the buyer's NAME was typed into the agent box: the row must be refused.
    lines = [("RESIDENTIAL PURCHASE OFFER", 15, "bold"), ("Offer ID: O99999        Date: 2025-01-01", 10, "normal"),
             ("Property: 1 Demo Oak St  (Listing L00001)", 10, "normal"),
             ("Buyer(s): Hana Kestrel and Owen Kestrel", 10, "normal"), ("Buyer's agent: Kestrel", 10, "normal"),
             ("Purchase price: $3,000,000", 12, "bold"), ("Financing: All cash", 10, "normal"),
             ("Contingencies: None", 10, "normal"), ("Close of escrow: 21 days after acceptance", 10, "normal")]
    make_offer_pdfs.write_pdf(tmp_path / "O99999.pdf", lines)
    table, failures = extract_offers.extract_folder(tmp_path)
    assert len(table) == 0 and "refused" in failures["reason"].iloc[0]
    assert "kestrel" not in failures.to_csv().lower()           # the report doesn't leak it either
