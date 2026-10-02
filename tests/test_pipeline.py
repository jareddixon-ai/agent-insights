import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import agent_model
import clean_data
import make_messy_data
import offer_win_rates
import shrinkage

ROOT = Path(__file__).resolve().parent.parent


def read(data_dir, name, **kw):
    return pd.read_csv(data_dir / name, keep_default_na=False, **kw)


# ---- Data generator ----
def test_same_seed_gives_same_files(data_dir, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "src" / "generate_fake_data.py")],
                   cwd=tmp_path, check=True, capture_output=True)
    for name in ["sales.csv", "offers.csv", "true_skills.csv"]:
        a = hashlib.md5((data_dir / name).read_bytes()).hexdigest()
        b = hashlib.md5((tmp_path / "data" / "fake" / name).read_bytes()).hexdigest()
        assert a == b, name


def test_prices_look_like_santa_barbara(data_dir):
    price = read(data_dir, "sales.csv")["sold_price"]
    assert price.min() >= 1_800_000          # nothing far under $2M
    assert 3_000_000 < price.median() < 4_500_000
    assert price.max() > 15_000_000          # heavy upper tail exists


def test_sales_basic_rules(data_dir):
    s = read(data_dir, "sales.csv")
    assert len(s) == 5000 and s["listing_id"].is_unique
    assert (s["listing_agent_id"] != s["buyer_agent_id"]).all()


def test_offers_winner_is_highest_offer(data_dir):
    s, o = read(data_dir, "sales.csv"), read(data_dir, "offers.csv")
    assert (o.groupby("listing_id")["outcome"].apply(lambda x: (x == "won").sum()) == 1).all()
    top = o.groupby("listing_id")["offer_price"].max()
    assert (top == s.set_index("listing_id")["sold_price"]).all()


# ---- Cleaning ----
def test_cleaning_removes_mess_without_leaking(data_dir):
    sales = read(data_dir, "sales.csv", dtype=str).replace("", np.nan)
    sales = sales.astype({c: float for c in ["beds", "baths", "sqft", "lot_sqft",
                                              "year_built", "assessed_value", "sold_price"]})
    messy = make_messy_data.make_messy(sales, np.random.default_rng(1))
    clean, report = clean_data.clean_sales(messy)
    assert clean["listing_id"].is_unique
    assert clean["sold_price"].notna().all() and clean["assessed_value"].notna().all()
    assert report["duplicates_dropped"] > 0 and report["unsold_dropped"] > 0
    assert clean["zip"].str.len().eq(5).all()


# ---- Model ----
def test_model_explains_prices(data_dir):
    _, r2, _ = agent_model.fit_agent_effects(read(data_dir, "sales.csv"))
    assert r2 > 0.9


def test_model_finds_real_skill(data_dir):
    sales, skills = read(data_dir, "sales.csv"), read(data_dir, "true_skills.csv").set_index("agent_id")
    effects, _, _ = agent_model.fit_agent_effects(sales)
    x = effects["seller_effect"].dropna()
    assert x.rank().corr(skills["seller_premium"].reindex(x.index).rank()) > 0.6


# ---- Shrinkage ----
def test_shrinkage_pulls_small_samples_more_and_cuts_error(data_dir):
    sales, skills = read(data_dir, "sales.csv"), read(data_dir, "true_skills.csv").set_index("agent_id")
    effects, _, sd = agent_model.fit_agent_effects(sales)
    t = shrinkage.shrink(effects["seller_effect"], effects["listings"], sd)
    assert t["weight"].between(0, 1).all()
    few, many = t[t["deals"] < 5], t[t["deals"] > 50]
    assert few["weight"].mean() < many["weight"].mean()
    truth = (skills["seller_premium"] - skills["seller_premium"].mean()).reindex(t.index)
    assert (t["score"] - truth).abs().mean() < (t["raw"] - truth).abs().mean()
    assert ((truth >= t["low"]) & (truth <= t["high"])).mean() > 0.9   # ranges are honest


def test_win_rates_shrink_toward_average(data_dir):
    t = offer_win_rates.win_rates(read(data_dir, "offers.csv"))
    assert t["win_rate"].between(0, 1).all()
    assert (t["win_rate"] - t.attrs["p0"]).abs().le((t["raw_rate"] - t.attrs["p0"]).abs() + 1e-9).all()
