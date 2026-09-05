"""冲击模型与最优执行测试。"""
from __future__ import annotations

import numpy as np
import pytest

from kairos_microstructure.book import build_book
from kairos_microstructure.data import synthetic_kyle
from kairos_microstructure.impact import (
    almgren_chriss_impact,
    implementation_shortfall,
    kyle_lambda,
    linear_impact,
    square_root_impact,
    walk_book,
)
from kairos_microstructure.optimal import (
    almgren_chriss,
    efficient_frontier,
    twap,
    vwap,
)

PARAMS = dict(sigma=0.02, eta=0.01, gamma=0.001, adv=1_000_000.0, price=100.0)


# --------------------------------------------------------------------------- #
# 冲击
# --------------------------------------------------------------------------- #
def test_square_root_impact_monotone_in_size():
    small = square_root_impact(0.02, 1_000, 1_000_000)
    large = square_root_impact(0.02, 100_000, 1_000_000)
    assert 0 < small < large
    assert square_root_impact(0.02, 0, 1_000_000) == 0.0
    assert large / small == pytest.approx(np.sqrt(100), rel=1e-9)


def test_linear_impact():
    assert linear_impact(0.1, 50_000, 1_000_000) == pytest.approx(0.005)
    with pytest.raises(ValueError):
        linear_impact(0.1, 1_000, 0)


def test_almgren_chriss_impact_decomposition():
    res = almgren_chriss_impact(100_000, 1_000_000, eta=0.01, gamma=0.001, participation=0.1)
    assert res["permanent"] == pytest.approx(0.001 * 0.1)
    assert res["temporary"] == pytest.approx(0.01 * 0.1 / 0.1)
    assert res["total"] == pytest.approx(res["temporary"] + res["permanent"])
    with pytest.raises(ValueError):
        almgren_chriss_impact(1_000, 1_000_000, participation=0.0)


def test_kyle_lambda_recovers_slope():
    data = synthetic_kyle(n=20000, lam=2e-5, noise=0.001, seed=11)
    lam = kyle_lambda(data["price_changes"], data["signed_volume"])
    assert lam == pytest.approx(2e-5, rel=0.05)


def test_implementation_shortfall_sign():
    buy = implementation_shortfall(100.0, 100.1, "buy", 1000)
    assert buy["shortfall"] == pytest.approx(100.0)
    assert buy["shortfall_bps"] == pytest.approx(10.0)
    sell = implementation_shortfall(100.0, 99.9, "sell", 1000)
    assert sell["shortfall"] > 0
    with pytest.raises(ValueError):
        implementation_shortfall(100, 100, "hold")


def test_walk_book_cost_increases_with_size():
    previous = -np.inf
    for qty in (100, 300, 600):
        book = build_book(100.0, levels=20, level_size=100)
        result = walk_book(book, "buy", qty)
        assert result["filled"] == pytest.approx(qty)
        assert result["shortfall_bps"] > previous
        previous = result["shortfall_bps"]


def test_walk_book_partial_fill():
    book = build_book(100.0, levels=1, level_size=10)
    result = walk_book(book, "buy", 100)
    assert result["filled"] == pytest.approx(10)
    assert result["unfilled"] == pytest.approx(90)
    with pytest.raises(ValueError):
        walk_book(__import__("kairos_microstructure").LimitOrderBook(), "buy", 10)


# --------------------------------------------------------------------------- #
# 最优执行
# --------------------------------------------------------------------------- #
def test_zero_risk_aversion_equals_twap():
    ac = almgren_chriss(100_000, 1.0, 10, risk_aversion=0.0, **PARAMS)
    tw = twap(100_000, 1.0, 10, **PARAMS)
    assert np.allclose(ac.trades, tw.trades)
    assert ac.cost_bps == pytest.approx(tw.cost_bps)
    assert ac.std_bps == pytest.approx(tw.std_bps)


def test_higher_risk_aversion_front_loads():
    low = almgren_chriss(100_000, 1.0, 10, risk_aversion=1.0, **PARAMS)
    high = almgren_chriss(100_000, 1.0, 10, risk_aversion=400.0, **PARAMS)
    assert high.trades[0] > low.trades[0]
    assert high.trades[-1] < low.trades[-1]
    assert high.std_bps < low.std_bps
    assert high.cost_bps > low.cost_bps


def test_schedule_conserves_shares():
    schedule = almgren_chriss(123_456, 2.0, 8, risk_aversion=10.0, **PARAMS)
    assert schedule.trades.sum() == pytest.approx(123_456)
    assert schedule.holdings[0] == pytest.approx(123_456)
    assert schedule.holdings[-1] == pytest.approx(0.0)
    assert np.all(schedule.trades >= -1e-9)
    assert schedule.trades[0] > 0


def test_vwap_follows_profile():
    profile = [1, 2, 3, 4]
    schedule = vwap(1000, profile, 1.0, **PARAMS)
    assert schedule.trades.tolist() == pytest.approx([100, 200, 300, 400])
    assert schedule.label == "vwap"


def test_efficient_frontier_tradeoff():
    points = efficient_frontier(100_000, 1.0, 10, **PARAMS)
    assert len(points) >= 4
    costs = [p["cost_bps"] for p in points]
    risks = [p["std_bps"] for p in points]
    assert costs == sorted(costs)                   # 风险厌恶越高成本越高
    assert risks == sorted(risks, reverse=True)     # 风险越低
    assert points[0]["risk_aversion"] == 0.0


def test_execution_validation():
    with pytest.raises(ValueError):
        almgren_chriss(0)
    with pytest.raises(ValueError):
        almgren_chriss(100, -1)
    with pytest.raises(ValueError):
        almgren_chriss(100, 1, 0)
    with pytest.raises(ValueError):
        almgren_chriss(100, risk_aversion=-1)
    with pytest.raises(ValueError):
        vwap(100, [0, 0])