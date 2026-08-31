"""真实数据校准测试。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kairos_microstructure.metrics import (
    abdi_ranaldo_spread,
    garman_klass_volatility,
    parkinson_volatility,
)
from kairos_microstructure.calibrate import (
    CalibrationResult,
    calibrate,
    calibrate_csv,
    cost_curve,
    load_price_volume,
)


def make_series(n: int = 1000, sigma_daily: float = 0.01, volume: float = 100_000.0,
                seed: int = 3):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, sigma_daily, size=n - 1)
    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = np.full(n, volume)
    return np.concatenate([[100.0], prices]), volumes


def write_csv(path, prices, volumes):
    lines = ["date,close,volume"]
    for i, (p, v) in enumerate(zip(prices, volumes)):
        lines.append(f"2020-01-{i % 28 + 1:02d},{p:.6f},{v:.0f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_price_volume(tmp_path):
    prices, volumes = make_series(n=100)
    path = tmp_path / "data.csv"
    write_csv(path, prices, volumes)
    p, v = load_price_volume(str(path))
    assert p.size == 100 and v.size == 100
    assert p[0] == pytest.approx(prices[0])
    with pytest.raises(FileNotFoundError):
        load_price_volume(str(tmp_path / "missing.csv"))


def test_calibrate_metrics():
    prices, volumes = make_series(n=2000, sigma_daily=0.015, volume=200_000)
    result = calibrate(prices, volumes, reference_participation=0.1)
    assert isinstance(result, CalibrationResult)
    assert result.n_obs == prices.size
    assert result.sigma == pytest.approx(0.015 * math.sqrt(252), rel=0.1)
    assert result.adv == pytest.approx(200_000)
    assert result.dollar_adv == pytest.approx(200_000 * result.mid_price, rel=0.2)
    assert result.eta == pytest.approx(0.5 * result.sigma_daily / math.sqrt(0.1), rel=1e-9)
    assert result.assumptions


def test_cost_curve_monotone():
    prices, volumes = make_series()
    result = calibrate(prices, volumes)
    rows = cost_curve(result)
    costs = [r["cost_bps"] for r in rows]
    risks = [r["risk_bps"] for r in rows]
    shares = [r["shares"] for r in rows]
    assert costs == sorted(costs)
    assert risks == sorted(risks)
    assert shares == sorted(shares)
    assert all(c > 0 for c in costs)


def test_calibrate_csv_roundtrip(tmp_path):
    prices, volumes = make_series(n=300)
    path = tmp_path / "csv.csv"
    write_csv(path, prices, volumes)
    result = calibrate_csv(str(path))
    assert result.n_obs == 300
    assert result.adv == pytest.approx(100_000)


def test_validation():
    with pytest.raises(ValueError):
        calibrate([100.0] * 10, [1.0] * 10)          # 数据太少
    with pytest.raises(ValueError):
        calibrate([100.0] * 40, [1.0] * 39)          # 长度不一致
    with pytest.raises(ValueError):
        calibrate([-1.0] * 40, [1.0] * 40)           # 负价格
    prices, volumes = make_series(n=100)
    with pytest.raises(ValueError):
        calibrate(prices, volumes, reference_participation=0.0)

def test_load_ohlcv_and_ohlc_calibration(tmp_path):
    from kairos_microstructure.calibrate import load_ohlcv

    prices, volumes = make_series(n=200)
    high = prices * 1.012
    low = prices * 0.988
    path = tmp_path / "ohlc.csv"
    lines = ["date,open,high,low,close,volume"]
    for i, (p, h, l, v) in enumerate(zip(prices, high, low, volumes)):
        lines.append(f"2020-01-{i % 28 + 1:02d},{p:.6f},{h:.6f},{l:.6f},{p:.6f},{v:.0f}")
    path.write_text("\n".join(lines), encoding="utf-8")

    data = load_ohlcv(str(path))
    assert data["close"].size == 200 and data["high"].size == 200

    result = calibrate(data["close"], data["volume"], high=data["high"], low=data["low"])
    assert result.sigma_intraday is not None and result.sigma_intraday > 0
    assert result.spread_ohlc >= 0
    assert result.spread_source in ("Roll", "Corwin-Schultz", "Abdi-Ranaldo")
    assert result.to_dict()["spread_ohlc_bps"] == pytest.approx(result.spread_ohlc * 1e4)


def test_ohlc_estimators_basic_properties():
    prices, _ = make_series(n=200)
    high = prices * 1.02
    low = prices * 0.98
    varying_high = prices * 1.04
    varying_low = prices * 0.96

    p1 = parkinson_volatility(high, low)
    p2 = parkinson_volatility(varying_high, varying_low)
    assert 0 < p1 < p2                      # 区间越宽，估计越大

    gk = garman_klass_volatility(prices, high, low, prices)
    assert gk > 0

    ar = abdi_ranaldo_spread(high, low, prices)
    assert ar >= 0
