"""最优执行：Almgren-Chriss 闭式解、TWAP、VWAP 与有效前沿。

归一化口径（关键）
------------------
把总量写成 ADV 的占比 ``Q = X / ADV``，持仓与交易量都除以 ``X``：

    E[成本/名义] = η·Q·Σ tⱼ² + ½·γ·Q²        （临时 + 永久冲击，tⱼ = nⱼ/X）
    V[成本/名义] = σ²·τ·Σ hⱼ²                 （hⱼ = xⱼ/X，价格风险）

于是 ``cost_bps = 成本/名义 × 1e4``，``std_bps = √V × 1e4``，量纲直观。

最优轨迹（Almgren-Chriss 2000）::

    x(t) = X · sinh(κ(T−t)) / sinh(κT),   κ = √(λσ²/η)

λ→0 退化为 TWAP；λ 越大越前置（更快完成、冲击成本更高、价格风险更低）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _sinh_ratio(kappa: float, tau_remaining: float, tau_total: float) -> float:
    """数值稳定的 ``sinh(κ(T−t)) / sinh(κT)``。"""
    if kappa <= 0:
        return tau_remaining / tau_total
    z = kappa * tau_remaining
    zt = kappa * tau_total
    if zt < 20.0:
        return math.sinh(z) / math.sinh(zt)
    return math.exp(z - zt) * (1.0 - math.exp(-2.0 * z)) / (1.0 - math.exp(-2.0 * zt))


@dataclass
class ExecutionSchedule:
    """一条执行轨迹及其成本/风险。"""

    shares: float
    horizon: float
    n_slices: int
    holdings: np.ndarray
    trades: np.ndarray
    cost_fraction: float
    variance: float
    price: float = 1.0
    risk_aversion: float = 0.0
    adv: float = 1e6
    label: str = ""

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.horizon, self.n_slices + 1)

    @property
    def cost_bps(self) -> float:
        return self.cost_fraction * 1e4

    @property
    def std_bps(self) -> float:
        return math.sqrt(max(self.variance, 0.0)) * 1e4

    @property
    def expected_cost(self) -> float:
        """以货币计价的预期成本。"""
        return self.cost_fraction * self.price * self.shares

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "shares": self.shares,
            "horizon": self.horizon,
            "n_slices": self.n_slices,
            "adv": self.adv,
            "holdings": self.holdings.tolist(),
            "trades": self.trades.tolist(),
            "cost_fraction": self.cost_fraction,
            "cost_bps": self.cost_bps,
            "variance": self.variance,
            "std_bps": self.std_bps,
            "expected_cost": self.expected_cost,
            "risk_aversion": self.risk_aversion,
        }


def _build(shares: float, horizon: float, n_slices: int, holdings: np.ndarray,
           sigma: float, eta: float, gamma: float, price: float,
           risk_aversion: float, adv: float, label: str) -> ExecutionSchedule:
    tau = horizon / n_slices
    trades = -np.diff(holdings)
    h = holdings / shares
    t = trades / shares
    q = shares / adv
    cost_fraction = eta * q * float(np.sum(t ** 2)) + 0.5 * gamma * q * q
    variance = sigma ** 2 * tau * float(np.sum(h[1:] ** 2))
    return ExecutionSchedule(
        shares=float(shares), horizon=float(horizon), n_slices=int(n_slices),
        holdings=holdings, trades=trades, cost_fraction=float(cost_fraction),
        variance=float(variance), price=float(price),
        risk_aversion=float(risk_aversion), adv=float(adv), label=label)


def almgren_chriss(
    shares: float,
    horizon: float = 1.0,
    n_slices: int = 10,
    *,
    sigma: float = 0.02,
    eta: float = 0.01,
    gamma: float = 0.001,
    risk_aversion: float = 0.0,
    adv: float = 1_000_000.0,
    price: float = 100.0,
) -> ExecutionSchedule:
    """Almgren-Chriss 最优轨迹（``risk_aversion=0`` 即 TWAP）。"""
    if shares <= 0 or horizon <= 0:
        raise ValueError("shares / horizon 必须为正")
    n_slices = int(n_slices)
    if n_slices < 1:
        raise ValueError("n_slices 必须 >= 1")
    if sigma < 0 or eta <= 0 or gamma < 0:
        raise ValueError("sigma>=0, eta>0, gamma>=0")
    if risk_aversion < 0:
        raise ValueError("risk_aversion 不能为负")
    if adv <= 0:
        raise ValueError("adv 必须为正")

    kappa = math.sqrt(risk_aversion * sigma ** 2 / eta) if risk_aversion > 0 else 0.0
    idx = np.arange(n_slices + 1)
    remaining = horizon - idx * (horizon / n_slices)
    if kappa * horizon > 1e-10:
        holdings = np.array([shares * _sinh_ratio(kappa, float(r), horizon)
                             for r in remaining])
    else:
        holdings = np.linspace(shares, 0.0, n_slices + 1)
    holdings[0], holdings[-1] = shares, 0.0
    return _build(shares, horizon, n_slices, holdings, sigma, eta, gamma, price,
                  risk_aversion, adv, label="almgren_chriss")


def twap(shares: float, horizon: float = 1.0, n_slices: int = 10, *,
         sigma: float = 0.02, eta: float = 0.01, gamma: float = 0.001,
         adv: float = 1_000_000.0, price: float = 100.0) -> ExecutionSchedule:
    """等量切分。"""
    holdings = np.linspace(float(shares), 0.0, int(n_slices) + 1)
    return _build(float(shares), float(horizon), int(n_slices), holdings,
                  sigma, eta, gamma, price, 0.0, adv, label="twap")


def vwap(shares: float, volume_profile: Sequence[float], horizon: float = 1.0, *,
         sigma: float = 0.02, eta: float = 0.01, gamma: float = 0.001,
         adv: float = 1_000_000.0, price: float = 100.0) -> ExecutionSchedule:
    """按成交量分布切分。"""
    profile = np.asarray(volume_profile, dtype="float64").ravel()
    if profile.size < 1 or np.any(profile < 0) or profile.sum() <= 0:
        raise ValueError("volume_profile 必须非负且总和为正")
    weights = profile / profile.sum()
    trades = float(shares) * weights
    holdings = np.concatenate([[float(shares)], float(shares) - np.cumsum(trades)])
    holdings[-1] = 0.0
    return _build(float(shares), float(horizon), profile.size, holdings,
                  sigma, eta, gamma, price, 0.0, adv, label="vwap")


def efficient_frontier(
    shares: float,
    horizon: float = 1.0,
    n_slices: int = 10,
    *,
    risk_aversions: Optional[Sequence[float]] = None,
    sigma: float = 0.02,
    eta: float = 0.01,
    gamma: float = 0.001,
    adv: float = 1_000_000.0,
    price: float = 100.0,
) -> List[Dict[str, float]]:
    """不同风险厌恶下的 (方差, 成本) 曲线。"""
    lambdas = list(risk_aversions) if risk_aversions is not None \
        else [0.0, 1.0, 5.0, 25.0, 100.0, 400.0]
    points = []
    for lam in lambdas:
        schedule = almgren_chriss(shares, horizon, n_slices, sigma=sigma, eta=eta,
                                  gamma=gamma, risk_aversion=lam, adv=adv, price=price)
        points.append({"risk_aversion": float(lam),
                       "expected_cost": schedule.expected_cost,
                       "cost_bps": schedule.cost_bps,
                       "variance": schedule.variance,
                       "std_bps": schedule.std_bps})
    return points


__all__ = ["ExecutionSchedule", "almgren_chriss", "efficient_frontier", "twap", "vwap"]