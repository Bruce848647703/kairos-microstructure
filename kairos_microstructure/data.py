"""合成微观结构数据：买卖价跳动的成交序列、Kyle 线性冲击、订单流。"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def synthetic_bounce(n: int = 20000, mid: float = 100.0, half_spread: float = 0.01,
                     mid_sigma: float = 0.0005, seed: int = 7) -> Dict[str, Any]:
    """买卖价跳动（bid-ask bounce）模型：``p_t = mid_t ± half_spread``。

    中间价做微小随机游走，成交方向随机 —— Roll 的隐含价差应能还原
    ``2·half_spread``。
    """
    if n < 3 or mid <= 0 or half_spread < 0:
        raise ValueError("参数非法")
    rng = np.random.default_rng(seed)
    mid_path = mid + np.cumsum(rng.normal(0.0, mid_sigma, size=n))
    side = rng.choice([-1.0, 1.0], size=n)
    prices = mid_path + side * half_spread
    return {
        "mid": mid_path,
        "price": prices,
        "side": side,
        "true_spread": 2.0 * half_spread,
    }


def synthetic_kyle(n: int = 5000, lam: float = 2e-5, mid: float = 100.0,
                   volume_sigma: float = 500.0, noise: float = 0.002,
                   seed: int = 3) -> Dict[str, Any]:
    """Kyle 模型：``Δp = λ·q + ε``，用于检验 ``kyle_lambda`` 的还原能力。"""
    if n < 2 or lam < 0 or volume_sigma <= 0:
        raise ValueError("参数非法")
    rng = np.random.default_rng(seed)
    signed_volume = rng.normal(0.0, volume_sigma, size=n)
    price_changes = lam * signed_volume + rng.normal(0.0, noise, size=n)
    return {
        "price_changes": price_changes,
        "signed_volume": signed_volume,
        "true_lambda": lam,
    }


def synthetic_order_flow(n: int = 2000, mid: float = 100.0, tick: float = 0.01,
                         seed: int = 5) -> Dict[str, Any]:
    """简单订单流：方向、价格偏移（tick 数）、数量。"""
    rng = np.random.default_rng(seed)
    side = rng.integers(0, 2, size=n)
    offset = rng.integers(1, 6, size=n)
    qty = rng.integers(1, 10, size=n) * 100.0
    return {
        "side": np.where(side == 0, "buy", "sell"),
        "offset_ticks": offset,
        "qty": qty,
        "mid": float(mid),
        "tick": float(tick),
    }


__all__ = ["synthetic_bounce", "synthetic_kyle", "synthetic_order_flow"]