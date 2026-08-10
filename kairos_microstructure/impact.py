"""市场冲击模型：平方根冲击、线性冲击、Kyle λ、执行短缺。

约定
----
- ``sigma`` 是**每单位时间**的价格波动率（价格单位），``qty`` 是股数，
  ``adv``（Average Daily Volume）用同一时间单位下的成交量；
- 冲击函数返回**价格的比例**（0.001 = 10bp），乘以参考价即为价格单位。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from .book import Fill, LimitOrderBook


def square_root_impact(sigma: float, qty: float, adv: float,
                       coefficient: float = 0.5) -> float:
    """平方根冲击：``c · σ · sqrt(Q/ADV)``（业界最常用的经验律）。"""
    if sigma < 0 or coefficient < 0:
        raise ValueError("sigma / coefficient 不能为负")
    if qty < 0:
        raise ValueError("qty 不能为负")
    if adv <= 0:
        raise ValueError("adv 必须为正")
    return float(coefficient * sigma * math.sqrt(qty / adv))


def linear_impact(coefficient: float, qty: float, adv: float) -> float:
    """线性冲击：``c · (Q/ADV)``。"""
    if coefficient < 0 or qty < 0 or adv <= 0:
        raise ValueError("参数非法（coefficient>=0, qty>=0, adv>0）")
    return float(coefficient * qty / adv)


def almgren_chriss_impact(
    total_shares: float,
    adv: float,
    *,
    sigma: float = 0.02,
    eta: float = 0.01,
    gamma: float = 0.001,
    participation: float = 0.1,
) -> Dict[str, float]:
    """Almgren-Chriss 的临时/永久冲击分解（比例形式）。

    ``temporary = η · (Q / (participation·ADV))``，``permanent = γ · (Q/ADV)``。
    """
    if total_shares <= 0 or adv <= 0:
        raise ValueError("total_shares 与 adv 必须为正")
    if not 0 < participation <= 1:
        raise ValueError("participation 必须在 (0, 1] 内")
    q = float(total_shares) / float(adv)
    temporary = float(eta * q / participation)
    permanent = float(gamma * q)
    return {
        "temporary": temporary,
        "permanent": permanent,
        "total": temporary + permanent,
        "participation": float(participation),
        "sigma": float(sigma),
    }


def kyle_lambda(price_changes: Sequence[float],
                signed_volume: Sequence[float]) -> float:
    """Kyle λ：``Δp = λ·q`` 的 OLS 斜率（价格单位的冲击/股）。"""
    dp = np.asarray(price_changes, dtype="float64").ravel()
    q = np.asarray(signed_volume, dtype="float64").ravel()
    if dp.size != q.size or dp.size < 2:
        raise ValueError("两个序列长度必须一致且至少 2 个点")
    design = np.column_stack([np.ones_like(q), q])
    coef, *_ = np.linalg.lstsq(design, dp, rcond=None)
    return float(coef[1])


def implementation_shortfall(decision_price: float, average_price: float,
                             side: str, qty: float = 1.0) -> Dict[str, float]:
    """执行短缺：相对决策价的不利成交（正数 = 亏）。"""
    side = str(side).lower()
    if side not in ("buy", "sell"):
        raise ValueError("side 必须是 buy / sell")
    if decision_price <= 0 or qty <= 0:
        raise ValueError("decision_price / qty 必须为正")
    sign = 1.0 if side == "buy" else -1.0
    diff = sign * (float(average_price) - float(decision_price))
    return {
        "shortfall": diff * float(qty),
        "shortfall_per_share": diff,
        "shortfall_bps": diff / float(decision_price) * 1e4,
    }


def walk_book(book: LimitOrderBook, side: str, qty: float,
              timestamp: float = 0.0) -> Dict[str, Any]:
    """用市价单吃掉订单簿，返回加权均价、逐档成交与冲击。"""
    if book.mid is None:
        raise ValueError("订单簿为空")
    decision_price = float(book.mid)
    fills: Sequence[Fill] = book.market_order(side, qty, timestamp)
    filled = float(sum(f.qty for f in fills))
    notional = float(sum(f.notional for f in fills))
    average = notional / filled if filled > 0 else decision_price
    impact = implementation_shortfall(decision_price, average, side, filled)
    return {
        "side": side,
        "requested": float(qty),
        "filled": filled,
        "unfilled": float(qty) - filled,
        "average_price": average,
        "decision_price": decision_price,
        "notional": notional,
        "levels_consumed": len({round(f.price, 10) for f in fills}),
        "fills": [f.to_dict() for f in fills],
        **impact,
    }


__all__ = [
    "almgren_chriss_impact",
    "implementation_shortfall",
    "kyle_lambda",
    "linear_impact",
    "square_root_impact",
    "walk_book",
]