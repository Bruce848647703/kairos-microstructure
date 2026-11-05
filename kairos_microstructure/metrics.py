"""微观结构指标：价差、深度、失衡、Roll、Amihud、Kyle λ、有效价差。"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .book import LimitOrderBook


def quoted_spread(bid: float, ask: float) -> float:
    if ask < bid:
        raise ValueError("ask 必须 >= bid")
    return float(ask) - float(bid)


def relative_spread(bid: float, ask: float) -> float:
    """``(ask-bid)/mid``。"""
    mid = 0.5 * (float(bid) + float(ask))
    if mid <= 0:
        raise ValueError("mid 必须为正")
    return quoted_spread(bid, ask) / mid


def book_imbalance(bid_depth: float, ask_depth: float) -> float:
    """订单簿失衡 ``(B-A)/(B+A)``，取值 [-1, 1]。"""
    total = float(bid_depth) + float(ask_depth)
    if total <= 0:
        return 0.0
    return (float(bid_depth) - float(ask_depth)) / total


def microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """微价格：按对手方挂单量加权，比 mid 更能预测下一笔成交方向。"""
    total = float(bid_size) + float(ask_size)
    if total <= 0:
        return 0.5 * (float(bid) + float(ask))
    return (float(bid) * float(ask_size) + float(ask) * float(bid_size)) / total


def roll_spread(prices: Sequence[float]) -> float:
    """Roll (1984) 隐含价差：``2·sqrt(-cov(Δpₜ, Δpₜ₋₁))``。"""
    p = np.asarray(prices, dtype="float64").ravel()
    if p.size < 4:                      # 少于 4 个点无法估计一阶自协方差
        return 0.0
    dp = np.diff(p)
    cov = float(np.cov(dp[:-1], dp[1:], ddof=1)[0, 1])
    return float(2.0 * np.sqrt(-cov)) if cov < 0 else 0.0


def amihud_illiquidity(returns: Sequence[float],
                       dollar_volume: Sequence[float]) -> float:
    """Amihud (2002)：``mean(|r| / 成交额)``（数值可再乘 1e6 放大）。"""
    r = np.asarray(returns, dtype="float64").ravel()
    dv = np.asarray(dollar_volume, dtype="float64").ravel()
    if r.size != dv.size or r.size == 0:
        raise ValueError("两个序列长度必须一致且非空")
    mask = dv > 0
    if not np.any(mask):
        raise ValueError("成交额必须存在正值")
    return float(np.mean(np.abs(r[mask]) / dv[mask]))


def effective_spread(trade_price: float, mid: float) -> float:
    """有效价差：``2·|成交价 - 中间价|``。"""
    if mid <= 0:
        raise ValueError("mid 必须为正")
    return float(2.0 * abs(float(trade_price) - float(mid)))


def effective_spread_bps(trade_price: float, mid: float) -> float:
    return effective_spread(trade_price, mid) / float(mid) * 1e4


def realized_spread(trade_price: float, mid_after: float, side: str) -> float:
    """已实现价差：成交后中间价相对成交价的变化（做市商视角，正数=赚）。"""
    side = str(side).lower()
    if side not in ("buy", "sell"):
        raise ValueError("side 必须是 buy / sell")
    sign = 1.0 if side == "sell" else -1.0     # taker 卖 -> 做市商买
    return float(sign * (float(mid_after) - float(trade_price)))


def depth_weighted_price(book: LimitOrderBook, side: str, qty: float) -> Optional[float]:
    """吃掉 ``qty`` 的加权均价（不改变订单簿）。"""
    if qty <= 0:
        raise ValueError("qty 必须为正")
    side = str(side).lower()
    levels = sorted(book.asks if side == "buy" else book.bids,
                    reverse=(side != "buy"))
    remaining, notional = float(qty), 0.0
    for price in levels:
        available = book.level_qty("sell" if side == "buy" else "buy", price)
        take = min(available, remaining)
        notional += take * price
        remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-12:
        return None
    return float(notional / qty)


def book_summary(book: LimitOrderBook, levels: int = 5) -> dict:
    """一次取齐常用指标。"""
    bid, ask = book.best_bid, book.best_ask
    if bid is None or ask is None:
        return {"best_bid": bid, "best_ask": ask, "mid": None,
                "spread": None, "relative_spread": None,
                "imbalance": 0.0, "bid_depth": 0.0, "ask_depth": 0.0}
    bid_depth = book.depth("buy", levels)
    ask_depth = book.depth("sell", levels)
    return {
        "best_bid": bid, "best_ask": ask, "mid": book.mid,
        "spread": quoted_spread(bid, ask),
        "relative_spread": relative_spread(bid, ask),
        "imbalance": book_imbalance(bid_depth, ask_depth),
        "bid_depth": bid_depth, "ask_depth": ask_depth,
        "microprice": microprice(bid, ask, book.level_qty("buy", bid),
                                 book.level_qty("sell", ask)),
    }


__all__ = [
    "amihud_illiquidity",
    "book_imbalance",
    "book_summary",
    "depth_weighted_price",
    "effective_spread",
    "effective_spread_bps",
    "microprice",
    "quoted_spread",
    "realized_spread",
    "relative_spread",
    "roll_spread",
]

# --------------------------------------------------------------------------- #
# OHLC 估计量：不用逐笔也能估波动与买卖价差
# --------------------------------------------------------------------------- #
def parkinson_volatility(high: Sequence[float], low: Sequence[float],
                         periods_per_year: float = 252.0) -> float:
    """Parkinson (1980) 高低价波动率：``σ² = E[ln(H/L)²] / (4 ln2)``。

    只用到最高/最低价，效率约是收盘价收益率的 5 倍，但对跳空不敏感。
    """
    h = np.asarray(high, dtype="float64").ravel()
    l = np.asarray(low, dtype="float64").ravel()
    if h.size != l.size or h.size < 2:
        raise ValueError("high / low 长度必须一致且至少 2 个点")
    if np.any(h <= 0) or np.any(l <= 0) or np.any(h < l):
        raise ValueError("价格必须为正且 high >= low")
    hl = np.log(h / l)
    variance = float(np.mean(hl ** 2) / (4.0 * math.log(2.0)))
    return float(np.sqrt(max(variance, 0.0) * periods_per_year))


def garman_klass_volatility(open_: Sequence[float], high: Sequence[float],
                            low: Sequence[float], close: Sequence[float],
                            periods_per_year: float = 252.0) -> float:
    """Garman-Klass (1980)：同时用 O/H/L/C，比 Parkinson 更有效。"""
    o = np.asarray(open_, dtype="float64").ravel()
    h = np.asarray(high, dtype="float64").ravel()
    l = np.asarray(low, dtype="float64").ravel()
    c = np.asarray(close, dtype="float64").ravel()
    if not (o.size == h.size == l.size == c.size) or o.size < 2:
        raise ValueError("O/H/L/C 长度必须一致且至少 2 个点")
    if np.any(np.minimum.reduce([o, h, l, c]) <= 0):
        raise ValueError("价格必须为正")
    hl = 0.5 * np.log(h / l) ** 2
    co = (2.0 * math.log(2.0) - 1.0) * np.log(c / o) ** 2
    variance = float(np.mean(hl - co))
    return float(np.sqrt(max(variance, 0.0) * periods_per_year))


def corwin_schultz_spread(high: Sequence[float], low: Sequence[float],
                          close: Optional[Sequence[float]] = None) -> float:
    """Corwin-Schultz (2012) 高低价买卖价差估计（返回**价格比例**）。

    用连续两日的高低点：单日波幅包含价差与真实波动，两日合并后的波幅
    稀释了价差，二者的差即可反推价差。日频数据上比 Roll 稳健得多
    （Roll 依赖收益率的负自相关，常被真实趋势淹没）。
    """
    h = np.asarray(high, dtype="float64").ravel()
    l = np.asarray(low, dtype="float64").ravel()
    if h.size != l.size or h.size < 3:
        raise ValueError("high / low 长度必须一致且至少 3 个点")
    if np.any(h <= 0) or np.any(l <= 0) or np.any(h < l):
        raise ValueError("价格必须为正且 high >= low")

    beta = float(np.mean(np.log(h[1:] / l[1:]) ** 2 + np.log(h[:-1] / l[:-1]) ** 2))
    hh = np.maximum(h[1:], h[:-1])
    ll = np.minimum(l[1:], l[:-1])
    gamma = float(np.mean(np.log(hh / ll) ** 2))

    k = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / k - math.sqrt(max(gamma, 0.0) / k)
    spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return float(max(spread, 0.0))


def abdi_ranaldo_spread(high: Sequence[float], low: Sequence[float],
                        close: Sequence[float]) -> float:
    """Abdi-Ranaldo (2017) 价差估计（价格比例）。

    用「收盘价相对当日中点的位置」与「下一日收盘价相对同一中点的位置」的
    协方差反推有效价差，比 Corwin-Schultz 更稳健，日频数据上很少被截断为 0。
    """
    h = np.asarray(high, dtype="float64").ravel()
    l = np.asarray(low, dtype="float64").ravel()
    c = np.asarray(close, dtype="float64").ravel()
    if not (h.size == l.size == c.size) or h.size < 4:
        raise ValueError("H/L/C 长度必须一致且至少 4 个点")
    if np.any(h <= 0) or np.any(l <= 0) or np.any(c <= 0) or np.any(h < l):
        raise ValueError("价格必须为正且 high >= low")
    eta = 0.5 * (np.log(h) + np.log(l))
    log_c = np.log(c)
    x = (log_c[:-2] - eta[:-2]) * (log_c[:-2] - eta[1:-1])
    estimate = 4.0 * float(np.mean(x))
    return float(math.sqrt(max(estimate, 0.0)))
