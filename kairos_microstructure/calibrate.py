"""用真实日线数据校准冲击参数。

能估什么、不能估什么（诚实说明）
--------------------------------
日线只有 OHLCV，**没有逐笔成交与订单流**，因此：

- 可估：年化波动 σ、ADV（成交量中位数）、Amihud 非流动性、Roll 隐含价差、
  以及用「带符号成交量」代理估计的 Kyle λ；
- 不可直接识别：Almgren-Chriss 的 η（临时冲击）。这里用**平方根经验律**在参考
  参与率上锚定：``η·Q_ref = c·σ·√Q_ref`` ⇒ ``η = c·σ/√Q_ref``；
- 永久冲击 γ 用 Kyle λ 的回归斜率（ADV 单位）直接映射：``γ = λ``。

因此本模块输出的是「**有先验锚定的校准**」，不是无假设的结构估计。所有假设都在
``CalibrationResult.assumptions`` 里显式列出，方便复核。
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import (
    abdi_ranaldo_spread,
    amihud_illiquidity,
    corwin_schultz_spread,
    parkinson_volatility,
    roll_spread,
)


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def load_price_volume(path: str, price_column: str = "close",
                      volume_column: str = "volume") -> Tuple[np.ndarray, np.ndarray]:
    """读取 ``date,close,volume`` 形式的 CSV（列名大小写不敏感）。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} 没有数据")

    lowered = {str(k).strip().lower(): k for k in rows[0].keys()}
    p_key = lowered.get(price_column.lower())
    v_key = lowered.get(volume_column.lower())
    if p_key is None:
        raise ValueError(f"{path} 缺少价格列 {price_column!r}，实际列：{list(rows[0])}")

    prices, volumes = [], []
    for row in rows:
        try:
            price = float(row[p_key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price) or price <= 0:
            continue
        volume = 0.0
        if v_key is not None:
            try:
                volume = float(row[v_key] or 0.0)
            except (TypeError, ValueError):
                volume = 0.0
        prices.append(price)
        volumes.append(max(volume, 0.0))
    if len(prices) < 30:
        raise ValueError(f"{path} 有效数据不足（{len(prices)} 行，至少 30 行）")
    return np.asarray(prices, dtype="float64"), np.asarray(volumes, dtype="float64")


# --------------------------------------------------------------------------- #
# 校准结果
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    """冲击参数校准结果（全部为可 JSON 序列化的 Python 标量）。"""

    n_obs: int
    sigma: float                     # 年化波动
    sigma_daily: float               # 日频波动（用于平方根锚定）
    sigma_intraday: Optional[float]  # OHLC 口径的日频波动（GK/Parkinson）
    adv: float                       # 成交量中位数（股）
    dollar_adv: float                # 成交额中位数（货币）
    amihud: float                    # mean(|r| / 成交额)
    roll_spread: float               # Roll 隐含价差（价格单位）
    roll_spread_bps: float
    spread_ohlc: float               # OHLC 估计的价差（比例）
    spread_ohlc_bps: float
    spread_source: str
    mid_price: float
    kyle_lambda: float               # 回归斜率：每 1×ADV 的带符号成交量对应的收益率
    kyle_lambda_price: float         # 换成价格单位（每 1×ADV）
    gamma: float                     # 永久冲击（分数 / Q）
    eta: float                       # 临时冲击（分数 / Q）
    sqrt_coefficient: float
    reference_participation: float
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_obs": self.n_obs, "sigma": self.sigma, "sigma_daily": self.sigma_daily, "adv": self.adv,
            "dollar_adv": self.dollar_adv, "amihud": self.amihud,
            "roll_spread": self.roll_spread, "roll_spread_bps": self.roll_spread_bps,
            "spread_ohlc": self.spread_ohlc, "spread_ohlc_bps": self.spread_ohlc_bps,
            "spread_source": self.spread_source,
            "mid_price": self.mid_price,
            "kyle_lambda": self.kyle_lambda,
            "kyle_lambda_price": self.kyle_lambda_price,
            "gamma": self.gamma, "eta": self.eta,
            "sqrt_coefficient": self.sqrt_coefficient,
            "reference_participation": self.reference_participation,
            "assumptions": list(self.assumptions),
        }


def calibrate(prices: Sequence[float], volumes: Sequence[float], *,
              high: Optional[Sequence[float]] = None,
              low: Optional[Sequence[float]] = None,
              periods_per_year: float = 252.0,
              sqrt_coefficient: float = 0.5,
              reference_participation: float = 0.1) -> CalibrationResult:
    """由价格与成交量序列校准冲击参数。"""
    p = np.asarray(prices, dtype="float64").ravel()
    v = np.asarray(volumes, dtype="float64").ravel()
    if p.size != v.size:
        raise ValueError("prices 与 volumes 长度必须一致")
    if p.size < 30:
        raise ValueError("至少需要 30 个观测")
    if np.any(p <= 0):
        raise ValueError("价格必须为正")
    if not 0 < reference_participation <= 1:
        raise ValueError("reference_participation 必须在 (0, 1] 内")

    returns = np.diff(np.log(p))
    sigma_daily = float(returns.std(ddof=1))
    sigma = sigma_daily * math.sqrt(periods_per_year)
    volume = v[1:]
    adv = float(np.median(volume)) if np.any(volume > 0) else 0.0
    if adv <= 0:
        raise ValueError("成交量为零，无法校准 ADV")
    dollar_volume = volume * p[1:]
    dollar_adv = float(np.median(dollar_volume))

    amihud = amihud_illiquidity(returns, dollar_volume)
    roll = float(roll_spread(p))
    mid_price = float(np.median(p))

    # OHLC 口径（可选）：更有效的波动估计 + 价差交叉校验
    sigma_intraday = None
    spread_ohlc = 0.0
    spread_source = "Roll"
    if high is not None and low is not None:
        h = np.asarray(high, dtype="float64").ravel()
        l = np.asarray(low, dtype="float64").ravel()
        if h.size == p.size and l.size == p.size:
            sigma_intraday = float(parkinson_volatility(h, l, periods_per_year) /
                                   math.sqrt(periods_per_year))
            cs = corwin_schultz_spread(h, l)
            ar = abdi_ranaldo_spread(h, l, p)
            candidates = {
                "Corwin-Schultz": cs,
                "Abdi-Ranaldo": ar,
                "Roll": roll / mid_price if mid_price > 0 else 0.0,
            }
            spread_source = max(candidates, key=candidates.get)
            spread_ohlc = float(max(candidates.values()))

    # 带符号成交量代理：sign(r)·volume/ADV
    signed_q = np.sign(returns) * volume / adv
    design = np.column_stack([np.ones_like(signed_q), signed_q])
    coef, *_ = np.linalg.lstsq(design, returns, rcond=None)
    lambda_frac = float(coef[1])

    gamma = max(lambda_frac, 0.0)                       # 永久冲击（分数/Q）
    # 平方根律的经验标定是「日频」口径：σ_daily 与日 ADV 匹配
    eta = float(sqrt_coefficient * sigma_daily / math.sqrt(reference_participation))

    assumptions = [
        "Kyle λ 用 sign(r)·volume/ADV 作为带符号成交量代理（日线无逐笔方向）",
        "OHLC 价差在 Roll / Corwin-Schultz / Abdi-Ranaldo 中取最大正值（日频估计噪声大）",
        "永久冲击 γ 直接取 Kyle λ 的回归斜率（ADV 单位）",
        f"临时冲击 η 用平方根律在 Q_ref={reference_participation:g} 处锚定："
        f"η = c·σ_daily/√Q_ref，c={sqrt_coefficient:g}（日频口径）",
        "Roll 价差在日线频率上通常偏小甚至为 0，仅作量级参考",
    ]
    return CalibrationResult(
        n_obs=int(p.size), sigma=sigma, sigma_daily=sigma_daily,
        sigma_intraday=sigma_intraday,
        adv=adv, dollar_adv=dollar_adv,
        amihud=float(amihud), roll_spread=roll,
        roll_spread_bps=roll / mid_price * 1e4 if mid_price > 0 else 0.0,
        spread_ohlc=spread_ohlc,
        spread_ohlc_bps=spread_ohlc * 1e4,
        spread_source=spread_source,
        mid_price=mid_price,
        kyle_lambda=lambda_frac, kyle_lambda_price=lambda_frac * mid_price,
        gamma=gamma, eta=eta, sqrt_coefficient=float(sqrt_coefficient),
        reference_participation=float(reference_participation),
        assumptions=assumptions)


def load_ohlcv(path: str) -> Dict[str, np.ndarray]:
    """读取 OHLCV（``date,open,high,low,close,volume``，列名大小写不敏感）。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} 没有数据")
    lowered = {str(k).strip().lower(): k for k in rows[0].keys()}
    required = ("high", "low", "close")
    missing = [c for c in required if c not in lowered]
    if missing:
        raise ValueError(f"{path} 缺少列 {missing}，实际列：{list(rows[0])}")

    out: Dict[str, List[float]] = {k: [] for k in ("open", "high", "low", "close", "volume")}
    for row in rows:
        try:
            vals = {k: float(row[lowered[k]]) for k in required}
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(v) and v > 0 for v in vals.values()):
            continue
        out["open"].append(float(row[lowered["open"]]) if "open" in lowered else vals["close"])
        out["high"].append(vals["high"])
        out["low"].append(vals["low"])
        out["close"].append(vals["close"])
        try:
            out["volume"].append(max(float(row[lowered["volume"]]), 0.0) if "volume" in lowered else 0.0)
        except (TypeError, ValueError):
            out["volume"].append(0.0)
    if len(out["close"]) < 30:
        raise ValueError(f"{path} 有效数据不足（{len(out['close'])} 行）")
    return {k: np.asarray(v, dtype="float64") for k, v in out.items()}


def calibrate_csv(path: str, **kwargs: Any) -> CalibrationResult:
    """读 CSV 并校准；有 OHLC 时自动使用高低价估计量。"""
    try:
        data = load_ohlcv(path)
    except ValueError:
        prices, volumes = load_price_volume(path)
        return calibrate(prices, volumes, **kwargs)
    return calibrate(data["close"], data["volume"], high=data["high"],
                     low=data["low"], **kwargs)


def cost_curve(result: CalibrationResult,
               participations: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
               horizon: float = 1.0, n_slices: int = 10,
               risk_aversion: float = 0.0) -> List[Dict[str, float]]:
    """按校准参数，给出不同参与率下的执行成本（bp）与风险（bp）。

    使用 Almgren-Chriss 的归一化成本公式（单次执行，不再切分）：
    ``cost_fraction = η·Q + ½γQ²``，``risk_fraction = σ·√(T/3)·Q``。
    """
    if not participations:
        raise ValueError("participations 不能为空")
    rows = []
    for q in participations:
        if not 0 < q <= 1:
            raise ValueError("participation 必须在 (0, 1] 内")
        cost = result.eta * q + 0.5 * result.gamma * q * q
        risk = result.sigma * math.sqrt(horizon / 3.0) * q
        rows.append({
            "participation": float(q),
            "shares": float(q * result.adv),
            "cost_bps": float(cost * 1e4),
            "risk_bps": float(risk * 1e4),
            "cost_per_share": float(cost * result.mid_price),
        })
    return rows


__all__ = [
    "CalibrationResult",
    "load_ohlcv",
    "calibrate",
    "calibrate_csv",
    "cost_curve",
    "load_price_volume",
]