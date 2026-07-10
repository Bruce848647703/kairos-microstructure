"""Kairos Microstructure —— 自研市场微观结构库。

- **订单簿**：限价/市价单、FIFO 队列、排队位置、深度与快照
- **冲击**：平方根 / 线性 / Almgren-Chriss 分解、Kyle λ、执行短缺、走簿均价
- **最优执行**：Almgren-Chriss 闭式解、TWAP、VWAP、有效前沿
- **指标**：价差、失衡、微价格、Roll、Amihud、有效/已实现价差

快速开始::

    pip install -e .
    python -m kairos_microstructure            # 完整演示
    python -m kairos_microstructure optimal --shares 100000 --risk 1e-5
"""
from __future__ import annotations

from .book import Fill, LimitOrderBook, Order, build_book
from .calibrate import (
    CalibrationResult,
    calibrate,
    calibrate_csv,
    cost_curve,
    load_ohlcv,
    load_price_volume,
)
from .data import synthetic_bounce, synthetic_kyle, synthetic_order_flow
from .impact import (
    almgren_chriss_impact,
    implementation_shortfall,
    kyle_lambda,
    linear_impact,
    square_root_impact,
    walk_book,
)
from .metrics import (
    amihud_illiquidity,
    book_imbalance,
    book_summary,
    depth_weighted_price,
    effective_spread,
    effective_spread_bps,
    microprice,
    quoted_spread,
    realized_spread,
    relative_spread,
    roll_spread,
)
from .optimal import (
    ExecutionSchedule,
    almgren_chriss,
    efficient_frontier,
    twap,
    vwap,
)

__version__ = "0.1.0"

__all__ = [
    "ExecutionSchedule",
    "CalibrationResult",
    "Fill",
    "LimitOrderBook",
    "Order",
    "__version__",
    "almgren_chriss",
    "almgren_chriss_impact",
    "amihud_illiquidity",
    "book_imbalance",
    "book_summary",
    "build_book",
    "calibrate",
    "calibrate_csv",
    "cost_curve",
    "depth_weighted_price",
    "effective_spread",
    "effective_spread_bps",
    "efficient_frontier",
    "implementation_shortfall",
    "kyle_lambda",
    "linear_impact",
    "load_ohlcv",
    "load_price_volume",
    "microprice",
    "quoted_spread",
    "realized_spread",
    "relative_spread",
    "roll_spread",
    "square_root_impact",
    "synthetic_bounce",
    "synthetic_kyle",
    "synthetic_order_flow",
    "twap",
    "vwap",
    "walk_book",
]
