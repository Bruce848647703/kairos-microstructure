"""生成 README 用图（需要 matplotlib：pip install -e ".[plot]"）。"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kairos_microstructure import (  # noqa: E402
    almgren_chriss,
    build_book,
    efficient_frontier,
    twap,
)

PARAMS = dict(sigma=0.02, eta=0.01, gamma=0.001, adv=1_000_000, price=100.0)


def main(out: str = "docs/microstructure.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # 1) 订单簿深度
    book = build_book(100.0, levels=8, level_size=100, imbalance=0.15)
    snap = book.snapshot(levels=8)
    bids_p = [r["price"] for r in snap["bids"]][::-1]
    bids_q = [r["qty"] for r in snap["bids"]][::-1]
    asks_p = [r["price"] for r in snap["asks"]]
    asks_q = [-r["qty"] for r in snap["asks"]]
    axes[0].barh([f"{p:.2f}" for p in bids_p], bids_q, color="#2e86c1", label="bids")
    axes[0].barh([f"{p:.2f}" for p in asks_p], asks_q, color="#c0392b", label="asks")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("resting size (shares)")
    axes[0].set_title("Limit order book")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    # 2) Almgren-Chriss 轨迹
    times = np.linspace(0, 1, 11)
    for lam, color in zip((0.0, 5.0, 25.0, 100.0),
                          ("#1f4e79", "#2e86c1", "#e67e22", "#c0392b")):
        sched = almgren_chriss(100_000, 1.0, 10, risk_aversion=lam, **PARAMS)
        axes[1].plot(times, sched.holdings / sched.shares * 100, "o-", ms=3,
                     color=color, label=f"λ={lam:g}")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("remaining position (%)")
    axes[1].set_title("Almgren-Chriss trajectories")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3)

    # 3) 有效前沿
    points = efficient_frontier(100_000, 1.0, 10, **PARAMS)
    axes[2].plot([p["std_bps"] for p in points], [p["cost_bps"] for p in points],
                 "o-", color="#1f4e79")
    axes[2].set_xlabel("execution risk (bp, std)")
    axes[2].set_ylabel("expected cost (bp)")
    axes[2].set_title("Efficient frontier")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
