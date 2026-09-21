"""示例 2：Almgren-Chriss —— 风险厌恶如何改变执行轨迹与成本。

运行：python examples/02_optimal_execution.py
"""
from __future__ import annotations

from kairos_microstructure import almgren_chriss, efficient_frontier, twap

PARAMS = dict(sigma=0.02, eta=0.01, gamma=0.001, adv=1_000_000, price=100.0)


def main() -> None:
    base = twap(100_000, 1.0, 10, **PARAMS)
    print(f"TWAP：首笔 {base.trades[0]:,.0f}  成本 {base.cost_bps:.2f}bp  风险 {base.std_bps:.1f}bp")

    print(f"\n{'λ':>6}{'首笔':>10}{'末笔':>10}{'成本(bp)':>10}{'风险(bp)':>10}")
    for lam in (0.0, 5.0, 25.0, 100.0, 400.0):
        sched = almgren_chriss(100_000, 1.0, 10, risk_aversion=lam, **PARAMS)
        print(f"{lam:>6.0f}{sched.trades[0]:>10,.0f}{sched.trades[-1]:>10,.0f}"
              f"{sched.cost_bps:>10.2f}{sched.std_bps:>10.1f}")

    print("\n有效前沿（成本 vs 风险）：")
    for p in efficient_frontier(100_000, 1.0, 10, **PARAMS):
        print(f"  λ={p['risk_aversion']:>6.0f}  成本 {p['cost_bps']:>5.2f}bp  "
              f"风险 {p['std_bps']:>6.1f}bp")


if __name__ == "__main__":
    main()
