"""示例 3：从成交数据还原微观指标（Roll 价差、Kyle λ、Amihud）。

运行：python examples/03_microstructure_metrics.py
"""
from __future__ import annotations

import numpy as np

from kairos_microstructure import (
    amihud_illiquidity,
    kyle_lambda,
    roll_spread,
    synthetic_bounce,
    synthetic_kyle,
)


def main() -> None:
    bounce = synthetic_bounce(n=30000, half_spread=0.01, seed=7)
    est = roll_spread(bounce["price"])
    print(f"Roll 价差：估计 {est:.5f}  真实 {bounce['true_spread']:.5f}  "
          f"误差 {abs(est - bounce['true_spread']) / bounce['true_spread']:.2%}")

    kyle = synthetic_kyle(n=10000, lam=2e-5, noise=0.001, seed=11)
    lam = kyle_lambda(kyle["price_changes"], kyle["signed_volume"])
    print(f"Kyle λ：  估计 {lam:.3e}  真实 {kyle['true_lambda']:.3e}  "
          f"误差 {abs(lam - kyle['true_lambda']) / kyle['true_lambda']:.2%}")

    returns = np.random.default_rng(1).normal(0, 0.01, 500)
    volumes = np.random.default_rng(2).uniform(1e6, 5e6, 500)
    print(f"Amihud：  {amihud_illiquidity(returns, volumes):.3e}（|r|/成交额）")


if __name__ == "__main__":
    main()
