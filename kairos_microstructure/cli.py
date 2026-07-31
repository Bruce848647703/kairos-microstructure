"""命令行入口：``kairos-microstructure`` / ``python -m kairos_microstructure``。"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .book import build_book
from .data import synthetic_bounce, synthetic_kyle
from .impact import (
    almgren_chriss_impact,
    kyle_lambda,
    square_root_impact,
    walk_book,
)
from .metrics import book_summary, roll_spread
from .optimal import almgren_chriss, efficient_frontier, twap


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kairos-microstructure",
        description="市场微观结构：订单簿、平方根冲击、Almgren-Chriss 最优执行与微观指标。")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="完整演示（默认）")

    book = sub.add_parser("book", help="打印合成订单簿与指标")
    book.add_argument("--mid", type=float, default=100.0)
    book.add_argument("--levels", type=int, default=10)
    book.add_argument("--size", type=float, default=100.0)
    book.add_argument("--imbalance", type=float, default=0.0)

    execute = sub.add_parser("execute", help="用市价单走簿，计算执行短缺")
    execute.add_argument("--qty", type=float, default=1000.0)
    execute.add_argument("--side", default="buy", choices=("buy", "sell"))
    execute.add_argument("--mid", type=float, default=100.0)

    cal = sub.add_parser("calibrate", help="用真实日线 CSV 校准冲击参数")
    cal.add_argument("--data", required=True, help="CSV 路径（date,close,volume）")
    cal.add_argument("--c", type=float, default=0.5, help="平方根系数 c")
    cal.add_argument("--qref", type=float, default=0.1, help="参考参与率 Q_ref")

    optimal = sub.add_parser("optimal", help="Almgren-Chriss 执行轨迹")
    optimal.add_argument("--shares", type=float, default=100000.0)
    optimal.add_argument("--horizon", type=float, default=1.0)
    optimal.add_argument("--slices", type=int, default=10)
    optimal.add_argument("--risk", type=float, default=1e-5)
    optimal.add_argument("--sigma", type=float, default=0.02)
    optimal.add_argument("--eta", type=float, default=0.1)
    optimal.add_argument("--gamma", type=float, default=0.01)

    return parser


def _cmd_book(args: argparse.Namespace) -> int:
    book = build_book(args.mid, levels=args.levels, level_size=args.size,
                      imbalance=args.imbalance)
    summary = book_summary(book, levels=5)
    print(f"mid={summary['mid']:.4f}  spread={summary['spread']:.4f} "
          f"({summary['relative_spread']*1e4:.2f}bp)  imbalance={summary['imbalance']:+.3f}")
    print(f"{'价格':>10}{'数量':>10}{'订单数':>8}")
    snap = book.snapshot(levels=5)
    for row in reversed(snap["asks"]):
        print(f"{row['price']:>10.3f}{row['qty']:>10.0f}{row['orders']:>8}")
    print("-" * 28)
    for row in snap["bids"]:
        print(f"{row['price']:>10.3f}{row['qty']:>10.0f}{row['orders']:>8}")
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    book = build_book(args.mid, levels=10, level_size=100.0)
    result = walk_book(book, args.side, args.qty)
    print(f"{args.side} {args.qty:.0f} 股  成交 {result['filled']:.0f}  "
          f"均价 {result['average_price']:.4f}  冲击 {result['shortfall_bps']:+.2f}bp  "
          f"消耗 {result['levels_consumed']} 档")
    if result["unfilled"] > 0:
        print(f"  流动性不足，未成交 {result['unfilled']:.0f}")
    return 0


def _cmd_optimal(args: argparse.Namespace) -> int:
    schedule = almgren_chriss(
        args.shares, args.horizon, args.slices, sigma=args.sigma, eta=args.eta,
        gamma=args.gamma, risk_aversion=args.risk)
    print(f"X={args.shares:,.0f}  T={args.horizon:g}  N={args.slices}  λ={args.risk:g}")
    print(f"{'slice':>7}{'剩余':>12}{'本笔':>12}")
    for i, (holding, trade) in enumerate(zip(schedule.holdings[1:], schedule.trades), 1):
        print(f"{i:>7}{holding:>12,.0f}{trade:>12,.0f}")
    print(f"\n预期成本 {schedule.expected_cost:,.2f}（{schedule.cost_bps:.2f}bp）  "
          f"方差 {schedule.variance:,.2f}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import calibrate_csv, cost_curve

    result = calibrate_csv(args.data, sqrt_coefficient=args.c,
                           reference_participation=args.qref)
    print(f"样本 {result.n_obs} 行  中位价 {result.mid_price:,.2f}")
    print("-" * 52)
    print(f"  年化波动 σ      {result.sigma:>8.2%}")
    print(f"  ADV（中位）     {result.adv:>8,.0f} 股")
    print(f"  成交额（中位）  {result.dollar_adv:>8,.0f}")
    print(f"  Amihud          {result.amihud:>8.3e}")
    print(f"  Roll 价差       {result.roll_spread:>8.4f}（{result.roll_spread_bps:.2f}bp）")
    if result.sigma_intraday is not None:
        print(f"  日频波动(GK)    {result.sigma_intraday:>8.2%}")
        print(f"  OHLC 价差       {result.spread_ohlc_bps:>8.2f}bp（{result.spread_source}）")
    print(f"  Kyle λ          {result.kyle_lambda:>8.3e}（每 1×ADV）")
    print(f"  ── 校准参数 ──")
    print(f"  永久冲击 γ      {result.gamma:>8.4f}")
    print(f"  临时冲击 η      {result.eta:>8.4f}")
    print()
    print(f"{'参与率':>8}{'股数':>12}{'成本(bp)':>10}{'风险(bp)':>10}")
    for row in cost_curve(result, horizon=1.0):
        print(f"{row['participation']:>8.0%}{row['shares']:>12,.0f}"
              f"{row['cost_bps']:>10.2f}{row['risk_bps']:>10.2f}")
    print()
    for note in result.assumptions:
        print(f"  · {note}")
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print("=" * 68)
    print("1) 订单簿快照与指标")
    print("=" * 68)
    book = build_book(100.0, levels=10, level_size=100.0, imbalance=0.2)
    summary = book_summary(book)
    print(f"  best_bid={summary['best_bid']:.2f} best_ask={summary['best_ask']:.2f} "
          f"mid={summary['mid']:.3f} spread={summary['relative_spread']*1e4:.2f}bp")
    print(f"  bid_depth={summary['bid_depth']:.0f} ask_depth={summary['ask_depth']:.0f} "
          f"imbalance={summary['imbalance']:+.3f} microprice={summary['microprice']:.4f}")

    print()
    print("=" * 68)
    print("2) 市价单走簿 vs 平方根冲击")
    print("=" * 68)
    print(f"  {'数量':>10}{'成交均价':>12}{'冲击(bp)':>12}{'平方根(bp)':>12}{'消耗档位':>10}")
    for qty in (100, 500, 1000, 2000):
        fresh = build_book(100.0, levels=20, level_size=100.0)
        result = walk_book(fresh, "buy", qty)
        model = square_root_impact(0.02, qty, 1_000_000.0, coefficient=0.5) * 1e4
        print(f"  {qty:>10}{result['average_price']:>12.4f}"
              f"{result['shortfall_bps']:>12.2f}{model:>12.2f}"
              f"{result['levels_consumed']:>10}")
    print(f"  参考：Almgren-Chriss 永久+临时冲击 = "
          f"{almgren_chriss_impact(1000, 1_000_000, participation=0.1)['total']*1e4:.2f}bp"
          f"（ADV=1e6）")

    print()
    print("=" * 68)
    print("3) Almgren-Chriss：风险厌恶如何改变执行轨迹")
    print("=" * 68)
    print(f"  {'λ':>8}{'首笔':>12}{'成本(bp)':>12}{'风险(bp)':>12}")
    for lam in (0.0, 1.0, 5.0, 25.0):
        sched = almgren_chriss(100_000, 1.0, 10, sigma=0.02, eta=0.01,
                               gamma=0.001, risk_aversion=lam, adv=1e6, price=100.0)
        print(f"  {lam:>8.0f}{sched.trades[0]:>12,.0f}{sched.cost_bps:>12.2f}{sched.std_bps:>12.2f}")
    base = twap(100_000, 1.0, 10, sigma=0.02, eta=0.01, gamma=0.001,
                adv=1e6, price=100.0)
    print(f"  对照 TWAP：首笔 {base.trades[0]:,.0f}  成本 {base.cost_bps:.2f}bp  "
          f"风险 {base.std_bps:.2f}bp")

    print()
    print("=" * 68)
    print("4) 微观指标还原：Roll 价差 / Kyle λ")
    print("=" * 68)
    bounce = synthetic_bounce(n=20000)
    est = roll_spread(bounce["price"])
    print(f"  Roll 价差估计 {est:.5f}  真实 {bounce['true_spread']:.5f}  "
          f"误差 {abs(est-bounce['true_spread'])/bounce['true_spread']:.2%}")
    kyle = synthetic_kyle(n=5000, lam=2e-5)
    lam = kyle_lambda(kyle["price_changes"], kyle["signed_volume"])
    print(f"  Kyle λ 估计 {lam:.3e}  真实 {kyle['true_lambda']:.3e}  "
          f"误差 {abs(lam-kyle['true_lambda'])/kyle['true_lambda']:.2%}")

    print()
    print("=" * 68)
    print("5) 有效前沿（成本 vs 方差）")
    print("=" * 68)
    for point in efficient_frontier(100_000, 1.0, 10, price=100.0):
        print(f"  λ={point['risk_aversion']:>6.0f}  成本={point['cost_bps']:>6.2f}bp  "
              f"风险={point['std_bps']:>6.2f}bp")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not argv_list or argv_list[0].startswith("-"):
        args = parser.parse_args(["demo"] + argv_list)
    else:
        args = parser.parse_args(argv_list)
    command = args.command or "demo"
    return {
        "demo": _cmd_demo,
        "book": _cmd_book,
        "execute": _cmd_execute,
        "optimal": _cmd_optimal,
        "calibrate": _cmd_calibrate,
    }[command](args)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())