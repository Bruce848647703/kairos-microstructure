"""示例 1：订单簿、走簿成本与平方根冲击。

运行：python examples/01_order_book.py
"""
from __future__ import annotations

from kairos_microstructure import book_summary, build_book, square_root_impact, walk_book


def main() -> None:
    book = build_book(100.0, levels=10, level_size=100, imbalance=0.2)
    summary = book_summary(book)
    print(f"mid={summary['mid']:.3f} spread={summary['relative_spread']*1e4:.2f}bp "
          f"imbalance={summary['imbalance']:+.2f}")

    print(f"\n{'数量':>8}{'均价':>10}{'冲击(bp)':>10}{'平方根(bp)':>12}{'档位':>6}")
    for qty in (100, 500, 1000, 2000):
        fresh = build_book(100.0, levels=20, level_size=100)
        result = walk_book(fresh, "buy", qty)
        model = square_root_impact(0.02, qty, 1_000_000) * 1e4
        print(f"{qty:>8}{result['average_price']:>10.4f}{result['shortfall_bps']:>10.2f}"
              f"{model:>12.2f}{result['levels_consumed']:>6}")


if __name__ == "__main__":
    main()
