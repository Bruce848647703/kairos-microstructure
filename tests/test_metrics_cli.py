"""微观指标与命令行测试。"""
from __future__ import annotations

import pytest

from kairos_microstructure.book import build_book
from kairos_microstructure.cli import main
from kairos_microstructure.data import synthetic_bounce
from kairos_microstructure.impact import walk_book
from kairos_microstructure.metrics import (
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


def test_spread_metrics():
    assert quoted_spread(99.99, 100.01) == pytest.approx(0.02)
    assert relative_spread(99.99, 100.01) == pytest.approx(0.02 / 100.0)
    with pytest.raises(ValueError):
        quoted_spread(100.02, 100.01)
    with pytest.raises(ValueError):
        relative_spread(-1, 1)


def test_imbalance_and_microprice():
    assert book_imbalance(600, 400) == pytest.approx(0.2)
    assert book_imbalance(0, 0) == 0.0
    # 买盘更多 -> 微价格偏向卖价
    assert microprice(99.99, 100.01, 600, 400) > 100.0
    assert microprice(99.99, 100.01, 0, 0) == pytest.approx(100.0)


def test_roll_spread_recovers_true_spread():
    data = synthetic_bounce(n=30000, half_spread=0.01, mid_sigma=0.0005, seed=7)
    estimated = roll_spread(data["price"])
    assert estimated == pytest.approx(data["true_spread"], rel=0.05)
    assert roll_spread([1.0, 2.0, 3.0]) >= 0.0


def test_amihud_illiquidity():
    value = amihud_illiquidity([0.01, -0.02], [1e6, 0.0])
    assert value == pytest.approx(0.01 / 1e6)
    with pytest.raises(ValueError):
        amihud_illiquidity([0.01, 0.02], [1e6])


def test_effective_and_realized_spread():
    assert effective_spread(100.02, 100.0) == pytest.approx(0.04)
    assert effective_spread_bps(100.02, 100.0) == pytest.approx(4.0)
    # 做市商卖出（taker 买）后中间价不动 -> 已实现价差为正
    assert realized_spread(100.02, 100.0, "buy") == pytest.approx(0.02)
    assert realized_spread(99.98, 100.0, "sell") == pytest.approx(0.02)
    with pytest.raises(ValueError):
        realized_spread(100, 100, "hold")


def test_depth_weighted_price_matches_walk():
    book = build_book(100.0, levels=20, level_size=100)
    quoted = depth_weighted_price(book, "buy", 250)
    result = walk_book(book, "buy", 250)
    assert quoted == pytest.approx(result["average_price"])
    assert depth_weighted_price(build_book(100.0, levels=1, level_size=10), "buy", 1000) is None


def test_book_summary():
    book = build_book(100.0, levels=10, level_size=100, imbalance=0.2)
    summary = book_summary(book)
    assert summary["mid"] == pytest.approx(100.0)
    assert summary["spread"] == pytest.approx(0.02)
    assert summary["imbalance"] > 0
    assert summary["bid_depth"] > summary["ask_depth"]
    assert summary["microprice"] > summary["mid"]

    assert book_summary(__import__("kairos_microstructure").LimitOrderBook())["mid"] is None


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "订单簿" in out
    assert "Almgren-Chriss" in out
    assert "Roll" in out


def test_cli_book_execute_optimal(capsys):
    assert main(["book", "--mid", "100", "--levels", "5"]) == 0
    out = capsys.readouterr().out
    assert "mid=" in out

    assert main(["execute", "--qty", "300", "--side", "buy"]) == 0
    out = capsys.readouterr().out
    assert "均价" in out and "冲击" in out

    assert main(["optimal", "--shares", "100000", "--risk", "25"]) == 0
    out = capsys.readouterr().out
    assert "预期成本" in out


def test_cli_version_and_top_level_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("kairos-microstructure ")

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert all(name in out for name in ("demo", "book", "execute", "optimal", "calibrate"))