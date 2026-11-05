"""订单簿测试。"""
from __future__ import annotations

import pytest

from kairos_microstructure.book import LimitOrderBook, build_book


def test_empty_book():
    book = LimitOrderBook()
    assert book.best_bid is None and book.best_ask is None
    assert book.mid is None and book.spread is None
    assert book.depth("buy") == 0.0
    assert book.snapshot()["bids"] == []


def test_add_limits_and_top_of_book():
    book = LimitOrderBook(tick=0.01)
    book.add_limit("buy", 99.99, 100)
    book.add_limit("buy", 99.98, 200)
    book.add_limit("sell", 100.01, 150)
    assert book.best_bid == pytest.approx(99.99)
    assert book.best_ask == pytest.approx(100.01)
    assert book.mid == pytest.approx(100.0)
    assert book.spread == pytest.approx(0.02)
    assert book.depth("buy") == pytest.approx(300)
    assert book.depth("sell") == pytest.approx(150)


def test_price_is_rounded_to_tick():
    book = LimitOrderBook(tick=0.01)
    order = book.add_limit("buy", 99.994, 10)
    assert order.price == pytest.approx(99.99)


def test_crossing_limit_executes_then_rests():
    book = LimitOrderBook()
    book.add_limit("sell", 100.01, 60)
    order = book.add_limit("buy", 100.02, 100)
    assert order.filled_qty == pytest.approx(60)
    assert order.remaining == pytest.approx(40)
    # 剩余 40 挂在 100.02（成为新的 best bid）
    assert book.best_bid == pytest.approx(100.02)
    assert book.best_ask is None
    assert len(book.trades) == 1


def test_market_order_walks_levels():
    book = build_book(100.0, tick=0.01, levels=5, level_size=100)
    fills = book.market_order("buy", 250)
    assert sum(f.qty for f in fills) == pytest.approx(250)
    prices = [f.price for f in fills]
    assert prices == sorted(prices)                 # 从最低卖价往上吃
    assert prices[0] == pytest.approx(100.01)


def test_market_order_partial_when_liquidity_short():
    book = build_book(100.0, levels=1, level_size=10)
    fills = book.market_order("sell", 100)
    assert sum(f.qty for f in fills) == pytest.approx(10)
    assert book.best_bid is None                    # 唯一一档被吃光


def test_cancel_removes_order():
    book = LimitOrderBook()
    order = book.add_limit("buy", 99.0, 100)
    assert book.depth("buy") == pytest.approx(100)
    book.cancel(order.id)
    assert book.depth("buy") == 0.0
    with pytest.raises(ValueError):
        book.cancel(order.id)


def test_queue_ahead_is_fifo():
    book = LimitOrderBook()
    first = book.add_limit("buy", 99.0, 100)
    second = book.add_limit("buy", 99.0, 200)
    assert book.queue_ahead(first.id) == pytest.approx(0.0)
    assert book.queue_ahead(second.id) == pytest.approx(100.0)


def test_build_book_imbalance_sign():
    positive = build_book(100.0, levels=5, level_size=100, imbalance=0.5)
    assert positive.depth("buy", 5) > positive.depth("sell", 5)
    negative = build_book(100.0, levels=5, level_size=100, imbalance=-0.5)
    assert negative.depth("buy", 5) < negative.depth("sell", 5)


def test_validation():
    book = LimitOrderBook()
    with pytest.raises(ValueError):
        book.add_limit("hold", 100, 10)
    with pytest.raises(ValueError):
        book.add_limit("buy", 0, 10)
    with pytest.raises(ValueError):
        book.add_limit("buy", 100, 0)
    with pytest.raises(ValueError):
        book.market_order("buy", 0)
    with pytest.raises(ValueError):
        build_book(100.0, imbalance=1.5)

def test_modify_keeps_queue_position():
    book = LimitOrderBook()
    first = book.add_limit("buy", 99.0, 100)
    second = book.add_limit("buy", 99.0, 300)
    book.modify(second.id, 500)
    assert second.remaining == pytest.approx(500)
    assert book.queue_ahead(second.id) == pytest.approx(first.remaining)   # 仍在 first 后面
    assert book.level_qty("buy", 99.0) == pytest.approx(600)


def test_modify_to_zero_cancels_and_rejects_below_filled():
    book = LimitOrderBook()
    order = book.add_limit("buy", 99.0, 100)
    book.modify(order.id, 40)
    assert order.remaining == pytest.approx(40)

    book.modify(order.id, 0)                   # 数量 0 视为撤单
    assert book.depth("buy") == 0.0

    other = LimitOrderBook()
    resting = other.add_limit("sell", 101.0, 100)
    other.market_order("buy", 40)              # 吃掉 40，剩 60
    with pytest.raises(ValueError, match="必须大于已成交量"):
        other.modify(resting.id, 40)           # 不能改到已成交量以下
