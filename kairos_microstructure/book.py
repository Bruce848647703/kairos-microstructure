"""限价订单簿（LOB）：限价/市价单、FIFO 队列、撮合与快照。

设计
----
- 价格 -> 该价位的**订单队列**（按到达顺序），因此可以计算排队位置 ``queue_ahead``；
- 限价单如果穿越对手价，会先吃对手盘、剩余量挂在簿上（与真实撮合一致）；
- 市价单逐档吃单，返回逐笔成交（``Fill``），流动性不足时按可用量部分成交。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SIDES = ("buy", "sell")


def _check_side(side: str) -> str:
    side = str(side).lower()
    if side not in SIDES:
        raise ValueError(f"side 必须是 buy / sell，收到 {side!r}")
    return side


@dataclass
class Order:
    """一笔限价单。"""

    id: int
    side: str
    price: float
    qty: float
    timestamp: float = 0.0
    filled_qty: float = 0.0

    @property
    def remaining(self) -> float:
        return self.qty - self.filled_qty

    @property
    def is_filled(self) -> bool:
        return self.remaining <= 1e-12

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "side": self.side, "price": self.price,
                "qty": self.qty, "filled_qty": self.filled_qty,
                "remaining": self.remaining, "timestamp": self.timestamp}


@dataclass
class Fill:
    """一笔成交（taker 视角）。"""

    order_id: int
    side: str
    price: float
    qty: float
    timestamp: float = 0.0
    maker_id: Optional[int] = None

    @property
    def notional(self) -> float:
        return self.price * self.qty

    def to_dict(self) -> Dict[str, Any]:
        return {"order_id": self.order_id, "side": self.side, "price": self.price,
                "qty": self.qty, "timestamp": self.timestamp,
                "maker_id": self.maker_id}


@dataclass
class LimitOrderBook:
    """价格档位 + FIFO 队列的限价订单簿。"""

    tick: float = 0.01
    bids: Dict[float, List[Order]] = field(default_factory=dict)
    asks: Dict[float, List[Order]] = field(default_factory=dict)
    trades: List[Fill] = field(default_factory=list)
    _next_id: int = 1

    def _round(self, price: float) -> float:
        return round(round(float(price) / self.tick) * self.tick, 10)

    # ------------------------------------------------------------------ 查询
    @property
    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    @property
    def spread(self) -> Optional[float]:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def level_qty(self, side: str, price: float) -> float:
        side = _check_side(side)
        book = self.bids if side == "buy" else self.asks
        return float(sum(o.remaining for o in book.get(self._round(price), [])))

    def depth(self, side: str, levels: int = 5) -> float:
        """前 ``levels`` 档的累计挂单量。"""
        side = _check_side(side)
        book = self.bids if side == "buy" else self.asks
        if not book:
            return 0.0
        prices = sorted(book, reverse=(side == "buy"))[: int(levels)]
        return float(sum(sum(o.remaining for o in book[p]) for p in prices))

    def queue_ahead(self, order_id: int) -> float:
        """同价位、比自己更早到达的挂单量（排队位置）。"""
        order = self._find(order_id)
        book = self.bids if order.side == "buy" else self.asks
        ahead = 0.0
        for other in book.get(order.price, []):
            if other.id == order_id:
                break
            ahead += other.remaining
        return float(ahead)

    def snapshot(self, levels: int = 5) -> Dict[str, Any]:
        def side_view(book: Dict[float, List[Order]], reverse: bool):
            prices = sorted(book, reverse=reverse)[: int(levels)]
            return [{"price": p, "qty": float(sum(o.remaining for o in book[p])),
                     "orders": len(book[p])} for p in prices]
        return {"bids": side_view(self.bids, True),
                "asks": side_view(self.asks, False),
                "best_bid": self.best_bid, "best_ask": self.best_ask,
                "mid": self.mid, "spread": self.spread}

    # ------------------------------------------------------------------ 下单
    def add_limit(self, side: str, price: float, qty: float,
                  timestamp: float = 0.0) -> Order:
        """下限价单；能立即成交的先成交，余量挂簿。"""
        side = _check_side(side)
        price = self._round(price)
        qty = float(qty)
        if price <= 0:
            raise ValueError("price 必须为正")
        if qty <= 0:
            raise ValueError("qty 必须为正")

        order = Order(id=self._next_id, side=side, price=price, qty=qty,
                      timestamp=timestamp)
        self._next_id += 1

        # 穿越成交
        while order.remaining > 1e-12:
            best = self.best_ask if side == "buy" else self.best_bid
            if best is None:
                break
            crosses = (best <= price) if side == "buy" else (best >= price)
            if not crosses:
                break
            self._consume(order, best, timestamp)

        if order.remaining > 1e-12:
            book = self.bids if side == "buy" else self.asks
            book.setdefault(price, []).append(order)
        return order

    def market_order(self, side: str, qty: float,
                     timestamp: float = 0.0) -> List[Fill]:
        """市价单：逐档吃对手盘，返回成交列表。"""
        side = _check_side(side)
        qty = float(qty)
        if qty <= 0:
            raise ValueError("qty 必须为正")
        order = Order(id=self._next_id, side=side,
                      price=float("inf") if side == "buy" else 0.0,
                      qty=qty, timestamp=timestamp)
        self._next_id += 1
        while order.remaining > 1e-12:
            best = self.best_ask if side == "buy" else self.best_bid
            if best is None:
                break
            if not self._consume(order, best, timestamp):
                break
        return [f for f in self.trades if f.order_id == order.id]

    def modify(self, order_id: int, qty: float) -> Order:
        """改量（cancel/replace）：保持原队列位置，``qty`` 小于等于 0 视为撤单。

        真实交易所改价会丢失队列优先级，改量一般保留；这里只支持改量，
        价格变动请撤单重下，语义更清楚。
        """
        order = self._find(order_id)
        qty = float(qty)
        if qty <= 0:
            return self.cancel(order_id)
        if qty <= order.filled_qty:
            raise ValueError(
                f"新数量 {qty} 必须大于已成交量 {order.filled_qty}")
        order.qty = qty
        return order

    def cancel(self, order_id: int) -> Order:
        """撤单并返回该订单。"""
        order = self._find(order_id)
        book = self.bids if order.side == "buy" else self.asks
        queue = book.get(order.price, [])
        book[order.price] = [o for o in queue if o.id != order_id]
        if not book[order.price]:
            del book[order.price]
        return order

    # ------------------------------------------------------------------ 内部
    def _find(self, order_id: int) -> Order:
        for book in (self.bids, self.asks):
            for queue in book.values():
                for order in queue:
                    if order.id == order_id:
                        return order
        raise ValueError(f"订单 {order_id} 不存在或已全部成交")

    def _consume(self, taker: Order, price: float, timestamp: float) -> bool:
        """在 ``price`` 档位吃掉 taker 的剩余量（FIFO）。"""
        book = self.asks if taker.side == "buy" else self.bids
        queue = book.get(price)
        if not queue:
            return False
        filled_any = False
        while queue and taker.remaining > 1e-12:
            maker = queue[0]
            trade_qty = min(maker.remaining, taker.remaining)
            maker.filled_qty += trade_qty
            taker.filled_qty += trade_qty
            self.trades.append(Fill(
                order_id=taker.id, side=taker.side, price=price, qty=trade_qty,
                timestamp=timestamp, maker_id=maker.id))
            filled_any = True
            if maker.is_filled:
                queue.pop(0)
        if not queue:
            del book[price]
        return filled_any


def build_book(mid: float = 100.0, tick: float = 0.01, levels: int = 10,
               level_size: float = 100.0, imbalance: float = 0.0) -> LimitOrderBook:
    """构造一个对称/带偏斜的静态订单簿（研究用）。"""
    if mid <= 0 or levels < 1 or level_size <= 0:
        raise ValueError("mid/level_size 必须为正，levels 必须 >= 1")
    if not -1.0 < imbalance < 1.0:
        raise ValueError("imbalance 必须在 (-1, 1) 内")
    book = LimitOrderBook(tick=tick)
    mid = round(float(mid) / tick) * tick          # 对齐到 tick 网格
    for i in range(1, int(levels) + 1):
        size = level_size * (1.0 + 0.25 * (i - 1))
        book.add_limit("buy", mid - i * tick, size * (1.0 + imbalance))
        book.add_limit("sell", mid + i * tick, size * (1.0 - imbalance))
    return book


__all__ = ["Fill", "LimitOrderBook", "Order", "SIDES", "build_book"]