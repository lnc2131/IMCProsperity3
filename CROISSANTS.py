# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 02:20:57 2025

@author: iccha
"""

#!/usr/bin/env python3
"""
CROISSANTS-only strategy that:

1) Keeps a static fair_value = 3000.
2) Computes a dynamic_fv from recent mid prices.
3) Blends them (50/50) into a final fair_value for trading decisions.
   You can change the weighting or implement your own blending logic.

This means the bot is partially anchored to the old 3000 value while still
following the market if it drifts significantly from 3000.
"""

import json
import math
import sys
from collections import deque
from typing import Any, List, Dict, Deque

try:
    from datamodel import (
        OrderDepth,
        TradingState,
        Order,
        Trade,
        Observation,
        ProsperityEncoder,
        Listing
    )
except ImportError:
    class Listing:
        def __init__(self, product, denomination):
            self.product = product
            self.denomination = denomination

    class OrderDepth:
        buy_orders: Dict[int, int] = {}
        sell_orders: Dict[int, int] = {}

    class TradingState:
        timestamp: int
        listings: Dict[str, Any]
        order_depths: Dict[str, OrderDepth]
        own_trades: Dict[str, List[Any]]
        market_trades: Dict[str, List[Any]]
        position: Dict[str, int]
        observations: Any
        traderData: str

    class Order:
        def __init__(self, symbol: str, price: int, quantity: int):
            self.symbol = symbol
            self.price = price
            self.quantity = quantity

    class Trade:
        pass

    class Observation:
        conversionObservations: Dict[str, Any] = {}
        plainValueObservations: Dict[str, Any] = {}

    class ProsperityEncoder(json.JSONEncoder):
        pass

# =============================================================================
# Logger
# =============================================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: "TradingState",
        orders: Dict[str, List["Order"]],
        conversions: int,
        trader_data: str
    ) -> None:
        # Construct a minimal compressed state placeholder
        compressed_state = [
            state.timestamp if state else 0,
            trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()] if state and state.listings else {},
            {s: [d.buy_orders, d.sell_orders] for s, d in state.order_depths.items()} if state and state.order_depths else {},
            [],  # own_trades placeholder
            [],  # market_trades placeholder
            state.position if state else {},
            {}   # observations placeholder
        ]
        compressed_orders = []
        for symbol, order_list in orders.items():
            for o in order_list:
                compressed_orders.append([o.symbol, o.price, o.quantity])
        output_array = [
            compressed_state,
            compressed_orders,
            conversions,
            trader_data,
            self.logs
        ]
        sys.stdout.write(json.dumps(output_array, cls=ProsperityEncoder, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        self.logs = ""

logger = Logger()

# =============================================================================
# Constants & Params
# =============================================================================
class Product:
    CROISSANTS = "CROISSANTS"

# Parameters for CROISSANTS strategy
PARAMS = {
    Product.CROISSANTS: {
        "static_fv": 4250,            # The static baseline fair value
        "window_size": 5000,            # Rolling window size for dynamic fair value calculation
        "initial_dynamic_fv": 4250,   # Initial dynamic fair value if no data exists yet
        "take_width": 1,              # Margin to trigger liquidity-taking orders
        "clear_width": 2,             # Width around fair value for clearing positions
        "volume_limit": 250,          # Maximum net position allowed
        "blend_factor": 0.5,          # 50% weighting for dynamic vs static fair value
    },
}

# =============================================================================
# Trader
# =============================================================================
class Trader:
    """
    CROISSANTS strategy that:

    - Maintains a rolling average (dynamic_fv) of recent mid prices.
    - Also uses a static_fv of 3000.
    - Blends them to form final_fv:
        final_fv = (1 - blend_factor)*static_fv + (blend_factor)*dynamic_fv
    - Uses final_fv for buy/sell/clear logic.
    """
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        window_size = self.params[Product.CROISSANTS]["window_size"]
        self.mid_prices: Deque[float] = deque(maxlen=window_size)
        self.dynamic_fv = self.params[Product.CROISSANTS]["initial_dynamic_fv"]
        # For position limit
        self.LIMIT = {
            Product.CROISSANTS: self.params[Product.CROISSANTS]["volume_limit"],
        }

    def update_dynamic_fv(self, best_bid: float, best_ask: float) -> float:
        """
        Computes a new dynamic fair value using the rolling average of mid prices.
        """
        mid_price = (best_bid + best_ask) / 2
        self.mid_prices.append(mid_price)
        if len(self.mid_prices) == 0:
            return self.dynamic_fv
        avg = sum(self.mid_prices) / len(self.mid_prices)
        self.dynamic_fv = avg
        return avg

    def blend_fair_values(self, static_fv: float, dynamic_fv: float) -> float:
        """
        Computes the final fair value as a blend of static_fv and dynamic_fv.
        For example, with blend_factor=0.5:
           final_fv = 0.5 * static_fv + 0.5 * dynamic_fv
        """
        alpha = self.params[Product.CROISSANTS]["blend_factor"]
        final_fv = (1 - alpha) * static_fv + alpha * dynamic_fv
        return final_fv

    def make_croissants_orders(self, order_depth, fair_value: float, position: int) -> List["Order"]:
        """
        Trading logic (same as before):
         - Buy if best_ask < fair_value - take_width.
         - Sell if best_bid > fair_value + take_width.
         - Clear position (flatten) if near fair_value.
        """
        orders: List[Order] = []
        take_width = self.params[Product.CROISSANTS]["take_width"]
        clear_width = self.params[Product.CROISSANTS]["clear_width"]
        vol_limit = self.params[Product.CROISSANTS]["volume_limit"]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            logger.print("[CROISSANTS] Empty order book, skipping.")
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_ask_qty = -order_depth.sell_orders[best_ask]
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_qty = order_depth.buy_orders[best_bid]

        # Buy if the best ask is low enough
        if best_ask <= fair_value - take_width:
            buy_qty = min(best_ask_qty, vol_limit - position)
            if buy_qty > 0:
                orders.append(Order(Product.CROISSANTS, best_ask, buy_qty))
                logger.print(f"[CROISSANTS] BUY {buy_qty} @ {best_ask}")

        # Sell if the best bid is high enough
        if best_bid >= fair_value + take_width:
            sell_qty = min(best_bid_qty, vol_limit + position)
            if sell_qty > 0:
                orders.append(Order(Product.CROISSANTS, best_bid, -sell_qty))
                logger.print(f"[CROISSANTS] SELL {sell_qty} @ {best_bid}")

        # Clear positions if near fair_value
        if position > 0:
            possible_bids = [p for p in order_depth.buy_orders if abs(p - fair_value) <= clear_width]
            if possible_bids:
                best_clear_bid = max(possible_bids)
                avail_qty = order_depth.buy_orders[best_clear_bid]
                sellable = min(avail_qty, position)
                if sellable > 0:
                    orders.append(Order(Product.CROISSANTS, best_clear_bid, -sellable))
                    logger.print(f"[CROISSANTS] Clearing SELL {sellable} @ {best_clear_bid}")
        elif position < 0:
            possible_asks = [p for p in order_depth.sell_orders if abs(p - fair_value) <= clear_width]
            if possible_asks:
                best_clear_ask = min(possible_asks)
                avail_qty = -order_depth.sell_orders[best_clear_ask]
                buyable = min(avail_qty, -position)
                if buyable > 0:
                    orders.append(Order(Product.CROISSANTS, best_clear_ask, buyable))
                    logger.print(f"[CROISSANTS] Clearing BUY {buyable} @ {best_clear_ask}")

        return orders

    def run(self, state: "TradingState") -> tuple[Dict[str, List["Order"]], int, str]:
        """
        Called every tick. Returns (orders, conversions, trader_data).
        Computes a dynamic fair value, blends it with a static value, and places orders accordingly.
        """
        if not state or not state.order_depths:
            logger.print("[CROISSANTS] No data in state, returning empty.")
            empty_result: Dict[str, List[Order]] = {}
            logger.flush(state, empty_result, 0, "{}")
            return empty_result, 0, "{}"

        result: Dict[str, List[Order]] = {}
        conversions = 0

        if Product.CROISSANTS in state.order_depths:
            position = state.position.get(Product.CROISSANTS, 0)
            od = state.order_depths[Product.CROISSANTS]

            static_fv = self.params[Product.CROISSANTS]["static_fv"]

            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders.keys())
                best_ask = min(od.sell_orders.keys())
                new_dynamic = self.update_dynamic_fv(best_bid, best_ask)
                final_fv = self.blend_fair_values(static_fv, new_dynamic)
            else:
                final_fv = self.blend_fair_values(static_fv, self.dynamic_fv)

            orders_list = self.make_croissants_orders(od, final_fv, position)
            result[Product.CROISSANTS] = orders_list

            logger.print(f"[CROISSANTS] final_fv={final_fv:.2f}, dynamic_fv={self.dynamic_fv:.2f}, static_fv={static_fv}")

        trader_data = json.dumps({
            "positions": state.position,
            "notes": "Blended static+dynamic fair value for CROISSANTS",
            "dynamic_fv": self.dynamic_fv
        })

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
