#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 13:47:24 2025

@author: lnc
"""

from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Dict, Tuple, Any
import string
import json
import math
import numpy as np


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    # Adding Picnic Basket and components
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES= "DJEMBES"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 2, 
        "clear_width": 1,
        "volume_limit": 50
    },
    
    Product.KELP: {
        "common_maker": 13, 
        "timespan": 10,
        "adverse_volume": 15,
        "take_width": 1,
        "clear_width": 2,
        "prevent_adverse": True,
        "volume_limit": 50
    },
    
    Product.SQUID_INK: {
        "base_value": 2000, # need to update this to moving average 
        "deviation_threshold": 100,
        "max_order_size": 50,
        "no_trade_zone": 0
    },
    
    # Adding Picnic Basket strategy parameters
    "SPREAD": {
        "default_spread_mean": 80.24,  # This will be dynamically calculated
        "spread_std_window": 30,
        "zscore_threshold": 3,
        "target_position": 60
    }
}

# Define basket component weights
BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1
}

class Trader:
    def __init__(self, params = None):
        if params is None:
            params = PARAMS
        self.params = params 
        
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            # Add position limits for new products
            Product.PICNIC_BASKET1: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60
        }
        
        # Price history for each product
        self.price_history = {
            Product.RAINFOREST_RESIN: [],
            Product.KELP: [],
            Product.SQUID_INK: [],
            Product.PICNIC_BASKET1: [],
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: []
        }
        
        # SQUID_INK specific parameters
        self.squid_position_scale = [
            (-50, 1.0),  # When position is -50, buy scale is 1.0 (max)
            (-25, 0.8),  # When position is -25, buy scale is 0.8
            (0, 0.5),    # When position is 0, scale is 0.5 (middle)
            (25, 0.2),   # When position is 25, buy scale is 0.2 (low)
            (50, 0.0)    # When position is 50, buy scale is 0 (don't buy)
        ]
        
        # Track SQUID_INK price extremes
        self.squid_min_price_seen = float('inf')
        self.squid_max_price_seen = float('-inf')
        
        # Tracking for basket spread trading
        self.spread_history = []
        self.prev_zscore = 0
        
    def get_squid_scaling_factor(self, position, is_buying):
        """Calculate scaling factor based on current position for SQUID_INK"""
        if is_buying:
            # For buying, scale down as position increases
            for pos, scale in reversed(self.squid_position_scale):
                if position >= pos:
                    return scale
            return 0.0
        else:
            # For selling, do the opposite (scale down as position decreases)
            # Convert the scale: 1.0 becomes 0.0, 0.0 becomes 1.0
            for pos, scale in self.squid_position_scale:
                if position <= pos:
                    return 1.0 - scale
            return 0.0
        
    def kelp_fv(self, order_depth: OrderDepth, timespan: int) -> float:
        # Check for empty order book
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        filtered_ask = [price for price in order_depth.sell_orders.keys() 
                       if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["common_maker"]]
        filtered_bid = [price for price in order_depth.buy_orders.keys() 
                       if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["common_maker"]]
        mmask = min(filtered_ask) if filtered_ask else best_ask
        mmbid = max(filtered_bid) if filtered_bid else best_bid 
        mmprice = (mmask + mmbid) / 2 
        
        # Update price history
        mid_price = (best_ask + best_bid) / 2
        self.price_history[Product.KELP].append(mid_price)
        if len(self.price_history[Product.KELP]) > 20:
            self.price_history[Product.KELP].pop(0)
        
        return int(mmprice)
    
    def squidink_fv(self, order_depth: OrderDepth, timespan: int) -> float:
        # Check for empty order book
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        # Calculate mid price
        mid_price = (best_ask + best_bid) / 2
        
        # Update price history
        self.price_history[Product.SQUID_INK].append(mid_price)
        if len(self.price_history[Product.SQUID_INK]) > 20:
            self.price_history[Product.SQUID_INK].pop(0)
        
        return mid_price
    
    def take_best_orders_with_adverse(self, product: str, fair_value: int, take_width: float, orders: List[Order], 
                                    order_depth: OrderDepth, position: int, buy_order_volume: int, 
                                    sell_order_volume: int, adverse_volume: int) -> (int, int):
        position_limit = self.LIMIT[product]
        
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        buy_order_volume += quantity
                        logger.print(f"Taking {product} BUY: {quantity} @ {best_ask}")

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        logger.print(f"Taking {product} SELL: {quantity} @ {best_bid}")

        return buy_order_volume, sell_order_volume
        
    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, 
                   position: int, prevent_adverse: bool = False, adverse_volume: int = 0):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        position_limit = self.LIMIT[product]
        
        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product, fair_value, take_width, orders, order_depth, position, 
                buy_order_volume, sell_order_volume, adverse_volume
            )
        else:
            if len(order_depth.sell_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -1*order_depth.sell_orders[best_ask]
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        logger.print(f"Taking {product} BUY: {quantity} @ {best_ask}")
    
            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        logger.print(f"Taking {product} SELL: {quantity} @ {best_bid}")

        return orders, buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order], 
                           order_depth: OrderDepth, position: int, buy_order_volume: int, 
                           sell_order_volume: int) -> (List[Order], int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
                logger.print(f"Clearing {product} SELL: {sent_quantity} @ {fair_for_ask}")

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
                logger.print(f"Clearing {product} BUY: {sent_quantity} @ {fair_for_bid}")
    
        return orders, buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int, 
                   buy_order_volume: int, sell_order_volume: int) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
            logger.print(f"MM {product} BUY: {buy_quantity} @ {bid}")

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
            logger.print(f"MM {product} SELL: {sell_quantity} @ {ask}")
        return buy_order_volume, sell_order_volume
    
    def make_KELP_orders(self, order_depth: OrderDepth, fair_value: float, position: int, 
                        buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        orders: List[Order] = []
        
        # Handle empty order book
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders, buy_order_volume, sell_order_volume
            
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbbf = max(bbf) if bbf else fair_value - 2
        
        bid_price = int(bbbf + 1)
        ask_price = int(baaf - 1)
        
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP, orders, bid_price, ask_price, position, 
            buy_order_volume, sell_order_volume
        )
        
        return orders, buy_order_volume, sell_order_volume
    
    def make_SQUIDINK_orders(self, order_depth: OrderDepth, mid_price: float, position: int) -> List[Order]:
        """Mean reversion strategy for SQUID_INK"""
        orders = []
        
        # Handle empty order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            logger.print("Empty order book - skipping")
            return orders
        
        # Get SQUID_INK parameters
        base_value = self.params[Product.SQUID_INK]["base_value"]
        deviation_threshold = self.params[Product.SQUID_INK]["deviation_threshold"] 
        max_order_size = self.params[Product.SQUID_INK]["max_order_size"]
        no_trade_zone = self.params[Product.SQUID_INK]["no_trade_zone"]
        
        # Calculate mid price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
            
        # Update price extremes
        self.squid_min_price_seen = min(self.squid_min_price_seen, mid_price)
        self.squid_max_price_seen = max(self.squid_max_price_seen, mid_price)
        
        # Calculate deviation from base value
        deviation = mid_price - base_value
        deviation_pct = deviation / base_value
        
        # Log current state
        logger.print(f"SQUID_INK Mid price: {mid_price}, Deviation: {deviation} ({deviation_pct:.2%})")
        logger.print(f"Price range seen: {self.squid_min_price_seen} - {self.squid_max_price_seen}")
        
        # Don't trade if price is within the no-trade zone
        if abs(deviation) < no_trade_zone:
            logger.print(f"Price within no-trade zone ({no_trade_zone}) - skipping")
            return orders
        
        # Mean reversion strategy
        if deviation >= deviation_threshold:
            # Price is above base - SELL
            logger.print(f"Price above base by {deviation} - looking to SELL")
            
            # Calculate scaling factor
            scale = self.get_squid_scaling_factor(position, False)
            
            # Scale order size based on deviation and position
            deviation_factor = min(1.0, deviation / 100)  # Cap at 1.0
            sell_size = int(max_order_size * scale * (1 + deviation_factor))
            
            # Ensure we don't exceed position limits
            sell_size = min(sell_size, self.LIMIT[Product.SQUID_INK] + position)
            
            # Also ensure we don't exceed available liquidity
            sell_size = min(sell_size, order_depth.buy_orders[best_bid])
            
            if sell_size > 0:
                orders.append(Order(Product.SQUID_INK, best_bid, -sell_size))
                logger.print(f"SELLING {sell_size} @ {best_bid} (scale: {scale:.2f})")
        
        elif deviation <= -deviation_threshold:
            # Price is below base - BUY
            logger.print(f"Price below base by {-deviation} - looking to BUY")
            
            # Calculate scaling factor
            scale = self.get_squid_scaling_factor(position, True)
            
            # Scale order size based on deviation and position
            deviation_factor = min(1.0, -deviation / 100)  # Cap at 1.0
            buy_size = int(max_order_size * scale * (1 + deviation_factor))
            
            # Ensure we don't exceed position limits
            buy_size = min(buy_size, self.LIMIT[Product.SQUID_INK] - position)
            
            # Also ensure we don't exceed available liquidity
            buy_size = min(buy_size, -order_depth.sell_orders[best_ask])
            
            if buy_size > 0:
                orders.append(Order(Product.SQUID_INK, best_ask, buy_size))
                logger.print(f"BUYING {buy_size} @ {best_ask} (scale: {scale:.2f})")
        
        return orders
    
    def make_RESIN_orders(self, order_depth: OrderDepth, fair_value: int, position: int, 
                         buy_order_volume: int, sell_order_volume: int, volume_limit: int) -> (List[Order], int, int):
        orders: List[Order] = []
        
        # Handle empty order book
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders, buy_order_volume, sell_order_volume
            
        asks_above = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(asks_above) if asks_above else fair_value + 3
        
        bids_below = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(bids_below) if bids_below else fair_value - 3
        
        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3
        
        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3

        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN, orders, bbbf + 1, baaf - 1, position, 
            buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume
        
    # === NEW FUNCTIONS FOR PICNIC BASKET TRADING ===
    
    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Calculate the smart weighted mid price"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
    
    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        """Create a synthetic order depth for the basket based on component prices"""
        # Initialize the synthetic basket order depth
        synthetic_order_depth = OrderDepth()
        
        # Check if all components are available
        required_products = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        for product in required_products:
            if product not in order_depths or not order_depths[product].buy_orders or not order_depths[product].sell_orders:
                logger.print(f"Missing order data for {product} - cannot create synthetic basket")
                return synthetic_order_depth
                
        # Calculate the best bid and ask for each component
        croissant_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys())
        croissant_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys())
        jam_best_bid = max(order_depths[Product.JAMS].buy_orders.keys())
        jam_best_ask = min(order_depths[Product.JAMS].sell_orders.keys())
        djembe_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys())
        djembe_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys())
        
        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissant_best_bid * BASKET_WEIGHTS[Product.CROISSANTS] +
            jam_best_bid * BASKET_WEIGHTS[Product.JAMS] +
            djembe_best_bid * BASKET_WEIGHTS[Product.DJEMBES]
        )
        
        implied_ask = (
            croissant_best_ask * BASKET_WEIGHTS[Product.CROISSANTS] +
            jam_best_ask * BASKET_WEIGHTS[Product.JAMS] +
            djembe_best_ask * BASKET_WEIGHTS[Product.DJEMBES]
        )
        
        # Calculate the maximum number of synthetic baskets available
        # For bid side (how many baskets we can sell)
        croissant_bid_volume = order_depths[Product.CROISSANTS].buy_orders[croissant_best_bid] // BASKET_WEIGHTS[Product.CROISSANTS]
        jam_bid_volume = order_depths[Product.JAMS].buy_orders[jam_best_bid] // BASKET_WEIGHTS[Product.JAMS]
        djembe_bid_volume = order_depths[Product.DJEMBES].buy_orders[djembe_best_bid] // BASKET_WEIGHTS[Product.DJEMBES]
        implied_bid_volume = min(croissant_bid_volume, jam_bid_volume, djembe_bid_volume)
        
        # For ask side (how many baskets we can buy)
        croissant_ask_volume = -order_depths[Product.CROISSANTS].sell_orders[croissant_best_ask] // BASKET_WEIGHTS[Product.CROISSANTS]
        jam_ask_volume = -order_depths[Product.JAMS].sell_orders[jam_best_ask] // BASKET_WEIGHTS[Product.JAMS]
        djembe_ask_volume = -order_depths[Product.DJEMBES].sell_orders[djembe_best_ask] // BASKET_WEIGHTS[Product.DJEMBES]
        implied_ask_volume = min(croissant_ask_volume, jam_ask_volume, djembe_ask_volume)
        
        # Add to synthetic order book
        if implied_bid_volume > 0:
            synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume
            
        if implied_ask_volume > 0:
            synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume
            
        return synthetic_order_depth
    
    def execute_spread_orders(self, target_position: int, basket_position: int, 
                            order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        """Execute orders to trade the spread between basket and components"""
        if target_position == basket_position:
            return None
            
        # How many baskets to buy/sell
        target_quantity = abs(target_position - basket_position)
        
        # Get order depths
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        
        # Check if we have valid depths
        if (not basket_order_depth.buy_orders or not basket_order_depth.sell_orders or 
            not synthetic_order_depth.buy_orders or not synthetic_order_depth.sell_orders):
            logger.print("Missing order depths for basket trade")
            return None
            
        # Create result dict for orders
        result_orders = {
            Product.PICNIC_BASKET1: [],
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: []
        }
        
        if target_position > basket_position:
            # We need to BUY baskets and SELL components
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            
            # How many we can execute based on order book and target
            execute_volume = min(basket_ask_volume, synthetic_bid_volume, target_quantity)
            
            if execute_volume <= 0:
                logger.print("No executable volume for basket trade")
                return None
                
            # Create orders
            # Buy basket
            result_orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume))

            croissant_bid_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
            jam_bid_price = max(order_depths[Product.JAMS].buy_orders.keys())
            djembe_bid_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            
            # Create component sell orders
            result_orders[Product.CROISSANTS].append(
                Order(Product.CROISSANTS, croissant_bid_price, -execute_volume * BASKET_WEIGHTS[Product.CROISSANTS])
            )
            result_orders[Product.JAMS].append(
                Order(Product.JAMS, jam_bid_price, -execute_volume * BASKET_WEIGHTS[Product.JAMS])
            )
            result_orders[Product.DJEMBES].append(
                Order(Product.DJEMBES, djembe_bid_price, -execute_volume * BASKET_WEIGHTS[Product.DJEMBES])
            )
            
            logger.print(f"Buying {execute_volume} baskets @ {basket_ask_price} and selling components @ {synthetic_bid_price}")
            
        else:
            # We need to SELL baskets and BUY components
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            
            # How many we can execute based on order book and target
            execute_volume = min(basket_bid_volume, synthetic_ask_volume, target_quantity)
            
            if execute_volume <= 0:
                logger.print("No executable volume for basket trade")
                return None
                
            # Create orders
            # Sell basket
            result_orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume))
            
            # Buy components
            # Get component prices
            croissant_ask_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
            jam_ask_price = min(order_depths[Product.JAMS].sell_orders.keys())
            djembe_ask_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            
            # Create component buy orders
            result_orders[Product.CROISSANTS].append(
                Order(Product.CROISSANTS, croissant_ask_price, execute_volume * BASKET_WEIGHTS[Product.CROISSANTS])
            )
            result_orders[Product.JAMS].append(
                Order(Product.JAMS, jam_ask_price, execute_volume * BASKET_WEIGHTS[Product.JAMS])
            )
            result_orders[Product.DJEMBES].append(
                Order(Product.DJEMBES, djembe_ask_price, execute_volume * BASKET_WEIGHTS[Product.DJEMBES])
            )
            
            logger.print(f"Selling {execute_volume} baskets @ {basket_bid_price} and buying components @ {synthetic_ask_price}")
            
        return result_orders
    

    def picnic_basket_strategy(self, order_depths: Dict[str, OrderDepth], basket_position: int) -> Dict[str, List[Order]]:
        """Implement the spread arbitrage strategy for picnic baskets"""
        # Check if we have all required order depths
        required_products = [Product.PICNIC_BASKET1, Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        for product in required_products:
            if product not in order_depths or not order_depths[product].buy_orders or not order_depths[product].sell_orders:
                logger.print(f"Missing order data for {product} - skipping basket trade")
                return {}
                    
        # Get order depths and calculate prices
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        
        # Calculate smart weighted midpoint prices
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        
        if basket_swmid is None or synthetic_swmid is None:
            logger.print("Could not calculate midpoint prices - skipping basket trade")
            return {}
                
        # Calculate the spread
        spread = basket_swmid - synthetic_swmid
        logger.print(f"Basket spread: {spread} (Basket: {basket_swmid}, Synthetic: {synthetic_swmid})")
        
        # Update spread history
        self.spread_history.append(spread)
        
        # Keep spread history to a limited window
        window_size = self.params["SPREAD"]["spread_std_window"]
        if len(self.spread_history) > window_size:
            self.spread_history.pop(0)
                
        # Only start trading after collecting enough history
        if len(self.spread_history) < window_size:
            logger.print(f"Building spread history: {len(self.spread_history)}/{window_size}")
            return {}
                
        # Calculate mean and standard deviation
        spread_mean = np.mean(self.spread_history)
        spread_std = np.std(self.spread_history)
        
        # Update the default spread mean dynamically
        self.params["SPREAD"]["default_spread_mean"] = spread_mean
        
        # Calculate z-score
        zscore = (spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Trading logic parameters
        zscore_threshold = self.params["SPREAD"]["zscore_threshold"]
        target_position = self.params["SPREAD"]["target_position"]
        TRANSACTION_COST_PER_UNIT = 10  # Match your backtester
        
        # Log detailed spread information
        logger.print(f"Spread Analysis:")
        logger.print(f"  Current Spread: {spread}")
        logger.print(f"  Spread Mean: {spread_mean}")
        logger.print(f"  Spread Std Dev: {spread_std}")
        logger.print(f"  Z-Score: {zscore}")
        logger.print(f"  Threshold: {zscore_threshold}")
        logger.print(f"  Current Position: {basket_position}")
        logger.print(f"  Target Position: {target_position}")
        
        # Initialize result dictionary
        result = {
            Product.PICNIC_BASKET1: []
        }
        
        # Get best bid/ask for basket
        if not basket_order_depth.buy_orders or not basket_order_depth.sell_orders:
            return result
            
        best_basket_bid = max(basket_order_depth.buy_orders.keys())
        best_basket_ask = min(basket_order_depth.sell_orders.keys())
        
        # Calculate position limits
        max_basket_position = self.LIMIT[Product.PICNIC_BASKET1]
        
        if zscore >= zscore_threshold:
            # Spread is too high - sell basket
            if basket_position > -target_position:
                # Calculate how much we can sell
                sell_size = min(
                    abs(target_position + basket_position),  # How far from target short position
                    abs(basket_order_depth.buy_orders[best_basket_bid]),  # Available liquidity
                    max_basket_position + basket_position  # Position limit check
                )
                
                # Check if trade is profitable after costs
                if sell_size > 0:
                    trade_cost = sell_size * TRANSACTION_COST_PER_UNIT
                    trade_revenue = abs(spread) * sell_size
                    
                    if trade_revenue > trade_cost:
                        result[Product.PICNIC_BASKET1].append(
                            Order(Product.PICNIC_BASKET1, best_basket_bid, -sell_size)
                        )
                        logger.print(f"SELL BASKET: {sell_size} @ {best_basket_bid}")
                        logger.print(f"Expected profit: {trade_revenue - trade_cost}")
                    
        elif zscore <= -zscore_threshold:
            # Spread is too low - buy basket
            if basket_position < target_position:
                # Calculate how much we can buy
                buy_size = min(
                    abs(target_position - basket_position),  # How far from target long position
                    abs(basket_order_depth.sell_orders[best_basket_ask]),  # Available liquidity
                    max_basket_position - basket_position  # Position limit check
                )
                
                # Check if trade is profitable after costs
                if buy_size > 0:
                    trade_cost = buy_size * TRANSACTION_COST_PER_UNIT
                    trade_revenue = abs(spread) * buy_size
                    
                    if trade_revenue > trade_cost:
                        result[Product.PICNIC_BASKET1].append(
                            Order(Product.PICNIC_BASKET1, best_basket_ask, buy_size)
                        )
                        logger.print(f"BUY BASKET: {buy_size} @ {best_basket_ask}")
                        logger.print(f"Expected profit: {trade_revenue - trade_cost}")
        
        return result

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        
        # Handle SQUID_INK trading using mean reversion strategy
        if Product.SQUID_INK in state.order_depths:
            position = state.position.get(Product.SQUID_INK, 0)
            logger.print(f"SQUID_INK position: {position}")
            
            squid_orders = self.make_SQUIDINK_orders(
                state.order_depths[Product.SQUID_INK],
                self.squidink_fv(state.order_depths[Product.SQUID_INK], 0),
                position
            )
            
            result[Product.SQUID_INK] = squid_orders
        
        # Handle KELP trading
        if Product.KELP in state.order_depths and Product.KELP in self.params:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fairvalue = self.kelp_fv(state.order_depths[Product.KELP], self.params[Product.KELP]["timespan"])
            
            if kelp_fairvalue is not None:  # Only trade if we have a valid fair value
                buy_order_volume = 0
                sell_order_volume = 0
                
                kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fairvalue,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"]
                )
                
                kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_position_order(
                    Product.KELP,
                    kelp_fairvalue,
                    self.params[Product.KELP]["clear_width"],
                    [],
                    state.order_depths[Product.KELP],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume
                )
                
                kelp_make_orders, _, _ = self.make_KELP_orders(
                    state.order_depths[Product.KELP],
                    kelp_fairvalue,
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume
                )
                
                result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders
        
        # Handle RAINFOREST_RESIN trading
        if Product.RAINFOREST_RESIN in state.order_depths and Product.RAINFOREST_RESIN in self.params:
            position = state.position.get(Product.RAINFOREST_RESIN, 0)
            buy_vol = 0
            sell_vol = 0
            
            take_orders, buy_vol, sell_vol = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                position
            )
            
            clear_orders, buy_vol, sell_vol = self.clear_position_order(
                Product.RAINFOREST_RESIN,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                [],
                state.order_depths[Product.RAINFOREST_RESIN],
                position,
                buy_vol,
                sell_vol
            )
            
            make_orders, _, _ = self.make_RESIN_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                position,
                buy_vol,
                sell_vol,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"]
            )
            
            result[Product.RAINFOREST_RESIN] = take_orders + clear_orders + make_orders
            
        # Handle PICNIC_BASKET1 trading
        if Product.PICNIC_BASKET1 in state.order_depths:
            basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
            logger.print(f"PICNIC_BASKET1 position: {basket_position}")
            
            # Run the basket spread strategy
            basket_orders = self.picnic_basket_strategy(state.order_depths, basket_position)
            
            # Add basket orders to result if they exist
            if basket_orders and Product.PICNIC_BASKET1 in basket_orders:
                result[Product.PICNIC_BASKET1] = basket_orders[Product.PICNIC_BASKET1]
    
        # Debug data
        trader_data = json.dumps({
            "squid_price_range": {
                "min": self.squid_min_price_seen if self.squid_min_price_seen != float('inf') else "N/A",
                "max": self.squid_max_price_seen if self.squid_max_price_seen != float('-inf') else "N/A",
                "base_value": self.params[Product.SQUID_INK]["base_value"]
            },
            "positions": {
                "RAINFOREST_RESIN": state.position.get(Product.RAINFOREST_RESIN, 0),
                "KELP": state.position.get(Product.KELP, 0),
                "SQUID_INK": state.position.get(Product.SQUID_INK, 0),
                "PICNIC_BASKET1": state.position.get(Product.PICNIC_BASKET1, 0)
            },
            "basket_spread": {
                "history_length": len(self.spread_history),
                "latest": self.spread_history[-1] if self.spread_history else "N/A",
                "mean": np.mean(self.spread_history) if self.spread_history else "N/A",
                "std": np.std(self.spread_history) if self.spread_history else "N/A"
            }
        })
        
        conversions = 1
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data