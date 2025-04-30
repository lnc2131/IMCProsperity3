from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation
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
                getattr(observation, 'sugarPrice', None),  # Use getattr to handle potential missing attribute
                getattr(observation, 'sunlightIndex', None),  # Use getattr to handle potential missing attribute
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"))

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
    PICNIC_BASKET2 = "PICNIC_BASKET2"
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
        "deviation_threshold": 0,
        "max_order_size": 50,
        "no_trade_zone": 0
    },
    
    # Adding Picnic Basket strategy parameters
    "SPREAD": {
        "default_spread_mean": 70.05,  # This will be dynamically calculated
        "spread_std_window": 30,
        "zscore_threshold": 2.0,  # Slightly reduced threshold for more trades
        "target_position": 60,
        "min_profit_per_unit": 1.0  # Minimum profit required per unit to trade
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
            Product.PICNIC_BASKET2: 100,
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
        
        # Last execution price tracking for basket and components
        self.last_execution_prices = {
            Product.PICNIC_BASKET1: None,
            Product.CROISSANTS: None,
            Product.JAMS: None,
            Product.DJEMBES: None
        }
        
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
        
    # === FUNCTIONS FOR PICNIC BASKET TRADING ===
    
    def get_vwap(self, order_depth: OrderDepth) -> float:
        """Calculate the volume-weighted average price (VWAP) from the order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        # Get best bid and ask
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        
        total_volume = best_bid_vol + best_ask_vol
        if total_volume == 0:
            return (best_bid + best_ask) / 2
            
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_volume
    
    def get_synthetic_basket_price(self, order_depths: Dict[str, OrderDepth]) -> Dict[str, float]:
        """Calculate synthetic basket price from component prices"""
        # Check if all components are available
        required_products = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        for product in required_products:
            if product not in order_depths or not order_depths[product].buy_orders or not order_depths[product].sell_orders:
                logger.print(f"Missing order data for {product} - cannot calculate synthetic basket")
                return None
                
        # Calculate the VWAP for each component
        component_prices = {}
        for product in required_products:
            component_prices[product] = self.get_vwap(order_depths[product])
            if component_prices[product] is None:
                return None
        
        # Calculate synthetic basket price
        synthetic_price = (
            component_prices[Product.CROISSANTS] * BASKET_WEIGHTS[Product.CROISSANTS] +
            component_prices[Product.JAMS] * BASKET_WEIGHTS[Product.JAMS] +
            component_prices[Product.DJEMBES] * BASKET_WEIGHTS[Product.DJEMBES]
        )
        
        return {
            'basket': synthetic_price,
            'components': component_prices
        }
    
    def check_profit_potential(self, basket_price: float, synthetic_price: float) -> float:
        """Calculate profit potential per trade"""
        # Transaction costs could be modeled here
        TRANSACTION_COST_PER_UNIT = 5  # Estimated cost including slippage, etc.
        
        # For buy basket, sell components
        if basket_price < synthetic_price:
            profit_per_unit = synthetic_price - basket_price - TRANSACTION_COST_PER_UNIT
            return profit_per_unit
            
        # For sell basket, buy components
        elif basket_price > synthetic_price:
            profit_per_unit = basket_price - synthetic_price - TRANSACTION_COST_PER_UNIT
            return profit_per_unit
            
        return 0.0
    
    def calculate_trade_quantities(self, 
                                  basket_position: int, 
                                  target_position: int,
                                  basket_order_depth: OrderDepth, 
                                  component_depths: Dict[str, OrderDepth]) -> Dict[str, int]:
        """Calculate optimal quantities for basket and component trades"""
        # How much we want to trade
        quantity_to_trade = abs(target_position - basket_position)
        if quantity_to_trade <= 0:
            return None
            
        # Check liquidity of basket
        if target_position > basket_position:  # We want to buy baskets
            if not basket_order_depth.sell_orders:
                return None
            basket_available = min(sum(-qty for qty in basket_order_depth.sell_orders.values()), 
                                 self.LIMIT[Product.PICNIC_BASKET1] - basket_position)
        else:  # We want to sell baskets
            if not basket_order_depth.buy_orders:
                return None
            basket_available = min(sum(qty for qty in basket_order_depth.buy_orders.values()),
                                 self.LIMIT[Product.PICNIC_BASKET1] + basket_position)
        
        # Calculate how many components we need
        croissant_needed = BASKET_WEIGHTS[Product.CROISSANTS] * quantity_to_trade
        jam_needed = BASKET_WEIGHTS[Product.JAMS] * quantity_to_trade
        djembe_needed = BASKET_WEIGHTS[Product.DJEMBES] * quantity_to_trade
        
        # Check component position limits
        croissant_position = 0  # Need to get these from state
        jam_position = 0
        djembe_position = 0
        
        if target_position > basket_position:  # We want to buy baskets and sell components
            croissant_available = min(
                sum(qty for qty in component_depths[Product.CROISSANTS].buy_orders.values()),
                self.LIMIT[Product.CROISSANTS] + croissant_position
            )
            jam_available = min(
                sum(qty for qty in component_depths[Product.JAMS].buy_orders.values()),
                self.LIMIT[Product.JAMS] + jam_position
            )
            djembe_available = min(
                sum(qty for qty in component_depths[Product.DJEMBES].buy_orders.values()),
                self.LIMIT[Product.DJEMBES] + djembe_position
            )
        else:  # We want to sell baskets and buy components
            croissant_available = min(
                sum(-qty for qty in component_depths[Product.CROISSANTS].sell_orders.values()),
                self.LIMIT[Product.CROISSANTS] - croissant_position
            )
            jam_available = min(
                sum(-qty for qty in component_depths[Product.JAMS].sell_orders.values()),
                self.LIMIT[Product.JAMS] - jam_position
            )
            djembe_available = min(
                sum(-qty for qty in component_depths[Product.DJEMBES].sell_orders.values()),
                self.LIMIT[Product.DJEMBES] - djembe_position
            )
        
        # Calculate max executable quantity based on all constraints
        max_basket_component_equivalents = [
            basket_available,
            croissant_available // BASKET_WEIGHTS[Product.CROISSANTS],
            jam_available // BASKET_WEIGHTS[Product.JAMS],
            djembe_available // BASKET_WEIGHTS[Product.DJEMBES]
        ]
        
        executable_quantity = min(quantity_to_trade, min(max_basket_component_equivalents))
        
        if executable_quantity <= 0:
            return None
            
        return {
            'quantity': executable_quantity,
            'croissants': executable_quantity * BASKET_WEIGHTS[Product.CROISSANTS],
            'jams': executable_quantity * BASKET_WEIGHTS[Product.JAMS],
            'djembes': executable_quantity * BASKET_WEIGHTS[Product.DJEMBES]
        }