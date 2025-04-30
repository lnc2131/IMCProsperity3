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
    SQUID_INK = "SQUID_INK"  # Added back SQUID_INK
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"

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
    
    Product.SQUID_INK: {  # Added back SQUID_INK parameters
        "base_value": 2000,
        "deviation_threshold": 0,
        "max_order_size": 50,
        "no_trade_zone": 0
    },
    
    "SPREAD": {
        "spread_std_window": 30,
        "zscore_threshold": 3.5,
        "target_position": 60
    }
}

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
            Product.SQUID_INK: 50,  # Added SQUID_INK limit
            Product.PICNIC_BASKET1: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60
        }
        
        self.spread_history = []
        
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
        
    def kelp_fv(self, order_depth: OrderDepth, timespan: int) -> float:
        """Calculate fair value for KELP"""
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
        return int(mmprice)

    def take_best_orders_with_adverse(self, product: str, fair_value: int, take_width: float, orders: List[Order], 
                                    order_depth: OrderDepth, position: int, buy_order_volume: int, 
                                    sell_order_volume: int, adverse_volume: int) -> (int, int):
        """Take orders considering adverse selection"""
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
        """Execute orders at favorable prices"""
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
        """Clear positions near fair value"""
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
        """Place market making orders"""
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
        """Generate orders for KELP"""
        orders: List[Order] = []
        
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders, buy_order_volume, sell_order_volume
            
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbbf = max(bbf) if bbf else fair_value - 2
        
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP, orders, bbbf + 1, baaf - 1, position, 
            buy_order_volume, sell_order_volume
        )
        
        return orders, buy_order_volume, sell_order_volume

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
        """Generate orders for RAINFOREST_RESIN"""
        orders: List[Order] = []
        
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

    def calculate_mid_price_value(self, order_depths: Dict[str, OrderDepth]) -> float:
        """Calculate basket value using simple mid prices"""
        try:
            # Calculate mid price for each component
            croissant_mid = (max(order_depths[Product.CROISSANTS].buy_orders.keys()) + 
                           min(order_depths[Product.CROISSANTS].sell_orders.keys())) / 2
            
            jam_mid = (max(order_depths[Product.JAMS].buy_orders.keys()) + 
                      min(order_depths[Product.JAMS].sell_orders.keys())) / 2
            
            djembe_mid = (max(order_depths[Product.DJEMBES].buy_orders.keys()) + 
                         min(order_depths[Product.DJEMBES].sell_orders.keys())) / 2
            
            # Calculate basket value using component weights
            basket_value = (croissant_mid * BASKET_WEIGHTS[Product.CROISSANTS] +
                          jam_mid * BASKET_WEIGHTS[Product.JAMS] +
                          djembe_mid * BASKET_WEIGHTS[Product.DJEMBES])
            
            return basket_value
        except:
            return None
    
    def get_basket_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid price for basket"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def picnic_basket_strategy(self, order_depths: Dict[str, OrderDepth], basket_position: int) -> Dict[str, List[Order]]:
        """Implement the spread arbitrage strategy for picnic baskets using mid prices"""
        # Check if we have all required order depths 
        required_products = [Product.PICNIC_BASKET1, Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        for product in required_products:
            if product not in order_depths or not order_depths[product].buy_orders or not order_depths[product].sell_orders:
                logger.print(f"Missing order data for {product} - skipping basket trade")
                return {}
                    
        # Calculate actual basket mid price
        basket_mid = self.get_basket_mid_price(order_depths[Product.PICNIC_BASKET1])
        
        # Calculate synthetic basket value using mid prices
        synthetic_value = self.calculate_mid_price_value(order_depths)
        
        if basket_mid is None or synthetic_value is None:
            logger.print("Could not calculate prices - skipping basket trade")
            return {}
                
        # Calculate the spread
        spread = basket_mid - synthetic_value
        logger.print(f"Basket spread: {spread} (Basket: {basket_mid}, Synthetic: {synthetic_value})")
        
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
        
        # Calculate z-score
        zscore = (spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Trading logic parameters
        zscore_threshold = self.params["SPREAD"]["zscore_threshold"]
        target_position = self.params["SPREAD"]["target_position"]
        TRANSACTION_COST_PER_UNIT = 10
        
        # Log detailed spread information
        logger.print(f"Spread Analysis:")
        logger.print(f"  Current Spread: {spread}")
        logger.print(f"  Spread Mean: {spread_mean}")
        logger.print(f"  Spread Std Dev: {spread_std}")
        logger.print(f"  Z-Score: {zscore}")
        logger.print(f"  Current Position: {basket_position}")
        
        # Initialize result dictionary
        result = {
            Product.PICNIC_BASKET1: []
        }
        
        # Get best bid/ask for basket
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        if not basket_order_depth.buy_orders or not basket_order_depth.sell_orders:
            return result
            
        best_basket_bid = max(basket_order_depth.buy_orders.keys())
        best_basket_ask = min(basket_order_depth.sell_orders.keys())
        
        if zscore >= zscore_threshold:
            # Spread is too high - sell basket
            if basket_position > -target_position:
                # Calculate how much we can sell
                sell_size = min(
                    abs(target_position + basket_position),  # How far from target short position
                    abs(basket_order_depth.buy_orders[best_basket_bid]),  # Available liquidity
                    self.LIMIT[Product.PICNIC_BASKET1] + basket_position  # Position limit check
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
                    self.LIMIT[Product.PICNIC_BASKET1] - basket_position  # Position limit check
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
            
            # Use mean reversion strategy for SQUID_INK
            squid_orders = self.make_SQUIDINK_orders(
                state.order_depths[Product.SQUID_INK],
                (max(state.order_depths[Product.SQUID_INK].buy_orders.keys()) + 
                 min(state.order_depths[Product.SQUID_INK].sell_orders.keys())) / 2,
                position
            )
            
            result[Product.SQUID_INK] = squid_orders
        
        # Handle KELP trading
        if Product.KELP in state.order_depths and Product.KELP in self.params:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_take_orders = []
            kelp_clear_orders = []
            kelp_make_orders = []
            
            logger.print(f"KELP position: {kelp_position}")
            
            if kelp_position < self.LIMIT[Product.KELP]:
                kelp_fairvalue = self.kelp_fv(state.order_depths[Product.KELP], self.params[Product.KELP]["timespan"])
                
                if kelp_fairvalue is not None:  # Only trade if we have a valid fair value
                    buy_order_volume = 0  # Initialize volumes
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
            buy_vol = 0  # Initialize volumes
            sell_vol = 0
            
            logger.print(f"RESIN position: {position}")
            
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
            
            if basket_orders and Product.PICNIC_BASKET1 in basket_orders:
                result[Product.PICNIC_BASKET1] = basket_orders[Product.PICNIC_BASKET1]

        # State data for debugging
        trader_data = json.dumps({
            "positions": {
                "RAINFOREST_RESIN": state.position.get(Product.RAINFOREST_RESIN, 0),
                "KELP": state.position.get(Product.KELP, 0),
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