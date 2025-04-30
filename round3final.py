import json # USE STANDARD JSON LIBRARY
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist # Ensure available or replace
from typing import List, Dict, Any, Optional, Tuple

# Assuming datamodel contains necessary classes like OrderDepth, TradingState, Order etc.
# If not, mock classes would be needed.
try:
    from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Listing, ProsperityEncoder
except ImportError:
    # Add mock classes here if running standalone
    class Listing:
        def __init__(self, product, denomination): self.product = product; self.denomination = denomination
    class OrderDepth: buy_orders: Dict[int, int] = {}; sell_orders: Dict[int, int] = {}
    class TradingState:
        timestamp: int; listings: Dict[str, Listing]; order_depths: Dict[str, OrderDepth]
        own_trades: Dict[str, Any]; market_trades: Dict[str, Any]
        position: Dict[str, int]; observations: Any; traderData: str
    class Order:
        def __init__(self, symbol: str, price: int, quantity: int): self.symbol=symbol; self.price=price; self.quantity=quantity
    class Symbol(str): pass
    # Use standard JSONEncoder if ProsperityEncoder isn't specifically needed
    class ProsperityEncoder(json.JSONEncoder): pass

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    VOLCANIC_ROCK = "VOLCANIC_ROCK" # Underlying for vouchers

    # New Volcanic Rock Vouchers - ** REPLACE KEYS WITH ACTUAL SIMULATION SYMBOLS **
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500" # Assumed symbol
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750" # Assumed symbol
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000" # Assumed symbol
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250" # Assumed symbol
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500" # Assumed symbol

    # New Baskets - ** REPLACE KEYS WITH ACTUAL SIMULATION SYMBOLS **
    PICNIC_BASKET1 = "PICNIC_BASKET1" # Assumed symbol
    PICNIC_BASKET2 = "PICNIC_BASKET2" # Assumed symbol

    # Internal synthetic product name
    SYNTHETIC_PICNIC1 = "SYNTHETIC_PICNIC1" # Used for spread calc
    SYNTHETIC_PICNIC2 = "SYNTHETIC_PICNIC2"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


# Information about the Volcanic Rock Vouchers
VOUCHER_INFO = {
    # ** CONFIRM THESE SYMBOLS WITH YOUR SIMULATION **
    Product.VOLCANIC_ROCK_VOUCHER_9500: {"strike": 9500, "limit": 200},
    Product.VOLCANIC_ROCK_VOUCHER_9750: {"strike": 9750, "limit": 200},
    Product.VOLCANIC_ROCK_VOUCHER_10000: {"strike": 10000, "limit": 200},
    Product.VOLCANIC_ROCK_VOUCHER_10250: {"strike": 10250, "limit": 200},
    Product.VOLCANIC_ROCK_VOUCHER_10500: {"strike": 10500, "limit": 200},
}
VOUCHER_SYMBOLS = list(VOUCHER_INFO.keys())


SHARED_VOUCHER_PARAMS = {
     "r": 0.0,                       # Risk-free rate
     "global_fallback_sigma": 0.1,  # Fallback volatility if ATM IV calculation fails ** TUNE THIS **
     "total_duration_days": 7,
     "trading_days_year": 252,
     "ticks_per_day": 1_000_000
}

# Initial PARAMS definition with specific strategy blocks
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000, "take_width": 2, "clear_width": 1, "volume_limit": 0,
    },
    Product.KELP: {
        "take_width": 1, "clear_width": 0, "prevent_adverse": True, "adverse_volume": 15,
        "reversion_beta": -0.229, "kelp_min_edge": 2,
    },
    "PICNIC_BASKET1_SPREAD": {
        # ** TUNE THESE **
        "default_spread_mean": 60.0, "default_spread_std": 15.0, "spread_std_window": 50,
        "zscore_threshold": 2.0, "target_position": 50,
    },
    "PICNIC_BASKET2_SPREAD": {
         # ** TUNE THESE **
        "default_spread_mean": 40.0, "default_spread_std": 5.0, "spread_std_window": 50,
        "zscore_threshold": 2.0, "target_position": 50,
    },
    # Per-Voucher Parameters (Specific thresholds, sizes) - ** TUNE THESE **
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "trade_threshold_abs": 2, "exit_threshold_abs": 1.25, "base_order_size_voucher": 20,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "trade_threshold_abs": 2.25, "exit_threshold_abs": 1, "base_order_size_voucher": 20,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "trade_threshold_abs": 1.5, "exit_threshold_abs": 0.75, "base_order_size_voucher": 20,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "trade_threshold_abs": 1.5, "exit_threshold_abs": 0.5, "base_order_size_voucher": 20,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "trade_threshold_abs": 2, "exit_threshold_abs": 0.5, "base_order_size_voucher": 20,
    },
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 1,
        "make_probability": 0.800,
        "storage_cost": 0.1,
        "csi_threshold": 0.5,   # Critical Sunlight Index
        "sunlight_factor": 10,  # How strongly sunlight affects pricing
        "min_conversion_size": 8,  # Minimum position size to trigger conversion
        "take_width": 0,        # Minimum edge for arb opportunities
        "long_bias": -0.5,
        "position_limit": 75# Bias against long positions due to storage costs
    }
}

# *** FIX: Merge shared params into each voucher's specific params AFTER PARAMS is defined ***
for symbol in VOUCHER_SYMBOLS:
    if symbol not in PARAMS: # Ensure key exists
         PARAMS[symbol] = {}
    # Create a copy of shared params and update with specific params
    # This ensures each voucher has all necessary keys (r, time, thresholds, etc.)
    merged_params = SHARED_VOUCHER_PARAMS.copy()
    merged_params.update(PARAMS[symbol])
    PARAMS[symbol] = merged_params


# =============================================================================
# Position Limits (Using NEW Product Names)
# =============================================================================
LIMIT = {
    Product.RAINFOREST_RESIN: 20, Product.KELP: 20, Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 60, Product.JAMS: 250, Product.CROISSANTS: 350,
    Product.DJEMBES: 60, Product.VOLCANIC_ROCK: 400, Product.MAGNIFICENT_MACARONS: 75,
    # Limits per voucher from VOUCHER_INFO
    **{symbol: info["limit"] for symbol, info in VOUCHER_INFO.items()}
}

# =============================================================================
# Basket Definitions (Using NEW Product Names)
# =============================================================================
BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: { Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1 },
    Product.PICNIC_BASKET2: { Product.CROISSANTS: 4, Product.JAMS: 2 }
}

# =============================================================================
# BlackScholes Class
# =============================================================================
class BlackScholes:
    # ... (Keep static methods black_scholes_call, delta, vega, implied_volatility as defined previously) ...
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility, r=0.0):
        if volatility <= 1e-9 or time_to_expiry <= 1e-9 or spot <= 0 or strike <= 0: return max(0.0, spot - strike * exp(-r * time_to_expiry))
        try:
            d1 = (log(spot / strike) + (r + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
            d2 = d1 - volatility * sqrt(time_to_expiry)
            call_price = spot * NormalDist().cdf(d1) - strike * exp(-r * time_to_expiry) * NormalDist().cdf(d2)
            return max(0.0, call_price)
        except (ValueError, OverflowError, ZeroDivisionError): return max(0.0, spot - strike * exp(-r * time_to_expiry))

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility, r=0.0):
         if volatility <= 1e-9 or time_to_expiry <= 1e-9 or spot <= 0 or strike <= 0: return 1.0 if spot > strike else 0.0
         try:
            d1 = (log(spot / strike) + (r + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
            return NormalDist().cdf(d1)
         except (ValueError, OverflowError, ZeroDivisionError): return 1.0 if spot > strike else 0.0

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility, r=0.0):
        if volatility <= 1e-9 or time_to_expiry <= 1e-9 or spot <= 0 or strike <= 0: return 0.0
        try:
            d1 = (log(spot / strike) + (r + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
            return NormalDist().pdf(d1) * spot * sqrt(time_to_expiry)
        except (ValueError, OverflowError, ZeroDivisionError): return 0.0

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, r=0.0, max_iterations=100, tolerance=1e-6):
        if call_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 1e-9: return None
        intrinsic = max(0.0, spot - strike * exp(-r * time_to_expiry))
        if call_price < intrinsic - max(1.0, spot * 0.001) : return None
        low_vol, high_vol = 1e-4, 5.0
        for _ in range(max_iterations):
            mid_vol = (low_vol + high_vol) / 2.0;
            if mid_vol < 1e-9: mid_vol = 1e-9
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, mid_vol, r)
            if estimated_price is None:
                 if mid_vol == low_vol: low_vol += tolerance
                 elif mid_vol == high_vol: high_vol -= tolerance
                 else: return None
                 continue
            diff = estimated_price - call_price
            if abs(diff) < tolerance: return mid_vol
            if diff < 0: low_vol = mid_vol
            else: high_vol = mid_vol
            if high_vol - low_vol < tolerance: return mid_vol
        return None

# =============================================================================
# Logger Class (Using standard json, accepts dict trader_data)
# =============================================================================
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


# =============================================================================
# Helper Functions (Updated TTE signature)
# =============================================================================


def get_mid_price(symbol: str, state: TradingState) -> Optional[float]:
    order_depth = state.order_depths.get(symbol)
    if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders: return None
    best_bid = max(order_depth.buy_orders.keys()); best_ask = min(order_depth.sell_orders.keys())
    if best_bid == best_ask: return float(best_bid)
    return (best_bid + best_ask) / 2.0

def get_swmid(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth.buy_orders or not order_depth.sell_orders: return None
    best_bid = max(order_depth.buy_orders.keys()); best_ask = min(order_depth.sell_orders.keys())
    best_bid_vol = abs(order_depth.buy_orders[best_bid]); best_ask_vol = abs(order_depth.sell_orders[best_ask])
    total_vol = best_bid_vol + best_ask_vol
    if total_vol == 0: return (best_bid + best_ask) / 2.0
    return (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_vol

# *** MODIFIED TTE signature ***
def calculate_tte(timestamp: int, time_params: Dict) -> float:
    total_days = time_params["total_duration_days"]
    trading_days_year = time_params["trading_days_year"]
    ticks_per_day = time_params["ticks_per_day"]
    days_passed = timestamp // ticks_per_day
    remaining_days = max(0, total_days - days_passed)
    TTE = max(1e-9, remaining_days / trading_days_year)
    return TTE

def find_atm_voucher(St: float, state: TradingState) -> Optional[tuple[str, float]]:
    min_diff = float('inf'); atm_symbol = None; atm_strike = None
    if St is None: return None
    for symbol, info in VOUCHER_INFO.items():
        if symbol in state.listings:
            diff = abs(info["strike"] - St)
            if diff < min_diff: min_diff = diff; atm_symbol = symbol; atm_strike = info["strike"]
    if atm_symbol: return atm_symbol, float(atm_strike)
    else: return None

# =============================================================================
# Trader Class (Updated __init__, voucher param access, added basket2 logic)
# =============================================================================
class Trader:
    def __init__(self, params=None, limits=None):
        self.params = params if params is not None else PARAMS
        self.LIMIT = limits if limits is not None else LIMIT
        self.bs = BlackScholes()
        # Store shared voucher params for easier access
        self.shared_voucher_params = SHARED_VOUCHER_PARAMS


    # Inside your Trader class definition, add these methods:
    
    def macarons_implied_bid_ask(
        self,
        observation: ConversionObservation,
        sunlight_index: float,
        params: Dict
    ) -> (float, float):
        # Use parameters from PARAMS dictionary
        CSI = params["csi_threshold"]
        storage_cost = params["storage_cost"]
        sunlight_scaling = params["sunlight_factor"]
        
        # Base calculation similar to orchids
        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees
        
        # Adjust for storage costs on long positions
        implied_bid -= storage_cost  # Account for storage costs
        
        # Adjust for sunlight impact
        if sunlight_index < CSI:
            # Adjust fair values based on sunlight being below critical threshold
            # When sunlight is low, prices tend to rise (especially on the ask side)
            sunlight_factor = max(0.1, (CSI - sunlight_index) * sunlight_scaling)
            implied_ask += sunlight_factor  # Prices rise when sunlight is low
            implied_bid += sunlight_factor * 0.8  # Bid rises too but not as much as ask
            
        return implied_bid, implied_ask
    
    def macarons_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        params: Dict
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = params["position_limit"]  # Get position limit from params
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Extract sunlight index from observation
        sunlight_index = observation.sunlightIndex
        sugar_price = observation.sugarPrice
        
        # Use take_width from params instead of hardcoded value
        take_width = params["take_width"]
        
        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation, sunlight_index, params)
    
        # Available position capacity
        buy_quantity = position_limit - position
        sell_quantity = position_limit + position
    
        # Look for arbitrage opportunities
        for price in sorted(list(order_depth.sell_orders.keys())):
            # Only buy if market price is significantly below implied bid
            # Adjust with long_bias for storage costs
            if price < implied_bid - take_width - params["long_bias"]:
                quantity = min(
                    abs(order_depth.sell_orders[price]), buy_quantity
                )
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), quantity))
                    buy_order_volume += quantity
    
        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            # Sell if market price is above implied ask
            if price > implied_ask + take_width:
                quantity = min(
                    abs(order_depth.buy_orders[price]), sell_quantity
                )
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), -quantity))
                    sell_order_volume += quantity
    
        return orders, buy_order_volume, sell_order_volume
    
    def macarons_arb_clear(self, position: int, params: Dict) -> int:
        # More strategic with conversions due to limited conversions
        min_conversion_size = params["min_conversion_size"]
        
        # Only convert if position is significant
        if abs(position) >= min_conversion_size:
            return -position
        else:
            return 0  # Save conversions for when really needed
    
    def macarons_arb_make(
        self,
        observation: ConversionObservation,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        params: Dict
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = params["position_limit"]
        
        # Extract and use sunlight index
        sunlight_index = observation.sunlightIndex
        sugar_price = observation.sugarPrice
        
        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation, sunlight_index, params)
        
        # Get parameters from PARAMS dictionary
        edge = params["make_edge"]
        make_probability = params["make_probability"]
        CSI = params["csi_threshold"]
        
        # Adjust market making strategy based on sunlight index
        if sunlight_index < CSI:
            # When below CSI, widen spreads and be more aggressive when offering prices
            # Adjust the edge based on how far below CSI we are
            sunlight_adj = (CSI - sunlight_index) * params["sunlight_factor"] * 0.5
            bid_edge = edge + sunlight_adj
            ask_edge = edge + sunlight_adj * 1.5  # More conservative on ask side
        else:
            # Normal market making when sunlight is sufficient
            bid_edge = edge
            ask_edge = edge
        
        # Skew quotes based on position (prefer to reduce position)
        position_factor = position / position_limit
        if position > 0:  # Long - prefer to sell
            bid_edge += position_factor * 2
            ask_edge -= position_factor * 1
        elif position < 0:  # Short - prefer to buy
            bid_edge -= abs(position_factor) * 1
            ask_edge += abs(position_factor) * 2
            
        # Set our prices
        bid_price = implied_bid - bid_edge
        ask_price = implied_ask + ask_edge
        
        # Place orders with adjusted sizes
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid_price), buy_quantity))
    
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask_price), -sell_quantity))
    
        return orders, buy_order_volume, sell_order_volume
    # --- Generic Order Execution Helpers ---
    # (Insert implementations take_best_orders, take_best_orders_with_adverse, market_make, clear_position_order, take_orders, clear_orders)
    def take_best_orders( self, product: str, fair_value: float, take_width: float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int ) -> Tuple[int, int]:
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value - take_width:
                best_ask_amount = abs(order_depth.sell_orders[best_ask])
                quantity = min(best_ask_amount, position_limit - (position + buy_order_volume))
                if quantity > 0: orders.append(Order(product, best_ask, quantity)); buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value + take_width:
                best_bid_amount = abs(order_depth.buy_orders[best_bid])
                quantity = min(best_bid_amount, position_limit + (position - sell_order_volume))
                if quantity > 0: orders.append(Order(product, best_bid, -quantity)); sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse( self, product: str, fair_value: float, take_width: float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, adverse_volume: int ) -> Tuple[int, int]:
         position_limit = self.LIMIT[product]
         if order_depth.sell_orders:
             best_ask = min(order_depth.sell_orders.keys()); best_ask_amount = abs(order_depth.sell_orders[best_ask])
             if best_ask_amount <= adverse_volume:
                 if best_ask <= fair_value - take_width:
                     quantity = min(best_ask_amount, position_limit - (position + buy_order_volume))
                     if quantity > 0: orders.append(Order(product, best_ask, quantity)); buy_order_volume += quantity
         if order_depth.buy_orders:
             best_bid = max(order_depth.buy_orders.keys()); best_bid_amount = abs(order_depth.buy_orders[best_bid])
             if best_bid_amount <= adverse_volume:
                 if best_bid >= fair_value + take_width:
                     quantity = min(best_bid_amount, position_limit + (position - sell_order_volume))
                     if quantity > 0: orders.append(Order(product, best_bid, -quantity)); sell_order_volume += quantity
         return buy_order_volume, sell_order_volume

    def market_make( self, product: str, orders: List[Order], bid: float, ask: float, position: int, buy_order_volume: int, sell_order_volume: int, ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0: orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0: orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order( self, product: str, fair_value: float, width: float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, ) -> Tuple[int, int]:
         position_after_take = position + buy_order_volume - sell_order_volume
         fair_for_bid = fair_value - width; fair_for_ask = fair_value + width
         buy_avail_limit = self.LIMIT[product] - (position + buy_order_volume); sell_avail_limit = self.LIMIT[product] + (position - sell_order_volume)
         if position_after_take > 0: # Sell to clear long
             clear_qty_avail = sum(abs(vol) for price, vol in order_depth.buy_orders.items() if price >= fair_for_ask)
             qty_to_clear = min(clear_qty_avail, abs(position_after_take))
             sent_quantity = min(sell_avail_limit, qty_to_clear)
             if sent_quantity > 0: orders.append(Order(product, math.ceil(fair_for_ask), -sent_quantity)); sell_order_volume += sent_quantity
         elif position_after_take < 0: # Buy to clear short
             clear_qty_avail = sum(abs(vol) for price, vol in order_depth.sell_orders.items() if price <= fair_for_bid)
             qty_to_clear = min(clear_qty_avail, abs(position_after_take))
             sent_quantity = min(buy_avail_limit, qty_to_clear)
             if sent_quantity > 0: orders.append(Order(product, math.floor(fair_for_bid), sent_quantity)); buy_order_volume += sent_quantity
         return buy_order_volume, sell_order_volume

    def take_orders( self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int, prevent_adverse: bool = False, adverse_volume: int = 0, ) -> Tuple[List[Order], int, int]:
         orders: List[Order] = []; buy_order_volume = 0; sell_order_volume = 0
         if prevent_adverse:
             buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse( product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, adverse_volume)
         else:
             buy_order_volume, sell_order_volume = self.take_best_orders( product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume )
         return orders, buy_order_volume, sell_order_volume

    def clear_orders( self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: float, position: int, buy_order_volume: int, sell_order_volume: int, ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order( product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume )
        return orders, buy_order_volume, sell_order_volume

    # --- Strategy methods for specific products ---
    def make_resin_orders( self, order_depth: OrderDepth, fair_value: float, position: int, buy_order_volume: int, sell_order_volume: int, ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []; params = self.params[Product.RAINFOREST_RESIN]
        bid_price = fair_value - params["take_width"] - params["clear_width"]
        ask_price = fair_value + params["take_width"] + params["clear_width"]
        _, _ = self.market_make(Product.RAINFOREST_RESIN, orders, bid_price, ask_price, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> Optional[float]:
        if not order_depth.sell_orders or not order_depth.buy_orders: return None
        params = self.params[Product.KELP]; adverse_vol = params["adverse_volume"]
        mm_asks = [p for p, v in order_depth.sell_orders.items() if abs(v) >= adverse_vol]; mm_bids = [p for p, v in order_depth.buy_orders.items() if abs(v) >= adverse_vol]
        mm_ask = min(mm_asks) if mm_asks else None; mm_bid = max(mm_bids) if mm_bids else None
        last_price = traderObject.get("kelp_last_price")
        if mm_ask is None or mm_bid is None:
            if last_price is not None: mmmid_price = last_price
            else: best_ask = min(order_depth.sell_orders.keys()); best_bid = max(order_depth.buy_orders.keys()); mmmid_price = (best_ask + best_bid) / 2.0
        else: mmmid_price = (mm_ask + mm_bid) / 2.0
        if last_price is not None and last_price != 0:
            last_returns = (mmmid_price - last_price) / last_price; pred_returns = last_returns * params["reversion_beta"]; fair = mmmid_price * (1 + pred_returns)
        else: fair = mmmid_price
        traderObject["kelp_last_price"] = mmmid_price
        return fair

    def make_kelp_orders( self, order_depth: OrderDepth, fair_value: float, min_edge: float, position: int, buy_order_volume: int, sell_order_volume: int, ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []; bid_price = fair_value - min_edge; ask_price = fair_value + min_edge
        _, _ = self.market_make(Product.KELP, orders, bid_price, ask_price, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    # --- Basket and Spread Logic (Using NEW Product Names) ---
    def get_synthetic_basket_order_depth( self, basket_product: str, order_depths: Dict[str, OrderDepth] ) -> OrderDepth:
        synthetic_depth = OrderDepth(); basket_spec = BASKET_WEIGHTS.get(basket_product)
        if not basket_spec: return synthetic_depth
        comp_bids = {}; comp_asks = {}; comp_bid_vols = {}; comp_ask_vols = {}
        max_buy_cap = float('inf'); max_sell_cap = float('inf')
        for product, weight in basket_spec.items():
            if product not in order_depths: return OrderDepth()
            depth = order_depths[product];
            if not depth.buy_orders or not depth.sell_orders: return OrderDepth()
            b_bid = max(depth.buy_orders.keys()); b_ask = min(depth.sell_orders.keys())
            bid_v = abs(depth.buy_orders[b_bid]); ask_v = abs(depth.sell_orders[b_ask])
            comp_bids[product] = b_bid; comp_asks[product] = b_ask; comp_bid_vols[product] = bid_v; comp_ask_vols[product] = ask_v
            if weight == 0: continue
            max_buy_cap = min(max_buy_cap, ask_v // weight if weight else float('inf'))
            max_sell_cap = min(max_sell_cap, bid_v // weight if weight else float('inf'))
        implied_bid = sum(comp_bids[p] * w for p, w in basket_spec.items()); implied_ask = sum(comp_asks[p] * w for p, w in basket_spec.items())
        if max_sell_cap > 0 and max_sell_cap != float('inf') : synthetic_depth.buy_orders[round(implied_bid)] = round(max_sell_cap)
        if max_buy_cap > 0 and max_buy_cap != float('inf') : synthetic_depth.sell_orders[round(implied_ask)] = -round(max_buy_cap)
        return synthetic_depth

    def convert_synthetic_basket_orders( self, basket_product: str, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth] ) -> Dict[str, List[Order]]:
        basket_spec = BASKET_WEIGHTS.get(basket_product);
        if not basket_spec: return {}
        component_orders: Dict[str, List[Order]] = {prod: [] for prod in basket_spec}
        for order in synthetic_orders:
            qty = order.quantity; comp_prices = {}; all_avail = True
            for product, weight in basket_spec.items():
                depth = order_depths.get(product);
                if not depth: all_avail = False; break
                if qty > 0: # Buy synthetic -> hit asks
                    if not depth.sell_orders: all_avail = False; break
                    comp_prices[product] = min(depth.sell_orders.keys())
                else: # Sell synthetic -> hit bids
                    if not depth.buy_orders: all_avail = False; break
                    comp_prices[product] = max(depth.buy_orders.keys())
            if not all_avail: continue
            for product, weight in basket_spec.items():
                 comp_qty = qty * weight; comp_price = comp_prices[product]
                 # Only add order if quantity is non-zero
                 if abs(round(comp_qty)) > 0:
                     component_orders[product].append(Order(product, comp_price, round(comp_qty)))
        return component_orders

    def execute_spread_orders(self, basket_product: str, target_basket_pos: int, current_basket_pos: int, order_depths: Dict[str, OrderDepth]):
        if target_basket_pos == current_basket_pos:
            return None
    
        qty_needed = target_basket_pos - current_basket_pos
        basket_depth = order_depths.get(basket_product)
        if not basket_depth or not basket_depth.buy_orders or not basket_depth.sell_orders:
            logger.print(f"[Spread:{basket_product}] Missing basket depth.")
            return None
    
        b_ask = min(basket_depth.sell_orders.keys())
        b_bid = max(basket_depth.buy_orders.keys())
        agg_orders: Dict[str, List[Order]] = {basket_product: []}
        trade_vol = 0
    
        if qty_needed > 0:  # Buy basket
            b_price = b_ask
            avail_b_vol = abs(basket_depth.sell_orders[b_price])
            trade_vol = min(qty_needed, avail_b_vol)
            if trade_vol > 0:
                agg_orders[basket_product].append(Order(basket_product, b_price, trade_vol))
    
        elif qty_needed < 0:  # Sell basket
            b_price = b_bid
            avail_b_vol = abs(basket_depth.buy_orders[b_price])
            trade_vol = min(abs(qty_needed), avail_b_vol)
            if trade_vol > 0:
                agg_orders[basket_product].append(Order(basket_product, b_price, -trade_vol))
    
        if trade_vol == 0:
            logger.print(f"[Spread:{basket_product}] No volume.")
            return None
    
        logger.print(f"[Spread:{basket_product}] Executing Qty={trade_vol if qty_needed > 0 else -trade_vol}")
        return agg_orders


    def calculate_basket_spread(self, basket_product: str, order_depths: Dict[str, OrderDepth], spread_data: Dict[str, Any], params: Dict) -> Optional[float]:
        basket_depth = order_depths.get(basket_product)
        synthetic_depth = self.get_synthetic_basket_order_depth(basket_product, order_depths)
        if not basket_depth or not synthetic_depth: return None
        b_swmid = get_swmid(basket_depth); s_swmid = get_swmid(synthetic_depth)
        if b_swmid is None or s_swmid is None: logger.print(f"[Spread:{basket_product}] SWMID fail."); return None
        spread = b_swmid - s_swmid;
        # Ensure history list exists before appending
        if "spread_history" not in spread_data: spread_data["spread_history"] = []
        spread_data["spread_history"].append(spread)
        if len(spread_data["spread_history"]) > params["spread_std_window"]: spread_data["spread_history"].pop(0)
        if len(spread_data["spread_history"]) < params["spread_std_window"]: return None
        spread_mean = params["default_spread_mean"]; spread_std = np.std(spread_data["spread_history"])
        if spread_std < 1e-6: return None
        zscore = (spread - spread_mean) / spread_std;
        # logger.print(f"[{basket_product} Spread] Z={zscore:.2f}") # Reduce log spam
        spread_data["prev_zscore"] = zscore
        return zscore

    def picnic1_spread_orders( self, order_depths: Dict[str, OrderDepth], current_basket1_pos: int, spread_data: Dict[str, Any], ):
        basket_product = Product.PICNIC_BASKET1; params_key = "PICNIC_BASKET1_SPREAD"
        params = self.params.get(params_key)
        if not params: logger.print(f"[{basket_product}] Params not found."); return None
        zscore = self.calculate_basket_spread(basket_product, order_depths, spread_data, params)
        if zscore is None: return None

        target_pos = current_basket1_pos
        if zscore > params["zscore_threshold"]: target_pos = -params["target_position"]
        elif zscore < -params["zscore_threshold"]: target_pos = params["target_position"]
        if target_pos != current_basket1_pos:
             logger.print(f"[{basket_product} Spread] Target:{target_pos}, Curr:{current_basket1_pos}, Z={zscore:.2f}")
             return self.execute_spread_orders(basket_product, target_pos, current_basket1_pos, order_depths)
        return None

    # *** ADDED: Picnic Basket 2 Spread Logic ***
    def picnic2_spread_orders( self, order_depths: Dict[str, OrderDepth], current_basket2_pos: int, spread_data: Dict[str, Any], ):
        basket_product = Product.PICNIC_BASKET2; params_key = "PICNIC_BASKET2_SPREAD"
        params = self.params.get(params_key)
        if not params: logger.print(f"[{basket_product}] Params not found."); return None
        zscore = self.calculate_basket_spread(basket_product, order_depths, spread_data, params)
        if zscore is None: return None

        target_pos = current_basket2_pos
        if zscore > params["zscore_threshold"]: target_pos = -params["target_position"]
        elif zscore < -params["zscore_threshold"]: target_pos = params["target_position"]
        if target_pos != current_basket2_pos:
             logger.print(f"[{basket_product} Spread] Target:{target_pos}, Curr:{current_basket2_pos}, Z={zscore:.2f}")
             return self.execute_spread_orders(basket_product, target_pos, current_basket2_pos, order_depths)
        return None


    # --- Volcanic Rock Voucher Trading Logic (Using Individual PARAMS) ---
    def trade_vouchers( self, state: TradingState, traderObject: Dict[str, Any] ) -> Dict[str, List[Order]]:
        voucher_orders: Dict[str, List[Order]] = {symbol: [] for symbol in VOUCHER_SYMBOLS}
        # Access shared parameters from the instance variable
        shared_params = self.shared_voucher_params
        r = shared_params["r"]
        global_fallback_sigma = shared_params["global_fallback_sigma"]

        St = get_mid_price(Product.VOLCANIC_ROCK, state)
        if St is None: logger.print("[Vouchers] No VOLC price."); return voucher_orders
        # Pass shared time parameters to calculate TTE
        TTE = calculate_tte(state.timestamp, shared_params)
        if TTE <= 1e-9: logger.print("[Vouchers] TTE <= 0."); return voucher_orders

        # Calculate sigma_to_use (ATM IV or global fallback)
        sigma_to_use = global_fallback_sigma; atm_iv = None
        atm_info = find_atm_voucher(St, state)
        if atm_info:
            atm_symbol, atm_K = atm_info; Vt_atm = get_mid_price(atm_symbol, state)
            if Vt_atm is not None:
                iv = self.bs.implied_volatility(Vt_atm, St, atm_K, TTE, r)
                if iv is not None: sigma_to_use = iv; atm_iv = iv; logger.print(f"[Vouchers] ATM IV: {sigma_to_use:.4f}")
                else: logger.print(f"[Vouchers] ATM IV fail. Fallback.")
            else: logger.print(f"[Vouchers] No ATM Vt. Fallback.")
        else: logger.print(f"[Vouchers] No ATM found. Fallback.")

        traderObject["vouchers_atm_iv"] = atm_iv; traderObject["vouchers_used_sigma"] = sigma_to_use
        traderObject["vouchers_fair_values"] = {}

        for symbol, info in VOUCHER_INFO.items():
            if symbol not in state.listings: continue
            K = info["strike"]; limit = info["limit"]
            orders: List[Order] = []

            # Get parameters specific to this voucher symbol from self.params
            voucher_params = self.params.get(symbol)
            if not voucher_params: logger.print(f"[{symbol}] Warning: Missing params. Skipping."); continue

            # Use per-voucher thresholds and size
            trade_thresh = voucher_params["trade_threshold_abs"]
            exit_thresh = voucher_params["exit_threshold_abs"]
            base_order_size = voucher_params["base_order_size_voucher"]

            fair_value = self.bs.black_scholes_call(St, K, TTE, sigma_to_use, r)

            if fair_value is None: logger.print(f"[{symbol}] FV fail."); continue
            traderObject["vouchers_fair_values"][symbol] = round(fair_value, 2)

            Vt_market = get_mid_price(symbol, state); current_pos = state.position.get(symbol, 0)
            order_depth = state.order_depths.get(symbol)
            if Vt_market is None or order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders: logger.print(f"[{symbol}] Market data fail."); continue

            best_bid = max(order_depth.buy_orders.keys()); best_ask = min(order_depth.sell_orders.keys())
            diff = Vt_market - fair_value; exit_placed = False

            # Exit
            if current_pos > 0 and diff > -exit_thresh: # Long exit
                qty = abs(current_pos); orders.append(Order(symbol, best_bid, -qty)); logger.print(f"[{symbol}] EXIT LONG {qty}"); exit_placed = True
            elif current_pos < 0 and diff < exit_thresh: # Short exit
                qty = abs(current_pos); orders.append(Order(symbol, best_ask, qty)); logger.print(f"[{symbol}] EXIT SHORT {qty}"); exit_placed = True
            # Entry
            if not exit_placed:
                 current_pos = state.position.get(symbol, 0) # Re-check position
                 if diff < -trade_thresh: # Buy
                     qty = min(base_order_size, limit - current_pos);
                     if qty > 0: orders.append(Order(symbol, best_ask, qty)); logger.print(f"[{symbol}] ENTRY LONG {qty}")
                 elif diff > trade_thresh: # Sell
                     qty = min(base_order_size, limit + current_pos);
                     if qty > 0: orders.append(Order(symbol, best_bid, -qty)); logger.print(f"[{symbol}] ENTRY SHORT {qty}")

            if orders: voucher_orders[symbol].extend(orders)
        return voucher_orders


    # === Main Run Method ===
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData:
            try:
                decoded_data = json.loads(state.traderData) # USE STANDARD JSON
                if isinstance(decoded_data, dict): traderObject = decoded_data
                else: logger.print("Warn: traderData non-dict. Init empty.")
            except Exception as e: logger.print(f"Error decode traderData: {e}. Init empty.")

        result: Dict[str, List[Order]] = {}
        conversions = 0

        # --- Initialize traderObject state ---
        if "kelp_last_price" not in traderObject: traderObject["kelp_last_price"] = None
        if "picnic1_spread_data" not in traderObject: traderObject["picnic1_spread_data"] = {"spread_history": [], "prev_zscore": 0}
        if "picnic2_spread_data" not in traderObject: traderObject["picnic2_spread_data"] = {"spread_history": [], "prev_zscore": 0} # Initialize for basket 2


        # --- Execute logic ---
        # 1. Rainforest Resin
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            prod = Product.RAINFOREST_RESIN; params = self.params[prod]; pos = state.position.get(prod, 0); od = state.order_depths[prod]
            fv = float(params["fair_value"]);
            take_orders, buy_vol, sell_vol = self.take_orders(prod, od, fv, params["take_width"], pos)
            clear_orders, buy_vol, sell_vol = self.clear_orders(prod, od, fv, params["clear_width"], pos, buy_vol, sell_vol)
            make_orders, _, _ = self.make_resin_orders(od, fv, pos, buy_vol, sell_vol)
            result[prod] = take_orders + clear_orders + make_orders

        # 2. Kelp
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            prod = Product.KELP; params = self.params[prod]; pos = state.position.get(prod, 0); od = state.order_depths[prod]
            fv = self.kelp_fair_value(od, traderObject)
            if fv is not None:
                take_orders, buy_vol, sell_vol = self.take_orders(prod, od, fv, params["take_width"], pos, params["prevent_adverse"], params["adverse_volume"])
                clear_orders, buy_vol, sell_vol = self.clear_orders(prod, od, fv, params["clear_width"], pos, buy_vol, sell_vol)
                make_orders, _, _ = self.make_kelp_orders(od, fv, params["kelp_min_edge"], pos, buy_vol, sell_vol)
                result[prod] = take_orders + clear_orders + make_orders
            else: logger.print(f"[{prod}] FV fail.")

        # 3. Picnic Basket 1 Spread Trade
        basket1_comps = list(BASKET_WEIGHTS[Product.PICNIC_BASKET1].keys())
        if Product.PICNIC_BASKET1 in state.listings and all(c in state.order_depths for c in basket1_comps):
            b1_pos = state.position.get(Product.PICNIC_BASKET1, 0)
            spread_exec_orders_b1 = self.picnic1_spread_orders(state.order_depths, b1_pos, traderObject["picnic1_spread_data"])
            if spread_exec_orders_b1:
                for product, orders in spread_exec_orders_b1.items(): result[product] = result.get(product, []) + orders
        else: logger.print(f"[{Product.PICNIC_BASKET1} Spread] Skip - missing data.")

        # 4. Picnic Basket 2 Spread Trade *** ADDED CALL ***
        basket2_comps = list(BASKET_WEIGHTS[Product.PICNIC_BASKET2].keys())
        if Product.PICNIC_BASKET2 in state.listings and all(c in state.order_depths for c in basket2_comps):
             b2_pos = state.position.get(Product.PICNIC_BASKET2, 0)
             spread_exec_orders_b2 = self.picnic2_spread_orders(state.order_depths, b2_pos, traderObject["picnic2_spread_data"]) # Use separate data key
             if spread_exec_orders_b2:
                 for product, orders in spread_exec_orders_b2.items(): result[product] = result.get(product, []) + orders
        else: logger.print(f"[{Product.PICNIC_BASKET2} Spread] Skip - missing data.")

        # 5. Volcanic Rock Vouchers Trade
        if Product.VOLCANIC_ROCK in state.order_depths:
            voucher_orders = self.trade_vouchers(state, traderObject)
            for product, orders in voucher_orders.items():
                result[product] = result.get(product, []) + orders
        else: logger.print("[Vouchers] Skip - VOLCANIC_ROCK missing.")

        if Product.MAGNIFICENT_MACARONS in self.params and Product.MAGNIFICENT_MACARONS in state.order_depths:
                    prod = Product.MAGNIFICENT_MACARONS
                    params = self.params[prod]  # Get parameters for macarons
                    pos = state.position.get(prod, 0)
                    od = state.order_depths[prod]
                    
                    # Check if we have conversion observations for macarons
                    if prod in state.observations.conversionObservations:
                        observation = state.observations.conversionObservations[prod]
                        
                        # Execute arbitrage take orders
                        macarons_take_orders, buy_vol, sell_vol = self.macarons_arb_take(
                            od,
                            observation,
                            pos,
                            params
                        )
                        
                        # Execute market making orders
                        macarons_make_orders, _, _ = self.macarons_arb_make(
                            observation,
                            pos,
                            buy_vol,
                            sell_vol,
                            params
                        )
                        
                        # Add conversion request
                        macarons_conversions = self.macarons_arb_clear(pos, params)
                        conversions = macarons_conversions
                        
                        # Combine all orders
                        result[prod] = macarons_take_orders + macarons_make_orders
                        
                        # Log trading info
                        logger.print(f"[{prod}] Position: {pos}, Sunlight: {observation.sunlightIndex:.2f}, Sugar: {observation.sugarPrice:.2f}")
                        
                        # Log if we're below CSI threshold for debugging
                        if observation.sunlightIndex < params["csi_threshold"]:
                            logger.print(f"[{prod}] ALERT: Below CSI! Sunlight Index: {observation.sunlightIndex:.2f}")
                    else:
                        logger.print(f"[{prod}] No conversion observations available")

        # --- Encode TraderData and Return ---
        traderDataString = ""
        try:
            traderDataString = json.dumps(traderObject) # USE STANDARD JSON
        except Exception as e: logger.print(f"Error encoding traderData: {e}")

        # *** Pass the DICTIONARY to the logger ***
        logger.flush(state, result, conversions, traderDataString)

        # Return the STRING for the next iteration's state.traderData
        return result, conversions, traderDataString


