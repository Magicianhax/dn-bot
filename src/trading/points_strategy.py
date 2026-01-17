"""Points farming strategy - Long/Short on two accounts."""

import asyncio
import random
import time
import uuid
from decimal import Decimal
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ethereal import AsyncRESTClient
from eth_account import Account

from ..config import Settings, get_settings, save_active_trades, load_active_trades, clear_active_trades
from ..utils.logger import get_logger

logger = get_logger("ethereal.points")


class TradeStatus(Enum):
    """Trade status."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TIME = "CLOSED_TIME"
    CLOSED_MANUAL = "CLOSED_MANUAL"
    FAILED = "FAILED"


@dataclass
class TradePair:
    """Represents a long/short trade pair across two accounts."""
    id: str
    product_id: str
    size: float
    leverage: int
    entry_price: float
    long_order_id: Optional[str] = None
    short_order_id: Optional[str] = None
    status: TradeStatus = TradeStatus.PENDING
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    close_reason: str = ""
    pnl_long: float = 0.0
    pnl_short: float = 0.0
    account1_is_long: bool = True  # Track which account is long for closing
    target_hold_minutes: float = 0.0  # Random hold time target for this trade
    # Exchange-side TP/SL order IDs
    long_tp_order_id: Optional[str] = None
    short_tp_order_id: Optional[str] = None
    long_sl_order_id: Optional[str] = None
    short_sl_order_id: Optional[str] = None
    long_entry_price: float = 0.0  # Actual entry price for long
    short_entry_price: float = 0.0  # Actual entry price for short
    # OCO group IDs for linked TP/SL orders
    long_oco_group: Optional[str] = None
    short_oco_group: Optional[str] = None

    @property
    def total_pnl(self) -> float:
        return self.pnl_long + self.pnl_short

    @property
    def hold_time_minutes(self) -> float:
        end = self.closed_at or datetime.now()
        return (end - self.opened_at).total_seconds() / 60


@dataclass
class StrategyStats:
    """Strategy statistics."""
    total_trades: int = 0
    daily_trades: int = 0
    daily_trades_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    next_trade_delay: float = 60.0  # Random delay for next trade


class AccountClient:
    """Wrapper for a single Ethereal account."""

    def __init__(self, private_key: str, wallet_address: str, api_url: str):
        self.private_key = private_key
        self.wallet_address = wallet_address  # EOA address
        self.api_url = api_url
        self._client: Optional[AsyncRESTClient] = None
        self._subaccount_id: Optional[str] = None
        self._subaccount_name: str = "primary"
        self._subaccount_name_raw: Optional[str] = None
        self._subaccount: Optional[object] = None
        self._is_main_wallet: bool = False
        self._signer_address: Optional[str] = None  # Address derived from private key
        self._connected = False
        self._products: dict[str, dict] = {}  # ticker -> product info

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    async def connect(self, rpc_url: str = "https://rpc.ethereal.trade") -> None:
        """Connect to Ethereal API."""
        if self._connected:
            return

        # Check if private key derives to wallet address (main wallet vs linked signer)
        derived_address = Account.from_key(self.private_key).address
        is_main_wallet = derived_address.lower() == self.wallet_address.lower()
        
        # Store signer address - this is the address we use as "sender" in orders
        self._signer_address = derived_address

        if is_main_wallet:
            logger.info(f"Connecting main wallet {self.wallet_address[:10]}...")
        else:
            logger.info(f"Connecting linked signer for EOA {self.wallet_address[:10]}...")
            logger.info(f"  Signer address (used as sender): {derived_address[:10]}...")

        self._is_main_wallet = is_main_wallet
        self._client = await AsyncRESTClient.create({
            "base_url": self.api_url,
            "chain_config": {
                "rpc_url": rpc_url,
                "private_key": self.private_key,
            }
        })
        self._connected = True

        # Load products to get UUIDs
        try:
            products = await self._client.list_products()
            for p in products:
                ticker = getattr(p, 'ticker', None) or getattr(p, 'symbol', '')
                product_id = str(getattr(p, 'id', ''))
                onchain_id = getattr(p, 'onchain_id', None) or getattr(p, 'onchainId', None) or getattr(p, 'product_id_onchain', None)
                max_leverage = float(getattr(p, 'max_leverage', 10) or 10)
                if ticker and product_id:
                    self._products[ticker] = {
                        'id': product_id,
                        'ticker': ticker,
                        'onchain_id': onchain_id,
                        'lot_size': float(getattr(p, 'lot_size', 0.0001) or 0.0001),
                        'tick_size': float(getattr(p, 'tick_size', 0.01) or 0.01),
                        'max_leverage': int(max_leverage),
                    }
            # Log first product to see available fields
            if products:
                logger.info(f"Product fields: {vars(products[0]) if hasattr(products[0], '__dict__') else products[0]}")
            logger.info(f"Loaded {len(self._products)} products")
        except Exception as e:
            logger.error(f"Failed to load products: {e}")

        # Load subaccount - use list_subaccounts with sender (main wallet/EOA address)
        try:
            subs = await self._client.list_subaccounts(sender=self.wallet_address)
            if subs:
                self._subaccount_id = str(subs[0].id)
                self._subaccount = subs[0]
                
                # Get subaccount name - keep raw format for API calls
                raw_name = getattr(subs[0], 'name', None)
                self._subaccount_name_raw = raw_name  # Keep raw hex for API
                
                # Decode for display only
                display_name = raw_name
                if isinstance(raw_name, str) and raw_name.startswith('0x'):
                    try:
                        display_name = bytes.fromhex(raw_name[2:]).rstrip(b'\x00').decode('utf-8')
                    except:
                        display_name = raw_name[:20] + "..."
                
                self._subaccount_name = display_name or 'primary'
                logger.info(f"Found subaccount: {self._subaccount_id[:8]}... (name: {self._subaccount_name}, raw: {str(raw_name)[:20]}...)")
                
                if not self._is_main_wallet:
                    signer_address = Account.from_key(self.private_key).address
                    logger.info(f"Using linked signer: {signer_address[:10]}... for EOA: {self.wallet_address[:10]}...")
                    
                    # Check if signer is linked and not expired
                    try:
                        from uuid import UUID
                        signers = await self._client.list_signers(
                            sender=self.wallet_address,
                            subaccount_id=UUID(self._subaccount_id),
                        )
                        signer_addresses = [getattr(s, 'signer', '') for s in signers] if signers else []
                        logger.info(f"Linked signers for this subaccount: {signer_addresses}")
                        
                        # Check for this specific signer and its status
                        signer_found = False
                        for s in signers or []:
                            s_addr = getattr(s, 'signer', '')
                            if s_addr.lower() == signer_address.lower():
                                signer_found = True
                                # Check expiration
                                expires_at = getattr(s, 'expires_at', None)
                                revoked = getattr(s, 'revoked', False)
                                logger.info(f"Signer details: address={s_addr[:10]}, expires_at={expires_at}, revoked={revoked}")
                                if revoked:
                                    logger.error(f"SIGNER REVOKED! {signer_address}")
                                elif expires_at:
                                    import time
                                    current_time = int(time.time() * 1000)  # milliseconds
                                    if isinstance(expires_at, (int, float)) and expires_at < current_time:
                                        logger.error(f"SIGNER EXPIRED! Expires at {expires_at}, current time {current_time}")
                                    else:
                                        logger.info(f"Signer {signer_address[:10]}... is properly linked and valid!")
                                else:
                                    logger.info(f"Signer {signer_address[:10]}... is properly linked (no expiry)!")
                                break
                        
                        if not signer_found:
                            logger.error(f"SIGNER NOT LINKED! {signer_address} is not in the linked signers list.")
                            logger.error("Please link this signer on https://ethereal.trade -> Settings -> Linked Signers")
                    except Exception as e:
                        logger.warning(f"Could not check linked signers: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                logger.warning(f"No subaccount found for {self.wallet_address[:10]}")
        except Exception as e:
            logger.error(f"Failed to load subaccount: {e}")

    async def disconnect(self) -> None:
        """Disconnect from API."""
        if self._client:
            await self._client.close()
        self._client = None
        self._connected = False

    async def get_balance(self) -> float:
        """Get available balance."""
        if not self._subaccount_id:
            return 0.0

        try:
            balances = await self._client.get_subaccount_balances(
                subaccount_id=self._subaccount_id,
                sender=self.wallet_address,
            )
            total = 0.0
            for b in balances:
                amt = getattr(b, 'amount', None) or getattr(b, 'balance', 0)
                total += float(amt or 0)
            return total
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    async def list_positions(self) -> list:
        """List all open positions for this account."""
        if not self._client or not self._subaccount_id:
            return []
        
        try:
            positions = await self._client.list_positions(
                subaccount_id=self._subaccount_id,
            )
            result = []
            for p in positions:
                size = float(getattr(p, 'size', 0) or 0)
                if abs(size) > 0:  # Only include non-zero positions
                    # Get product UUID and find ticker
                    product_id = str(getattr(p, 'product_id', ''))
                    ticker = product_id  # Default to UUID
                    
                    # Try to find the ticker name from product_id
                    for t, info in self._products.items():
                        if str(info.get('id', '')) == product_id:
                            ticker = t
                            break
                    
                    # Calculate entry price from cost
                    cost = float(getattr(p, 'cost', 0) or 0)
                    entry_price = abs(cost / size) if size != 0 else 0
                    
                    # Get mark price for unrealized PnL calculation
                    mark_price = await self.get_mark_price(ticker) if ticker else 0
                    
                    # Calculate unrealized PnL
                    # LONG: (mark - entry) * size
                    # SHORT: (entry - mark) * abs(size)
                    if size > 0:  # LONG
                        unrealized_pnl = (mark_price - entry_price) * size
                    else:  # SHORT
                        unrealized_pnl = (entry_price - mark_price) * abs(size)
                    
                    # Calculate notional
                    notional = abs(size) * mark_price if mark_price > 0 else abs(cost)
                    
                    result.append({
                        "ticker": ticker,
                        "product_id": product_id,
                        "size": size,
                        "entry_price": entry_price,
                        "mark_price": mark_price,
                        "unrealized_pnl": unrealized_pnl,
                        "notional": notional,
                        "side": "LONG" if size > 0 else "SHORT",
                    })
            return result
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_product_uuid(self, ticker: str) -> Optional[str]:
        """Get product UUID from ticker."""
        product = self._products.get(ticker)
        return product['id'] if product else None

    def get_lot_size(self, ticker: str) -> float:
        """Get lot size for a product."""
        product = self._products.get(ticker)
        return float(product['lot_size']) if product else 0.0001

    def get_tick_size(self, ticker: str) -> float:
        """Get tick size (price increment) for a product."""
        product = self._products.get(ticker)
        return float(product['tick_size']) if product else 0.01

    def get_max_leverage(self, ticker: str) -> int:
        """Get max leverage for a product."""
        product = self._products.get(ticker)
        return int(product.get('max_leverage', 10)) if product else 10

    def get_all_products(self) -> list[dict]:
        """Get all products with their info (for API)."""
        return [
            {
                'ticker': ticker,
                'max_leverage': info.get('max_leverage', 10),
                'lot_size': info.get('lot_size', 0.0001),
                'tick_size': info.get('tick_size', 0.01),
            }
            for ticker, info in self._products.items()
        ]

    def round_price_to_tick(self, price: float, ticker: str) -> float:
        """Round price to the nearest tick size for a product."""
        tick_size = self.get_tick_size(ticker)
        return round(round(price / tick_size) * tick_size, 10)

    async def get_mark_price(self, ticker: str) -> float:
        """Get current mark price."""
        if not self._client:
            logger.error("Client not connected!")
            return 0.0

        try:
            product_uuid = self.get_product_uuid(ticker)
            if not product_uuid:
                logger.error(f"Unknown product: {ticker}. Available: {list(self._products.keys())}")
                return 0.0

            prices = await self._client.list_market_prices(product_ids=[product_uuid])

            if prices and len(prices) > 0:
                price = prices[0]
                # Use oracle_price as mark price, or mid of bid/ask
                oracle = getattr(price, 'oracle_price', None)
                if oracle:
                    mark = float(oracle)
                else:
                    bid = float(getattr(price, 'best_bid_price', 0) or 0)
                    ask = float(getattr(price, 'best_ask_price', 0) or 0)
                    mark = (bid + ask) / 2 if bid and ask else 0

                if mark > 0:
                    return mark

            logger.warning(f"No price data returned for {ticker}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get price for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    async def get_current_price(self, ticker_or_uuid: str) -> float:
        """Get current price. Accepts either ticker (BTCUSD) or product UUID."""
        # Try to find ticker if UUID was passed
        if '-' in ticker_or_uuid:  # Looks like a UUID
            # Find the ticker for this UUID
            for t, info in self._products.items():
                if str(info.get('id', '')) == ticker_or_uuid:
                    return await self.get_mark_price(t)
            # Fallback: try getting price by UUID directly
            return await self.get_mark_price(ticker_or_uuid)
        else:
            return await self.get_mark_price(ticker_or_uuid)

    async def open_position(self, ticker: str, side: str, size: float, limit_price: Optional[float] = None) -> Optional[str]:
        """Open a position. Returns order ID if successful.

        Args:
            ticker: Product ticker (e.g., "BTCUSD")
            side: "buy" or "sell"
            size: Position size
            limit_price: If provided, uses limit order at this price. Otherwise uses market order.
        """
        if not self._subaccount_id:
            logger.error("No subaccount ID")
            return None

        try:
            # side: 0 = buy, 1 = sell
            side_int = 0 if side.lower() == "buy" else 1
            side_name = "BUY" if side_int == 0 else "SELL"

            # Determine order type
            order_type = "LIMIT" if limit_price else "MARKET"
            price_str = f" @ ${limit_price:,.2f}" if limit_price else ""
            logger.info(f"Opening {side_name} {size:.6f} {ticker}{price_str} ({order_type})")

            # For linked signers, the sender MUST be the signer address (derived from private key)
            # NOT the EOA/wallet address! The backend verifies signature matches sender.
            subaccount_param = self._subaccount_name_raw or self._subaccount_name

            # Use signer address as sender (critical for linked signers!)
            sender_address = self._signer_address

            logger.info(f"Order: sender={sender_address[:10]}..., subaccount={subaccount_param[:20] if subaccount_param else 'None'}...")

            # Build order params
            order_params = {
                "order_type": order_type,
                "ticker": ticker,
                "side": side_int,
                "quantity": Decimal(str(size)),
                "sender": sender_address,
                "subaccount": subaccount_param,
            }

            # Add price for limit orders
            if limit_price:
                # Round price to tick size
                rounded_price = self.round_price_to_tick(limit_price, ticker)
                order_params["price"] = Decimal(str(rounded_price))
                # Use IOC (Immediate-Or-Cancel) for limit orders to ensure quick fill or cancel
                order_params["time_in_force"] = "IOC"

            response = await self._client.create_order(**order_params)

            # Check order result
            order_id = str(getattr(response, 'id', '')) or str(getattr(response, 'order_id', ''))
            result = getattr(response, 'result', None)
            filled = getattr(response, 'filled', '0')

            logger.info(f"Order response: id={order_id[:8] if order_id else 'N/A'}, result={result}, filled={filled}")

            # Check if order was successful
            if result and 'ok' in str(result).lower():
                logger.info(f"Order SUCCESS: {side_name} {size} {ticker} filled={filled}")
                return order_id or "success"
            else:
                logger.error(f"Order FAILED: {side_name} {size} {ticker} - result={result}")
                return None
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def close_position(self, ticker: str, limit_price: Optional[float] = None) -> bool:
        """Close position for a product.

        Args:
            ticker: Product ticker
            limit_price: If provided, uses limit order at this price. Otherwise uses market order.
        """
        if not self._subaccount_id:
            logger.error(f"Cannot close {ticker}: No subaccount")
            return False

        try:
            # Get current position - need to match by product_id since ticker might not be in response
            positions = await self._client.list_positions(
                subaccount_id=self._subaccount_id,
                sender=self.wallet_address,
            )

            # Get product info
            product = self._products.get(ticker, {})
            product_id = product.get('id', '')

            position_found = False
            size = 0.0

            for pos in positions:
                pos_product_id = str(getattr(pos, 'product_id', ''))
                pos_ticker = getattr(pos, 'ticker', None)

                # Match by product_id or ticker
                if pos_product_id == product_id or pos_ticker == ticker:
                    size = float(getattr(pos, 'quantity', 0) or getattr(pos, 'size', 0) or 0)
                    if size != 0:
                        position_found = True
                        break

            if not position_found or size == 0:
                logger.info(f"Position {ticker} already closed or not found (size=0)")
                return True

            # Close with opposite side: 0=buy, 1=sell
            side_int = 1 if size > 0 else 0  # sell if long, buy if short
            side_name = "SELL" if size > 0 else "BUY"
            subaccount_param = self._subaccount_name_raw or self._subaccount_name

            # Determine order type
            order_type = "LIMIT" if limit_price else "MARKET"
            price_str = f" @ ${limit_price:,.2f}" if limit_price else ""
            logger.info(f"Closing {ticker}: size={size:.6f}, side={side_name}{price_str} ({order_type})")

            # Build order params
            order_params = {
                "order_type": order_type,
                "ticker": ticker,
                "side": side_int,
                "quantity": Decimal(str(abs(size))),
                "reduce_only": True,
                "sender": self._signer_address,
                "subaccount": subaccount_param,
            }

            # Add price for limit orders
            if limit_price:
                # Round price to tick size
                rounded_price = self.round_price_to_tick(limit_price, ticker)
                order_params["price"] = Decimal(str(rounded_price))
                # Use IOC (Immediate-Or-Cancel) for limit orders to ensure quick fill or cancel
                order_params["time_in_force"] = "IOC"

            result = await self._client.create_order(**order_params)

            # Check result
            result_status = getattr(result, 'result', None)
            filled = getattr(result, 'filled', '0')

            if result_status and 'ok' in str(result_status).lower():
                logger.info(f"Closed {ticker} position: filled={filled}")
                return True
            else:
                logger.error(f"Close order result: {result_status}, filled={filled}")
                return False

        except Exception as e:
            logger.error(f"Failed to close {ticker} position: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def get_position_pnl(self, ticker_or_uuid: str) -> float:
        """Get unrealized PnL for a position. Calculates from cost and mark price."""
        if not self._subaccount_id:
            return 0.0

        try:
            positions = await self._client.list_positions(
                subaccount_id=self._subaccount_id,
            )
            
            for pos in positions:
                product_id = str(getattr(pos, 'product_id', ''))
                
                # Match by ticker or product_id
                ticker = None
                for t, info in self._products.items():
                    if str(info.get('id', '')) == product_id:
                        ticker = t
                        break
                
                # Check if this is the position we're looking for
                if ticker_or_uuid not in [ticker, product_id]:
                    continue
                
                size = float(getattr(pos, 'size', 0) or 0)
                if abs(size) == 0:
                    continue
                
                # Calculate entry price from cost
                cost = float(getattr(pos, 'cost', 0) or 0)
                entry_price = abs(cost / size) if size != 0 else 0
                
                # Get mark price
                mark_price = await self.get_mark_price(ticker) if ticker else 0
                
                if mark_price <= 0 or entry_price <= 0:
                    return 0.0
                
                # Calculate unrealized PnL
                if size > 0:  # LONG
                    pnl = (mark_price - entry_price) * size
                else:  # SHORT
                    pnl = (entry_price - mark_price) * abs(size)
                
                return pnl
            
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get position PnL: {e}")
            return 0.0

    async def place_oco_tp_sl(
        self, 
        ticker: str, 
        is_long: bool,
        size: float, 
        tp_price: float, 
        sl_price: float,
        group_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Place OCO (One-Cancels-Other) TP and SL orders.
        
        When one order triggers, the other auto-cancels.
        
        stop_type meanings:
        - 0 = GAIN: triggers when price >= stop_price
        - 1 = LOSS: triggers when price <= stop_price

        For LONG (close by selling):
        - TP: Price rises -> stop_type=0 (GAIN, >=)
        - SL: Price falls -> stop_type=1 (LOSS, <=)
        
        For SHORT (close by buying):
        - TP: Price falls -> stop_type=1 (LOSS, <=) - INVERTED!
        - SL: Price rises -> stop_type=0 (GAIN, >=) - INVERTED!

        Args:
            ticker: Product ticker
            is_long: True if closing a LONG position, False if closing SHORT
            size: Position size to close
            tp_price: Take profit trigger price
            sl_price: Stop loss trigger price
            group_id: UUID to link TP and SL as OCO pair

        Returns:
            Tuple of (tp_order_id, sl_order_id), None for failed orders
        """
        if not self._subaccount_id:
            logger.error("No subaccount ID for OCO orders")
            return None, None

        # LONG closes by SELL (side=1), SHORT closes by BUY (side=0)
        side_int = 1 if is_long else 0
        side_name = "SELL" if is_long else "BUY"
        position_type = "LONG" if is_long else "SHORT"
        subaccount_param = self._subaccount_name_raw or self._subaccount_name

        # Round prices to tick size
        tp_rounded = self.round_price_to_tick(tp_price, ticker)
        sl_rounded = self.round_price_to_tick(sl_price, ticker)

        # Determine stop_types based on position direction
        # LONG: TP when price >= (GAIN=0), SL when price <= (LOSS=1)
        # SHORT: TP when price <= (LOSS=1), SL when price >= (GAIN=0)
        if is_long:
            tp_stop_type = 0  # GAIN - triggers when price >= tp_price
            sl_stop_type = 1  # LOSS - triggers when price <= sl_price
            tp_condition = ">="
            sl_condition = "<="
        else:
            tp_stop_type = 1  # LOSS - triggers when price <= tp_price (SHORT profits when falls)
            sl_stop_type = 0  # GAIN - triggers when price >= sl_price (SHORT loses when rises)
            tp_condition = "<="
            sl_condition = ">="

        logger.info(f"Placing OCO TP/SL for {position_type}: {side_name} {size:.6f} {ticker}")
        logger.info(f"  TP @ ${tp_rounded:,.2f} (price {tp_condition}) | SL @ ${sl_rounded:,.2f} (price {sl_condition})")
        logger.info(f"  Group: {group_id[:8]}...")

        tp_order_id = None
        sl_order_id = None

        try:
            # Place TP order (MARKET when triggered)
            tp_response = await self._client.create_order(
                order_type="MARKET",
                ticker=ticker,
                side=side_int,
                quantity=Decimal(str(size)),
                stop_price=Decimal(str(tp_rounded)),
                stop_type=tp_stop_type,
                reduce_only=True,
                group_id=group_id,
                group_contingency_type=1,  # 1 = OCO
                sender=self._signer_address,
                subaccount=subaccount_param,
            )

            tp_order_id = str(getattr(tp_response, 'id', '')) or str(getattr(tp_response, 'order_id', ''))
            tp_result = getattr(tp_response, 'result', None)

            if tp_result and 'ok' in str(tp_result).lower():
                logger.info(f"  TP [OK]: {tp_order_id[:8] if tp_order_id else 'N/A'}...")
            else:
                logger.error(f"  TP [FAIL]: {tp_result}")
                tp_order_id = None

        except Exception as e:
            logger.error(f"  TP ERROR: {e}")

        try:
            # Place SL order (MARKET when triggered)
            sl_response = await self._client.create_order(
                order_type="MARKET",
                ticker=ticker,
                side=side_int,
                quantity=Decimal(str(size)),
                stop_price=Decimal(str(sl_rounded)),
                stop_type=sl_stop_type,
                reduce_only=True,
                group_id=group_id,
                group_contingency_type=1,  # 1 = OCO
                sender=self._signer_address,
                subaccount=subaccount_param,
            )

            sl_order_id = str(getattr(sl_response, 'id', '')) or str(getattr(sl_response, 'order_id', ''))
            sl_result = getattr(sl_response, 'result', None)

            if sl_result and 'ok' in str(sl_result).lower():
                logger.info(f"  SL [OK]: {sl_order_id[:8] if sl_order_id else 'N/A'}...")
            else:
                logger.error(f"  SL [FAIL]: {sl_result}")
                sl_order_id = None

        except Exception as e:
            logger.error(f"  SL ERROR: {e}")

        return tp_order_id, sl_order_id

    async def place_tp_order(self, ticker: str, side: str, size: float, tp_price: float) -> Optional[str]:
        """Place a take-profit stop order on the exchange (legacy, non-OCO).

        Args:
            ticker: Product ticker
            side: "buy" or "sell" (opposite of position side)
            size: Position size to close
            tp_price: Take profit trigger price

        Returns:
            Order ID if successful, None otherwise
        """
        if not self._subaccount_id:
            logger.error("No subaccount ID for TP order")
            return None

        try:
            side_int = 0 if side.lower() == "buy" else 1
            side_name = "BUY" if side_int == 0 else "SELL"
            subaccount_param = self._subaccount_name_raw or self._subaccount_name

            # Round price to tick size
            rounded_price = self.round_price_to_tick(tp_price, ticker)
            logger.info(f"Placing TP {side_name} stop order: {size:.6f} {ticker} trigger @ ${rounded_price:,.2f}")

            response = await self._client.create_order(
                order_type="MARKET",  # Execute as market when triggered
                ticker=ticker,
                side=side_int,
                quantity=Decimal(str(size)),
                stop_price=Decimal(str(rounded_price)),  # Trigger price
                stop_type=0,  # 0 = take-profit
                reduce_only=True,
                sender=self._signer_address,
                subaccount=subaccount_param,
            )

            order_id = str(getattr(response, 'id', '')) or str(getattr(response, 'order_id', ''))
            result = getattr(response, 'result', None)

            if result and 'ok' in str(result).lower():
                logger.info(f"TP order placed: {side_name} {size} {ticker} trigger @ ${rounded_price:.2f}, ID={order_id[:8] if order_id else 'N/A'}")
                return order_id or "success"
            else:
                logger.error(f"TP order FAILED: {result}")
                return None
        except Exception as e:
            logger.error(f"Failed to place TP order: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def place_sl_order(self, ticker: str, side: str, size: float, sl_price: float) -> Optional[str]:
        """Place a stop-loss order on the exchange (legacy, non-OCO).

        Args:
            ticker: Product ticker
            side: "buy" or "sell" (opposite of position side)
            size: Position size to close
            sl_price: Stop loss trigger price

        Returns:
            Order ID if successful, None otherwise
        """
        if not self._subaccount_id:
            logger.error("No subaccount ID for SL order")
            return None

        try:
            side_int = 0 if side.lower() == "buy" else 1
            side_name = "BUY" if side_int == 0 else "SELL"
            subaccount_param = self._subaccount_name_raw or self._subaccount_name

            # Round price to tick size
            rounded_price = self.round_price_to_tick(sl_price, ticker)
            logger.info(f"Placing SL {side_name} stop order: {size:.6f} {ticker} trigger @ ${rounded_price:,.2f}")

            response = await self._client.create_order(
                order_type="MARKET",  # Execute as market when triggered
                ticker=ticker,
                side=side_int,
                quantity=Decimal(str(size)),
                stop_price=Decimal(str(rounded_price)),  # Trigger price
                stop_type=1,  # 1 = stop-loss
                reduce_only=True,
                sender=self._signer_address,
                subaccount=subaccount_param,
            )

            order_id = str(getattr(response, 'id', '')) or str(getattr(response, 'order_id', ''))
            result = getattr(response, 'result', None)

            if result and 'ok' in str(result).lower():
                logger.info(f"SL order placed: {side_name} {size} {ticker} trigger @ ${rounded_price:.2f}, ID={order_id[:8] if order_id else 'N/A'}")
                return order_id or "success"
            else:
                logger.error(f"SL order FAILED: {result}")
                return None
        except Exception as e:
            logger.error(f"Failed to place SL order: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def cancel_order(self, order_id: str, ticker: str = "") -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel
            ticker: Product ticker (optional, for logging only)

        Returns:
            True if cancelled, False otherwise
        """
        if not self._subaccount_id or not order_id:
            return False

        try:
            subaccount_param = self._subaccount_name_raw or self._subaccount_name
            logger.info(f"Cancelling order {order_id[:8]}...")

            # Use cancel_orders (plural) with order_ids list
            await self._client.cancel_orders(
                order_ids=[order_id],
                sender=self._signer_address,
                subaccount=subaccount_param,
            )
            logger.info(f"Order {order_id[:8]}... cancelled")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id[:8]}...: {e}")
            return False

    async def has_position(self, ticker: str) -> bool:
        """Check if there's an open position for the ticker.

        Returns:
            True if position exists with non-zero size
        """
        if not self._subaccount_id:
            return False

        try:
            positions = await self._client.list_positions(
                subaccount_id=self._subaccount_id,
                sender=self.wallet_address,
            )

            product = self._products.get(ticker, {})
            product_id = product.get('id', '')

            for pos in positions:
                pos_product_id = str(getattr(pos, 'product_id', ''))
                pos_ticker = getattr(pos, 'ticker', None)

                if pos_product_id == product_id or pos_ticker == ticker:
                    size = float(getattr(pos, 'quantity', 0) or getattr(pos, 'size', 0) or 0)
                    if abs(size) > 0:
                        return True
            return False
        except:
            return False


class PointsFarmingStrategy:
    """
    Points farming strategy.

    Opens opposing positions on two accounts (long on Account 1, short on Account 2)
    to farm points while maintaining delta neutrality.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # Accounts
        self.account1: Optional[AccountClient] = None  # Long side
        self.account2: Optional[AccountClient] = None  # Short side

        # State
        self.active_trades: dict[str, TradePair] = {}
        self.completed_trades: list[TradePair] = []
        self.stats = StrategyStats()
        self.running = False

        # Callbacks
        self._on_trade_open: Optional[Callable] = None
        self._on_trade_close: Optional[Callable] = None

    async def initialize(self) -> bool:
        """Initialize the strategy with both accounts."""
        if not self.settings.has_dual_accounts:
            logger.error("Dual accounts not configured. Set ACCOUNT1_* and ACCOUNT2_* in .env")
            return False

        logger.info("Initializing Points Farming Strategy...")

        # Initialize Account 1 (Long)
        self.account1 = AccountClient(
            private_key=self.settings.account1_private_key,
            wallet_address=self.settings.account1_wallet_address,
            api_url=self.settings.ethereal_api_url,
        )
        await self.account1.connect(rpc_url=self.settings.ethereal_rpc_url)

        # Initialize Account 2 (Short)
        self.account2 = AccountClient(
            private_key=self.settings.account2_private_key,
            wallet_address=self.settings.account2_wallet_address,
            api_url=self.settings.ethereal_api_url,
        )
        await self.account2.connect(rpc_url=self.settings.ethereal_rpc_url)

        # Check balances
        bal1 = await self.account1.get_balance()
        bal2 = await self.account2.get_balance()
        logger.info(f"Account 1 balance: ${bal1:.2f}")
        logger.info(f"Account 2 balance: ${bal2:.2f}")

        if bal1 == 0 or bal2 == 0:
            logger.warning("One or both accounts have zero balance!")

        # Resume existing positions
        await self._resume_existing_trades()

        return True

    async def _resume_existing_trades(self) -> None:
        """Detect and resume existing positions from both accounts."""
        if not self.account1 or not self.account2:
            return

        logger.info("Checking for existing positions to resume...")

        # Load saved trade data (for timing info)
        saved_trades = load_active_trades()
        saved_by_ticker = {}
        for trade_id, trade_data in saved_trades.items():
            ticker = trade_data.get("product_id", "")
            if ticker:
                saved_by_ticker[ticker] = trade_data

        # Get positions from both accounts
        pos1 = await self.account1.list_positions()
        pos2 = await self.account2.list_positions()

        if not pos1 and not pos2:
            logger.info("No existing positions found")
            clear_active_trades()  # Clean up stale saved trades
            return

        # Build a map of positions by ticker for each account
        acc1_positions = {p.get("ticker"): p for p in pos1 if p.get("ticker")}
        acc2_positions = {p.get("ticker"): p for p in pos2 if p.get("ticker")}

        # Find matching pairs (same ticker on both accounts with opposite sides)
        matched_tickers = set(acc1_positions.keys()) & set(acc2_positions.keys())

        for ticker in matched_tickers:
            p1 = acc1_positions[ticker]
            p2 = acc2_positions[ticker]

            # Verify they are opposite sides
            side1 = p1.get("side", "")
            side2 = p2.get("side", "")

            if side1 == side2:
                logger.warning(f"Both accounts have same side for {ticker}, skipping")
                continue

            # Determine which account is long
            account1_is_long = side1 == "LONG"

            # Get entry price (use average of both)
            entry1 = float(p1.get("entry_price", 0))
            entry2 = float(p2.get("entry_price", 0))
            entry_price = (entry1 + entry2) / 2 if entry1 and entry2 else entry1 or entry2

            # Get size (use smaller to be safe)
            size1 = abs(float(p1.get("size", 0)))
            size2 = abs(float(p2.get("size", 0)))
            size = min(size1, size2)

            # Check if we have saved data for this trade
            saved = saved_by_ticker.get(ticker, {})
            
            # Use saved opened_at if available, otherwise use now
            if saved.get("opened_at"):
                try:
                    opened_at = datetime.fromisoformat(saved["opened_at"])
                except:
                    opened_at = datetime.now()
            else:
                opened_at = datetime.now()
            
            # Use saved target_hold_minutes if available
            target_hold = saved.get("target_hold_minutes", 0)
            if target_hold <= 0:
                min_hold = self.settings.min_hold_time_minutes
                max_hold = self.settings.max_hold_time_minutes
                target_hold = random.uniform(min_hold, max_hold)

            # Create trade pair
            trade = TradePair(
                id=saved.get("id", f"resumed_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                product_id=ticker,
                size=size,
                leverage=self.settings.market_leverage_limits.get(ticker.upper(), 10),
                entry_price=entry_price,
                status=TradeStatus.OPEN,
                opened_at=opened_at,
                account1_is_long=account1_is_long,
                long_entry_price=entry1 if account1_is_long else entry2,
                short_entry_price=entry2 if account1_is_long else entry1,
                target_hold_minutes=target_hold,
            )

            self.active_trades[trade.id] = trade
            
            # Calculate time already held
            held_minutes = trade.hold_time_minutes
            remaining = max(0, target_hold - held_minutes)
            logger.info(f"Resumed trade: {ticker} | Entry: ${entry_price:.2f} | Held: {held_minutes:.1f}m | Remaining: {remaining:.1f}m")

        if self.active_trades:
            logger.info(f"Resumed {len(self.active_trades)} existing trade(s)")
            # Save the updated trades
            save_active_trades(self.active_trades)
        else:
            logger.info("No matching trade pairs found to resume")
            clear_active_trades()

    async def shutdown(self) -> None:
        """Shutdown the strategy."""
        logger.info("Shutting down strategy...")
        self.running = False

        if self.account1:
            await self.account1.disconnect()
        if self.account2:
            await self.account2.disconnect()

    async def start(self) -> None:
        """Start the farming strategy."""
        if self.running:
            logger.warning("Strategy already running")
            return

        logger.info("Starting Points Farming Strategy...")
        self.running = True
        self.stats = StrategyStats()

        try:
            while self.running:
                # Check if it's a new day - reset daily counter
                today = datetime.now().strftime("%Y-%m-%d")
                if today != self.stats.daily_trades_date:
                    logger.info(f"New day detected. Resetting daily trade counter (was {self.stats.daily_trades})")
                    self.stats.daily_trades = 0
                    self.stats.daily_trades_date = today

                # Get settings
                max_daily = getattr(self.settings, 'max_daily_trades', 100)
                max_concurrent = getattr(self.settings, 'max_concurrent_trades', 2)

                # Open new trades if:
                # 1. Have room for more trades (below max_concurrent_trades)
                # 2. Haven't reached daily max trades limit
                # 3. Enough time has passed since last trade
                current_trade_count = len(self.active_trades)
                trades_needed = max_concurrent - current_trade_count

                if trades_needed > 0 and self.stats.daily_trades < max_daily:
                    # Calculate how many more trades we can open (respecting daily limit)
                    remaining_daily = max_daily - self.stats.daily_trades
                    trades_to_open = min(trades_needed, remaining_daily)

                    if trades_to_open > 0:
                        await self._open_multiple_trades(trades_to_open)

                # Monitor active trades (check SL/TP/TIME conditions)
                if len(self.active_trades) > 0:
                    await self._monitor_trades()

                # Sleep before next check
                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Strategy cancelled")
        except Exception as e:
            logger.error(f"Strategy error: {e}")
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the strategy."""
        logger.info("Stopping strategy...")
        self.running = False

    async def close_all(self) -> None:
        """Close all active trades."""
        logger.info("Closing all active trades...")
        if not self.active_trades:
            logger.info("No active trades to close")
            return

        # Close all trades in parallel
        close_tasks = [
            self._close_trade(trade_id, "MANUAL")
            for trade_id in list(self.active_trades.keys())
        ]
        await asyncio.gather(*close_tasks, return_exceptions=True)

    def _get_available_pairs(self) -> list[str]:
        """Get trading pairs that don't have active trades."""
        all_pairs = self.settings.trading_pairs_list
        active_pairs = {trade.product_id for trade in self.active_trades.values()}
        return [p for p in all_pairs if p not in active_pairs]

    async def _open_multiple_trades(self, count: int) -> None:
        """Open multiple trades with different pairs.

        Args:
            count: Number of trades to open (will be limited to available pairs)
        """
        if not self.account1 or not self.account2:
            return

        # Check delay since last trade
        if self.completed_trades:
            last_trade = self.completed_trades[-1]
            time_since = (datetime.now() - last_trade.opened_at).total_seconds()
            if time_since < self.stats.next_trade_delay:
                return

        # Get pairs not already in active trades
        available_pairs = self._get_available_pairs()
        if not available_pairs:
            logger.info("No available pairs (all configured pairs have active trades)")
            return

        # Limit to available pairs and randomize selection
        random.shuffle(available_pairs)
        pairs_to_trade = available_pairs[:count]

        logger.info(f"Opening {len(pairs_to_trade)} trade(s) with pairs: {pairs_to_trade}")

        # Open trades sequentially to properly handle position sizing
        for product_id in pairs_to_trade:
            await self._open_new_trade(product_id)

    async def _open_new_trade(self, product_id: str = None) -> None:
        """Open a new long/short trade pair.

        Args:
            product_id: Specific pair to trade. If None, randomly selects from available pairs.
        """
        if not self.account1 or not self.account2:
            return

        # Get account balances
        bal1 = await self.account1.get_balance()
        bal2 = await self.account2.get_balance()
        min_balance = min(bal1, bal2)

        # Check minimum balance threshold
        min_threshold = getattr(self.settings, 'min_balance_threshold', 10.0)
        if min_balance < min_threshold:
            logger.warning(f"Balance too low: ${min_balance:.2f} < ${min_threshold:.2f} threshold. Stopping.")
            self.running = False
            return

        # Select a pair if not specified
        if product_id is None:
            available_pairs = self._get_available_pairs()
            if not available_pairs:
                logger.warning("No available pairs for trading")
                return
            product_id = random.choice(available_pairs)

        # Verify this pair isn't already active
        active_pairs = {trade.product_id for trade in self.active_trades.values()}
        if product_id in active_pairs:
            logger.warning(f"Pair {product_id} already has an active trade, skipping")
            return

        # Randomly decide which account goes long vs short (anti-sybil)
        account1_goes_long = random.choice([True, False])
        long_account = self.account1 if account1_goes_long else self.account2
        short_account = self.account2 if account1_goes_long else self.account1

        direction_info = "Acc1=LONG, Acc2=SHORT" if account1_goes_long else "Acc1=SHORT, Acc2=LONG"
        logger.info(f"Opening trade pair for {product_id} ({direction_info})")

        # Get current price
        price = await long_account.get_mark_price(product_id)
        if price <= 0:
            logger.error(f"Could not get price for {product_id}")
            return

        # Get leverage for this market from user's per-pair settings
        api_max_lev = long_account.get_max_leverage(product_id)
        user_lev_for_market = self.settings.market_leverage_limits.get(product_id.upper(), api_max_lev)
        # Use exact leverage user set (capped at API max for safety)
        effective_leverage = min(user_lev_for_market, api_max_lev)

        logger.info(f"Leverage for {product_id}: {effective_leverage}x (user set: {user_lev_for_market}x, market max: {api_max_lev}x)")

        # Calculate position size with consideration for multiple concurrent trades
        max_concurrent = getattr(self.settings, 'max_concurrent_trades', 2)
        use_full = getattr(self.settings, 'use_full_balance', True)

        if use_full or self.settings.position_size <= 0:
            # Use portion of balance based on number of concurrent trades
            # 80% safety buffer, divided by max concurrent trades
            safety_buffer = 0.80 / max_concurrent
            usable_balance = min_balance * safety_buffer
            position_usd = usable_balance * effective_leverage
            logger.info(f"Balances: Acc1=${bal1:.2f}, Acc2=${bal2:.2f} (using min ${min_balance:.2f} x {safety_buffer:.1%} for {max_concurrent} concurrent trades)")
            logger.info(f"Position: ${usable_balance:.2f} x {effective_leverage}x = ${position_usd:.2f}")
        else:
            # Fixed position size - divide by concurrent trades
            position_usd = self.settings.position_size / max_concurrent

        # Calculate size and round to lot size
        lot_size = long_account.get_lot_size(product_id)
        raw_size = position_usd / price
        size = round(raw_size / lot_size) * lot_size
        if size <= 0:
            size = lot_size  # Minimum one lot

        logger.info(f"Price: ${price:.2f}, Size: {size:.6f} (lot: {lot_size})")

        # Use the current price as reference for TP/SL calculation
        entry_price = long_account.round_price_to_tick(price, product_id)

        # Create trade pair
        trade = TradePair(
            id=f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{product_id}",
            product_id=product_id,
            size=size,
            leverage=effective_leverage,
            entry_price=price,
        )

        # Store which account is long for closing later
        trade.account1_is_long = account1_goes_long

        # Open BOTH positions simultaneously with MARKET orders for guaranteed fill
        # MARKET orders are more reliable than IOC limit orders
        logger.info(f"Opening LONG and SHORT positions simultaneously (MARKET orders)...")
        long_task = long_account.open_position(product_id, "buy", size)  # MARKET order
        short_task = short_account.open_position(product_id, "sell", size)  # MARKET order

        # Execute both orders at the same time
        results = await asyncio.gather(long_task, short_task, return_exceptions=True)
        long_order, short_order = results

        # Check for exceptions
        if isinstance(long_order, Exception):
            logger.error(f"Long order exception: {long_order}")
            long_order = None
        if isinstance(short_order, Exception):
            logger.error(f"Short order exception: {short_order}")
            short_order = None

        # Handle failures
        if not long_order and not short_order:
            logger.error("Both orders failed")
            trade.status = TradeStatus.FAILED
            return

        if not long_order:
            logger.error("Failed to open long position - closing short")
            await short_account.close_position(product_id)
            trade.status = TradeStatus.FAILED
            return

        if not short_order:
            logger.error("Failed to open short position - closing long")
            await long_account.close_position(product_id)
            trade.status = TradeStatus.FAILED
            return

        trade.long_order_id = long_order
        trade.short_order_id = short_order
        trade.status = TradeStatus.OPEN
        trade.opened_at = datetime.now()

        # Store entry prices for TP calculation
        trade.long_entry_price = entry_price
        trade.short_entry_price = entry_price

        # Set random hold time between min and max
        min_hold = self.settings.min_hold_time_minutes
        max_hold = self.settings.max_hold_time_minutes
        trade.target_hold_minutes = random.uniform(min_hold, max_hold)

        self.active_trades[trade.id] = trade
        self.stats.total_trades += 1
        self.stats.daily_trades += 1

        # Calculate TP/SL prices with INVERSE relationship
        # LONG: TP at HIGH price (profit when rises), SL at LOW price (loss when drops)
        # Calculate TP target prices for monitoring (bot will close, not exchange)
        # TP is monitored by bot: if 5% move on EITHER side, close both
        tp_percent = self.settings.take_profit_percent
        
        high_price = long_account.round_price_to_tick(entry_price * (1 + tp_percent), product_id)
        low_price = long_account.round_price_to_tick(entry_price * (1 - tp_percent), product_id)

        # Store TP prices in trade for monitoring (no exchange-side orders)
        trade.long_entry_price = entry_price
        trade.short_entry_price = entry_price

        logger.info(f"TP Prices (bot-monitored): HIGH=${high_price:,.2f} / LOW=${low_price:,.2f} (+/-{tp_percent*100:.1f}%)")
        logger.info(f"  If price >= ${high_price:,.2f}: LONG TP hit -> close both")
        logger.info(f"  If price <= ${low_price:,.2f}: SHORT TP hit -> close both")
        logger.info(f"  Otherwise: Hold for {trade.target_hold_minutes:.1f}m then close")

        # NO exchange-side TP/SL orders - bot monitors and closes manually
        # This is more reliable than OCO orders which trigger unexpectedly

        # Set random delay for next trade
        min_delay = getattr(self.settings, 'min_trade_delay_seconds', 60)
        max_delay = getattr(self.settings, 'max_trade_delay_seconds', 300)
        self.stats.next_trade_delay = random.uniform(min_delay, max_delay)

        logger.info(f"Trade opened: {trade.id} | {product_id} @ ${price:.2f} | Hold: {trade.target_hold_minutes:.1f}m | TP: +/-{tp_percent*100:.1f}%")

        # Save active trades to file for persistence
        save_active_trades(self.active_trades)

        if self._on_trade_open:
            self._on_trade_open(trade)

    async def _monitor_trades(self) -> None:
        """Monitor active trades for TP (5% move) or Time conditions.
        
        Bot monitors price and closes both positions when:
        - Price rises +5% from entry (LONG TP)
        - Price falls -5% from entry (SHORT TP)
        - Random hold time reached (close at current price)
        
        No stop-loss - delta neutral means net exposure is ~0.
        """
        for trade_id, trade in list(self.active_trades.items()):
            if trade.status != TradeStatus.OPEN:
                continue

            # Get correct accounts based on which is long
            long_account = self.account1 if trade.account1_is_long else self.account2
            short_account = self.account2 if trade.account1_is_long else self.account1

            # Get current price
            current_price = await long_account.get_current_price(trade.product_id)
            if current_price <= 0:
                continue

            entry_price = trade.entry_price
            tp_percent = self.settings.take_profit_percent

            # Calculate price change from entry
            price_change_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            # Get current PnLs
            trade.pnl_long = await long_account.get_position_pnl(trade.product_id)
            trade.pnl_short = await short_account.get_position_pnl(trade.product_id)

            hold_time = trade.hold_time_minutes
            target_hold = trade.target_hold_minutes if trade.target_hold_minutes > 0 else self.settings.max_hold_time_minutes

            # Check TP conditions (5% move on either side)
            if price_change_pct >= tp_percent:
                # Price went UP 5% -> LONG is profitable
                logger.info(f"TP hit for {trade_id}: Price ${current_price:,.2f} (+{price_change_pct*100:.2f}%) - LONG profit")
                await self._close_trade(trade_id, "TP_LONG")
                continue

            if price_change_pct <= -tp_percent:
                # Price went DOWN 5% -> SHORT is profitable
                logger.info(f"TP hit for {trade_id}: Price ${current_price:,.2f} ({price_change_pct*100:.2f}%) - SHORT profit")
                await self._close_trade(trade_id, "TP_SHORT")
                continue

            # Check random target hold time
            if hold_time >= target_hold:
                logger.info(f"Hold target reached for {trade_id}: {hold_time:.1f}m / {target_hold:.1f}m")
                await self._close_trade(trade_id, "TIME")
                continue

            # Log status every 30 seconds (6 checks at 5s interval)
            if not hasattr(trade, '_log_counter'):
                trade._log_counter = 0
            trade._log_counter += 1
            if trade._log_counter % 6 == 0:  # Every 30 seconds
                logger.info(
                    f"Monitoring {trade.product_id}: ${current_price:,.2f} ({price_change_pct*100:+.2f}%) | "
                    f"PnL: L=${trade.pnl_long:.2f} S=${trade.pnl_short:.2f} | "
                    f"Hold: {hold_time:.1f}m / {target_hold:.1f}m"
                )

    async def _close_trade(self, trade_id: str, reason: str) -> None:
        """Close a trade pair (both long and short positions)."""
        trade = self.active_trades.get(trade_id)
        if not trade:
            return

        logger.info(f"Closing trade {trade_id} - Reason: {reason}")

        # Get correct accounts based on which is long
        long_account = self.account1 if trade.account1_is_long else self.account2
        short_account = self.account2 if trade.account1_is_long else self.account1

        # Get current price for limit orders
        current_price = await long_account.get_mark_price(trade.product_id)
        if current_price <= 0:
            logger.warning(f"Could not get price for {trade.product_id}, falling back to market orders")
            sell_limit_price = None
            buy_limit_price = None
        else:
            # Use the SAME price for both close orders to ensure delta-neutral exit
            exit_price = long_account.round_price_to_tick(current_price, trade.product_id)
            sell_limit_price = exit_price
            buy_limit_price = exit_price
            logger.info(f"Close prices: SELL @ ${sell_limit_price:.2f}, BUY @ ${buy_limit_price:.2f} (same price)")

        # Close BOTH positions simultaneously with LIMIT orders
        logger.info(f"Closing LONG and SHORT positions simultaneously for {trade.product_id}...")
        # Long position closes with SELL, Short position closes with BUY
        close_long_task = long_account.close_position(trade.product_id, limit_price=sell_limit_price)
        close_short_task = short_account.close_position(trade.product_id, limit_price=buy_limit_price)

        # Execute both closes at the same time
        results = await asyncio.gather(close_long_task, close_short_task, return_exceptions=True)
        closed1, closed2 = results

        # Check for exceptions
        if isinstance(closed1, Exception):
            logger.error(f"Close LONG exception: {closed1}")
            closed1 = False
        if isinstance(closed2, Exception):
            logger.error(f"Close SHORT exception: {closed2}")
            closed2 = False

        # If limit orders failed, retry with market orders (fallback)
        if not closed1 or not closed2:
            logger.warning("Limit close order(s) failed, retrying with market orders...")

            retry_tasks = []
            if not closed1:
                retry_tasks.append(("long", long_account.close_position(trade.product_id)))
            if not closed2:
                retry_tasks.append(("short", short_account.close_position(trade.product_id)))

            if retry_tasks:
                retry_results = await asyncio.gather(*[t[1] for t in retry_tasks], return_exceptions=True)
                for i, (name, _) in enumerate(retry_tasks):
                    result = retry_results[i]
                    if isinstance(result, Exception):
                        logger.error(f"Close {name} market order exception: {result}")
                    elif result:
                        if name == "long":
                            closed1 = True
                        else:
                            closed2 = True

        if not closed1:
            logger.error(f"Failed to close LONG position for {trade_id}")
        if not closed2:
            logger.error(f"Failed to close SHORT position for {trade_id}")

        if not closed1 or not closed2:
            logger.error(f"Trade {trade_id} may not be fully closed! Check positions manually.")

        # Update trade
        trade.closed_at = datetime.now()
        trade.close_reason = reason

        if reason == "TP":
            trade.status = TradeStatus.CLOSED_TP
            self.stats.successful_trades += 1
        elif reason == "SL":
            trade.status = TradeStatus.CLOSED_SL
        elif reason == "TIME":
            trade.status = TradeStatus.CLOSED_TIME
        else:
            trade.status = TradeStatus.CLOSED_MANUAL

        self.stats.total_pnl += trade.total_pnl

        # Move to completed
        del self.active_trades[trade_id]
        self.completed_trades.append(trade)

        # Update saved trades (remove closed trade)
        save_active_trades(self.active_trades)

        logger.info(
            f"Trade closed: {trade_id} | PnL: ${trade.total_pnl:.2f} | "
            f"Hold: {trade.hold_time_minutes:.1f}m"
        )

        if self._on_trade_close:
            self._on_trade_close(trade)

    async def get_status(self) -> dict:
        """Get strategy status."""
        bal1 = await self.account1.get_balance() if self.account1 else 0
        bal2 = await self.account2.get_balance() if self.account2 else 0

        return {
            "running": self.running,
            "account1_balance": bal1,
            "account2_balance": bal2,
            "active_trades": len(self.active_trades),
            "completed_trades": len(self.completed_trades),
            "total_trades": self.stats.total_trades,
            "max_daily_trades": self.settings.max_daily_trades,
            "total_pnl": self.stats.total_pnl,
            "settings": {
                "pairs": self.settings.trading_pairs_list,
                "leverage": self.settings.leverage,
                "position_size": self.settings.position_size,
                "stop_loss": f"{self.settings.stop_loss_percent*100:.1f}%",
                "take_profit": f"{self.settings.take_profit_percent*100:.1f}%",
                "min_hold": f"{self.settings.min_hold_time_minutes}m",
                "max_hold": f"{self.settings.max_hold_time_minutes}m",
            }
        }

    def on_trade_open(self, callback: Callable) -> None:
        """Register callback for trade open events."""
        self._on_trade_open = callback

    def on_trade_close(self, callback: Callable) -> None:
        """Register callback for trade close events."""
        self._on_trade_close = callback
