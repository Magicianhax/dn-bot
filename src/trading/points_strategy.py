"""Points farming strategy - Long/Short on two accounts."""

import asyncio
import random
from decimal import Decimal
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ethereal import AsyncRESTClient
from eth_account import Account

from ..config import Settings, get_settings
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
                if ticker and product_id:
                    self._products[ticker] = {
                        'id': product_id,
                        'ticker': ticker,
                        'onchain_id': onchain_id,
                        'lot_size': float(getattr(p, 'lot_size', 0.0001) or 0.0001),
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
                    result.append({
                        "ticker": getattr(p, 'ticker', '') or getattr(p, 'product_id', ''),
                        "size": size,
                        "entry_price": float(getattr(p, 'entry_price', 0) or getattr(p, 'avg_entry_price', 0) or 0),
                        "unrealized_pnl": float(getattr(p, 'unrealized_pnl', 0) or 0),
                    })
            return result
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            return []

    def get_product_uuid(self, ticker: str) -> Optional[str]:
        """Get product UUID from ticker."""
        product = self._products.get(ticker)
        return product['id'] if product else None

    def get_lot_size(self, ticker: str) -> float:
        """Get lot size for a product."""
        product = self._products.get(ticker)
        return product['lot_size'] if product else 0.0001

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

    async def open_position(self, ticker: str, side: str, size: float) -> Optional[str]:
        """Open a position. Returns order ID if successful."""
        if not self._subaccount_id:
            logger.error("No subaccount ID")
            return None

        try:
            # side: 0 = buy, 1 = sell
            side_int = 0 if side.lower() == "buy" else 1
            side_name = "BUY" if side_int == 0 else "SELL"
            logger.info(f"Opening {side_name} {size:.6f} {ticker}")

            # For linked signers, the sender MUST be the signer address (derived from private key)
            # NOT the EOA/wallet address! The backend verifies signature matches sender.
            subaccount_param = self._subaccount_name_raw or self._subaccount_name
            
            # Use signer address as sender (critical for linked signers!)
            sender_address = self._signer_address
            
            logger.info(f"Order: sender={sender_address[:10]}..., subaccount={subaccount_param[:20] if subaccount_param else 'None'}...")
            
            # Simple create_order with correct sender
            response = await self._client.create_order(
                order_type="MARKET",
                ticker=ticker,
                side=side_int,
                quantity=Decimal(str(size)),
                sender=sender_address,  # Use signer address, not EOA!
                subaccount=subaccount_param,
            )

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

    async def close_position(self, ticker: str) -> bool:
        """Close position for a product."""
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
            
            logger.info(f"Closing {ticker}: size={size:.6f}, side={side_name}")
            
            # Use create_order with reduce_only - this works better than prepare_order
            result = await self._client.create_order(
                order_type="MARKET",
                ticker=ticker,
                side=side_int,
                quantity=Decimal(str(abs(size))),
                reduce_only=True,
                sender=self._signer_address,
                subaccount=subaccount_param,
            )
            
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

    async def get_position_pnl(self, ticker: str) -> float:
        """Get unrealized PnL for a position."""
        if not self._subaccount_id:
            return 0.0

        try:
            positions = await self._client.list_positions(
                subaccount_id=self._subaccount_id,
                sender=self.wallet_address,
            )
            for pos in positions:
                pos_ticker = getattr(pos, 'ticker', None) or str(getattr(pos, 'product_id', ''))
                if pos_ticker == ticker:
                    pnl = getattr(pos, 'unrealized_pnl', None) or getattr(pos, 'pnl', None) or 0
                    return float(pnl)
            return 0.0
        except:
            return 0.0


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

        return True

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
                
                # Get max daily trades setting
                max_daily = getattr(self.settings, 'max_daily_trades', 100)
                
                # Only open new trade if:
                # 1. No active trades currently (wait for current to close)
                # 2. Haven't reached daily max trades limit
                # 3. Enough time has passed since last trade closed
                if (len(self.active_trades) == 0 and 
                    self.stats.daily_trades < max_daily):
                    await self._open_new_trade()

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
        for trade_id in list(self.active_trades.keys()):
            await self._close_trade(trade_id, "MANUAL")

    async def _open_new_trade(self) -> None:
        """Open a new long/short trade pair."""
        if not self.account1 or not self.account2:
            return

        # Double-check no active trades (prevent race conditions)
        if len(self.active_trades) > 0:
            return

        # Check delay since last trade (random delay between min and max)
        if self.completed_trades:
            last_trade = self.completed_trades[-1]
            time_since = (datetime.now() - last_trade.opened_at).total_seconds()
            if time_since < self.stats.next_trade_delay:
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

        # Select a random pair from configured pairs
        pairs = self.settings.trading_pairs_list
        if not pairs:
            logger.error("No trading pairs configured")
            return

        product_id = random.choice(pairs)
        
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

        # Calculate position size
        use_full = getattr(self.settings, 'use_full_balance', True)
        if use_full or self.settings.position_size <= 0:
            # Use 80% of smaller balance to leave room for margin requirements and fees
            safety_buffer = 0.80
            usable_balance = min_balance * safety_buffer
            position_usd = usable_balance * self.settings.leverage
            logger.info(f"Balances: Acc1=${bal1:.2f}, Acc2=${bal2:.2f} (using min ${min_balance:.2f} x {safety_buffer:.0%})")
            logger.info(f"Position: ${usable_balance:.2f} x {self.settings.leverage}x = ${position_usd:.2f}")
        else:
            position_usd = self.settings.position_size

        # Calculate size and round to lot size
        lot_size = long_account.get_lot_size(product_id)
        raw_size = position_usd / price
        size = round(raw_size / lot_size) * lot_size
        if size <= 0:
            size = lot_size  # Minimum one lot

        logger.info(f"Price: ${price:.2f}, Size: {size:.6f} (lot: {lot_size})")

        # Create trade pair
        trade = TradePair(
            id=f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{product_id}",
            product_id=product_id,
            size=size,
            leverage=self.settings.leverage,
            entry_price=price,
        )
        
        # Store which account is long for closing later
        trade.account1_is_long = account1_goes_long

        # Open long position
        long_order = await long_account.open_position(product_id, "buy", size)
        if not long_order:
            logger.error("Failed to open long position")
            trade.status = TradeStatus.FAILED
            return

        trade.long_order_id = long_order

        # Open short position
        short_order = await short_account.open_position(product_id, "sell", size)
        if not short_order:
            logger.error("Failed to open short position - closing long")
            await long_account.close_position(product_id)
            trade.status = TradeStatus.FAILED
            return

        trade.short_order_id = short_order
        trade.status = TradeStatus.OPEN
        trade.opened_at = datetime.now()
        
        # Set random hold time between min and max
        min_hold = self.settings.min_hold_time_minutes
        max_hold = self.settings.max_hold_time_minutes
        trade.target_hold_minutes = random.uniform(min_hold, max_hold)

        self.active_trades[trade.id] = trade
        self.stats.total_trades += 1
        self.stats.daily_trades += 1
        
        # Set random delay for next trade
        min_delay = getattr(self.settings, 'min_trade_delay_seconds', 60)
        max_delay = getattr(self.settings, 'max_trade_delay_seconds', 300)
        self.stats.next_trade_delay = random.uniform(min_delay, max_delay)

        logger.info(f"Trade opened: {trade.id} | {product_id} @ ${price:.2f} | Hold: {trade.target_hold_minutes:.1f}m | Next delay: {self.stats.next_trade_delay:.0f}s")

        if self._on_trade_open:
            self._on_trade_open(trade)

    async def _monitor_trades(self) -> None:
        """Monitor active trades for SL/TP/Time conditions."""
        for trade_id, trade in list(self.active_trades.items()):
            if trade.status != TradeStatus.OPEN:
                continue

            # Get correct accounts based on which is long
            long_account = self.account1 if trade.account1_is_long else self.account2
            short_account = self.account2 if trade.account1_is_long else self.account1

            # Get current PnLs
            trade.pnl_long = await long_account.get_position_pnl(trade.product_id)
            trade.pnl_short = await short_account.get_position_pnl(trade.product_id)

            # Calculate PnL percentage (relative to position size)
            position_value = trade.size * trade.entry_price
            total_pnl_percent = trade.total_pnl / position_value if position_value > 0 else 0

            # Check conditions
            hold_time = trade.hold_time_minutes

            # Check Take Profit
            if total_pnl_percent >= self.settings.take_profit_percent:
                logger.info(f"TP hit for {trade_id}: {total_pnl_percent*100:.2f}%")
                await self._close_trade(trade_id, "TP")
                continue

            # Check Stop Loss
            if total_pnl_percent <= -self.settings.stop_loss_percent:
                logger.info(f"SL hit for {trade_id}: {total_pnl_percent*100:.2f}%")
                await self._close_trade(trade_id, "SL")
                continue

            # Check random target hold time
            target_hold = trade.target_hold_minutes if trade.target_hold_minutes > 0 else self.settings.max_hold_time_minutes
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
                    f"Monitoring {trade.product_id}: PnL ${trade.total_pnl:.2f} ({total_pnl_percent*100:.2f}%) | "
                    f"Hold: {hold_time:.1f}m / {target_hold:.1f}m"
                )

    async def _close_trade(self, trade_id: str, reason: str) -> None:
        """Close a trade pair."""
        trade = self.active_trades.get(trade_id)
        if not trade:
            return

        logger.info(f"Closing trade {trade_id} - Reason: {reason}")

        # Get correct accounts based on which is long
        long_account = self.account1 if trade.account1_is_long else self.account2
        short_account = self.account2 if trade.account1_is_long else self.account1

        # Close both positions
        logger.info(f"Closing LONG position for {trade.product_id}...")
        closed1 = await long_account.close_position(trade.product_id)
        if not closed1:
            logger.error(f"Failed to close LONG position for {trade_id}")
        
        logger.info(f"Closing SHORT position for {trade.product_id}...")
        closed2 = await short_account.close_position(trade.product_id)
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
