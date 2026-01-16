"""Delta-neutral trading strategy for Ethereal DN Trader."""

import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Settings, get_settings
from ..client import EtherealClient, get_client
from ..market.data import MarketDataFetcher, FundingInfo
from ..market.websocket import WebSocketHandler
from ..utils.logger import get_logger
from ..utils.helpers import format_price

from .orders import OrderManager, OrderSide
from .positions import PositionManager, Position

logger = get_logger("ethereal.dn_strategy")


class StrategyState(Enum):
    """Strategy execution state."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


@dataclass
class DNPosition:
    """Delta-neutral position tracking."""
    product_id: str
    target_size: float
    actual_size: float
    entry_price: float
    funding_collected: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    last_rebalance: datetime = field(default_factory=datetime.now)

    @property
    def delta(self) -> float:
        """Current delta (deviation from target)."""
        return self.actual_size - self.target_size


@dataclass
class StrategyStats:
    """Strategy performance statistics."""
    total_funding_collected: float = 0.0
    total_trades: int = 0
    total_rebalances: int = 0
    total_pnl: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)


class DNStrategy:
    """
    Delta-neutral trading strategy.

    Monitors funding rates across products and opens positions
    to collect funding while maintaining delta neutrality.
    """

    def __init__(
        self,
        client: Optional[EtherealClient] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the DN strategy.

        Args:
            client: Ethereal client instance
            settings: Strategy settings
        """
        self.client = client or get_client()
        self.settings = settings or get_settings()

        # Components
        self.market_data = MarketDataFetcher(self.client)
        self.order_manager = OrderManager(self.client, self.market_data)
        self.position_manager = PositionManager(self.client, self.market_data)
        self.ws_handler: Optional[WebSocketHandler] = None

        # State
        self.state = StrategyState.IDLE
        self.dn_positions: dict[str, DNPosition] = {}
        self.stats = StrategyStats()

        # Configuration
        self.target_products: list[str] = ["BTCUSD", "ETHUSD"]
        self.max_position_size = self.settings.max_position_size
        self.delta_threshold = self.settings.delta_threshold
        self.min_funding_rate = self.settings.min_funding_rate
        self.take_profit = self.settings.take_profit_percent
        self.stop_loss = self.settings.stop_loss_percent

        # Callbacks
        self._on_trade_callback: Optional[Callable] = None
        self._on_rebalance_callback: Optional[Callable] = None

    async def initialize(self) -> None:
        """Initialize strategy components."""
        logger.info("Initializing DN Strategy...")

        # Connect to API
        await self.client.connect()

        # Load market data
        await self.market_data.get_all_products()

        # Initialize WebSocket
        self.ws_handler = WebSocketHandler()

        logger.info("DN Strategy initialized")

    async def shutdown(self) -> None:
        """Shutdown strategy and cleanup."""
        logger.info("Shutting down DN Strategy...")

        self.state = StrategyState.STOPPED

        if self.ws_handler and self.ws_handler.is_connected:
            await self.ws_handler.disconnect()

        await self.client.disconnect()
        logger.info("DN Strategy shutdown complete")

    async def start(self) -> None:
        """Start the strategy execution loop."""
        if self.state == StrategyState.RUNNING:
            logger.warning("Strategy already running")
            return

        logger.info("Starting DN Strategy...")
        self.state = StrategyState.RUNNING
        self.stats.start_time = datetime.now()

        try:
            # Connect WebSocket
            if self.ws_handler:
                await self.ws_handler.connect()

            # Main strategy loop
            while self.state == StrategyState.RUNNING:
                await self._strategy_tick()
                await asyncio.sleep(60)  # Check every minute

        except asyncio.CancelledError:
            logger.info("Strategy cancelled")
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            raise
        finally:
            self.state = StrategyState.STOPPED

    async def stop(self) -> None:
        """Stop the strategy execution."""
        logger.info("Stopping DN Strategy...")
        self.state = StrategyState.STOPPED

    async def pause(self) -> None:
        """Pause the strategy (keep positions open)."""
        logger.info("Pausing DN Strategy...")
        self.state = StrategyState.PAUSED

    async def resume(self) -> None:
        """Resume a paused strategy."""
        if self.state != StrategyState.PAUSED:
            logger.warning("Strategy not paused")
            return

        logger.info("Resuming DN Strategy...")
        self.state = StrategyState.RUNNING

    async def _strategy_tick(self) -> None:
        """Execute one iteration of the strategy loop."""
        logger.debug("Strategy tick...")

        try:
            # Update positions
            await self.position_manager.refresh_positions()

            # Check funding rates
            funding_rates = await self._get_funding_opportunities()

            # Process each target product
            for product_id in self.target_products:
                funding = funding_rates.get(product_id)
                if funding is None:
                    continue

                await self._process_product(product_id, funding)

            # Check and rebalance delta
            await self._rebalance_if_needed()

            # Update stats
            self.stats.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Strategy tick error: {e}")

    async def _get_funding_opportunities(self) -> dict[str, FundingInfo]:
        """Get funding rates for target products."""
        rates = await self.market_data.get_all_funding_rates(self.target_products)

        # Sort by absolute funding rate
        sorted_rates = dict(
            sorted(
                rates.items(),
                key=lambda x: abs(x[1].funding_rate),
                reverse=True,
            )
        )

        return sorted_rates

    async def _process_product(
        self,
        product_id: str,
        funding: FundingInfo,
    ) -> None:
        """Process a single product for DN opportunity."""
        position = await self.position_manager.get_position(product_id)
        funding_rate = funding.funding_rate

        logger.debug(f"{product_id} funding rate: {funding_rate*100:.4f}%")

        # Check if funding rate is attractive
        if abs(funding_rate) < self.min_funding_rate:
            # Close position if funding is no longer attractive
            if product_id in self.dn_positions and position:
                logger.info(f"Closing {product_id} - funding rate too low")
                await self._close_dn_position(product_id)
            return

        # Open or maintain position based on funding direction
        if product_id not in self.dn_positions:
            # No existing DN position - consider opening
            await self._open_dn_position(product_id, funding_rate)
        else:
            # Check PnL thresholds
            await self._check_pnl_thresholds(product_id)

    async def _open_dn_position(
        self,
        product_id: str,
        funding_rate: float,
    ) -> None:
        """Open a new DN position."""
        # Determine side based on funding rate
        # Positive funding = longs pay shorts (we short)
        # Negative funding = shorts pay longs (we long)
        side = OrderSide.SELL if funding_rate > 0 else OrderSide.BUY

        # Calculate position size
        price_info = await self.market_data.get_price(product_id)
        if price_info is None:
            logger.warning(f"Cannot get price for {product_id}")
            return

        current_price = price_info.mark_price
        size = self.max_position_size / current_price

        # Get product specs for size limits
        product = await self.market_data.get_product(product_id)
        if product:
            size = min(size, product.max_position_notional / current_price)

        logger.info(
            f"Opening DN position: {side.value} {size:.4f} {product_id} "
            f"(funding: {funding_rate*100:.4f}%)"
        )

        try:
            order = await self.order_manager.place_market_order(
                product_id=product_id,
                side=side,
                size=size,
            )

            if order.is_filled or order.filled_size > 0:
                target_size = size if side == OrderSide.BUY else -size
                self.dn_positions[product_id] = DNPosition(
                    product_id=product_id,
                    target_size=target_size,
                    actual_size=order.filled_size if side == OrderSide.BUY else -order.filled_size,
                    entry_price=order.average_price or current_price,
                )
                self.stats.total_trades += 1

                if self._on_trade_callback:
                    self._on_trade_callback(product_id, side, order.filled_size, order.average_price)

                logger.info(f"DN position opened: {product_id}")

        except Exception as e:
            logger.error(f"Failed to open DN position: {e}")

    async def _close_dn_position(self, product_id: str) -> None:
        """Close an existing DN position."""
        position = await self.position_manager.get_position(product_id)
        if position is None or position.size == 0:
            if product_id in self.dn_positions:
                del self.dn_positions[product_id]
            return

        # Close the position
        side = OrderSide.SELL if position.size > 0 else OrderSide.BUY
        size = abs(position.size)

        logger.info(f"Closing DN position: {side.value} {size:.4f} {product_id}")

        try:
            order = await self.order_manager.place_market_order(
                product_id=product_id,
                side=side,
                size=size,
                reduce_only=True,
            )

            if order.is_filled or order.filled_size > 0:
                # Update stats with realized PnL
                dn_pos = self.dn_positions.get(product_id)
                if dn_pos:
                    pnl = position.unrealized_pnl + dn_pos.funding_collected
                    self.stats.total_pnl += pnl
                    self.stats.total_funding_collected += dn_pos.funding_collected
                    del self.dn_positions[product_id]

                self.stats.total_trades += 1
                logger.info(f"DN position closed: {product_id}")

        except Exception as e:
            logger.error(f"Failed to close DN position: {e}")

    async def _check_pnl_thresholds(self, product_id: str) -> None:
        """Check if position hit take profit or stop loss."""
        position = await self.position_manager.get_position(product_id)
        if position is None:
            return

        dn_pos = self.dn_positions.get(product_id)
        if dn_pos is None:
            return

        # Calculate total PnL including funding
        total_pnl = position.unrealized_pnl + dn_pos.funding_collected
        pnl_percent = total_pnl / (abs(position.size) * position.entry_price)

        # Take profit
        if pnl_percent >= self.take_profit:
            logger.info(f"Take profit triggered for {product_id}: {pnl_percent*100:.2f}%")
            await self._close_dn_position(product_id)
            return

        # Stop loss
        if pnl_percent <= -self.stop_loss:
            logger.warning(f"Stop loss triggered for {product_id}: {pnl_percent*100:.2f}%")
            await self._close_dn_position(product_id)

    async def _rebalance_if_needed(self) -> None:
        """Check and rebalance portfolio delta."""
        total_delta = await self.position_manager.calculate_total_delta()
        account = await self.position_manager.get_account_summary()

        if account.margin_balance == 0:
            return

        # Delta as percentage of margin
        delta_percent = abs(total_delta) / account.margin_balance

        if delta_percent > self.delta_threshold:
            logger.info(f"Delta drift detected: {delta_percent*100:.2f}% - rebalancing")
            await self._rebalance_delta()

    async def _rebalance_delta(self) -> None:
        """Rebalance portfolio to reduce delta."""
        positions = await self.position_manager.get_all_positions()

        if not positions:
            return

        # Find the largest position to trim
        largest_pos = max(positions.values(), key=lambda p: abs(p.notional_value))

        # Reduce by a portion to bring delta closer to neutral
        reduction_size = abs(largest_pos.size) * 0.1  # Reduce by 10%

        side = OrderSide.SELL if largest_pos.size > 0 else OrderSide.BUY

        logger.info(
            f"Rebalancing: {side.value} {reduction_size:.4f} {largest_pos.product_id}"
        )

        try:
            await self.order_manager.place_market_order(
                product_id=largest_pos.product_id,
                side=side,
                size=reduction_size,
                reduce_only=True,
            )
            self.stats.total_rebalances += 1
            self.stats.total_trades += 1

            if self._on_rebalance_callback:
                self._on_rebalance_callback(largest_pos.product_id, reduction_size)

        except Exception as e:
            logger.error(f"Rebalance failed: {e}")

    async def close_all_positions(self) -> None:
        """Close all DN positions."""
        logger.info("Closing all DN positions...")

        for product_id in list(self.dn_positions.keys()):
            await self._close_dn_position(product_id)

        # Cancel any open orders
        await self.order_manager.cancel_all_orders()

    async def get_status(self) -> dict:
        """Get current strategy status."""
        positions = await self.position_manager.get_all_positions()
        delta = await self.position_manager.calculate_total_delta()
        account = await self.position_manager.get_account_summary()

        return {
            "state": self.state.value,
            "positions": len(positions),
            "total_delta": delta,
            "margin_balance": account.margin_balance,
            "available_balance": account.available_balance,
            "unrealized_pnl": account.unrealized_pnl,
            "stats": {
                "total_trades": self.stats.total_trades,
                "total_rebalances": self.stats.total_rebalances,
                "total_funding_collected": self.stats.total_funding_collected,
                "total_pnl": self.stats.total_pnl,
                "running_since": self.stats.start_time.isoformat(),
            },
        }

    def on_trade(self, callback: Callable) -> None:
        """Register callback for trade events."""
        self._on_trade_callback = callback

    def on_rebalance(self, callback: Callable) -> None:
        """Register callback for rebalance events."""
        self._on_rebalance_callback = callback
