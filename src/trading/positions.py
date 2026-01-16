"""Position tracking and management for Ethereal DN Trader."""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..client import EtherealClient, get_client
from ..market.data import MarketDataFetcher
from ..utils.logger import get_logger
from ..utils.helpers import calculate_pnl, calculate_pnl_percent, format_price

logger = get_logger("ethereal.positions")


@dataclass
class Position:
    """Position data structure."""
    product_id: str
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    margin_used: float
    leverage: float
    unrealized_pnl: float
    realized_pnl: float
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0

    @property
    def side(self) -> str:
        """Get position side as string."""
        if self.size > 0:
            return "LONG"
        elif self.size < 0:
            return "SHORT"
        return "FLAT"

    @property
    def notional_value(self) -> float:
        """Get notional value of position."""
        return abs(self.size * self.mark_price)

    @property
    def pnl_percent(self) -> float:
        """Get PnL as percentage."""
        return calculate_pnl_percent(self.entry_price, self.mark_price, self.is_long)

    @classmethod
    def from_api_response(cls, data: dict) -> "Position":
        """Create Position from API response."""
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        return cls(
            product_id=data.get("product_id") or data.get("ticker") or "",
            size=safe_float(data.get("size") or data.get("quantity")),
            entry_price=safe_float(data.get("entry_price") or data.get("avg_entry_price")),
            mark_price=safe_float(data.get("mark_price")),
            liquidation_price=safe_float(data.get("liquidation_price")),
            margin_used=safe_float(data.get("margin_used") or data.get("margin")),
            leverage=safe_float(data.get("leverage"), 1.0),
            unrealized_pnl=safe_float(data.get("unrealized_pnl") or data.get("pnl")),
            realized_pnl=safe_float(data.get("realized_pnl")),
        )


@dataclass
class AccountSummary:
    """Account summary data."""
    margin_balance: float
    available_balance: float
    unrealized_pnl: float
    total_position_value: float
    margin_usage_percent: float


class PositionManager:
    """
    Manages position tracking and monitoring.

    Provides methods to fetch, track, and analyze positions
    across all products.
    """

    def __init__(
        self,
        client: Optional[EtherealClient] = None,
        market_data: Optional[MarketDataFetcher] = None,
    ):
        """
        Initialize the position manager.

        Args:
            client: Ethereal client instance
            market_data: Market data fetcher instance
        """
        self.client = client or get_client()
        self.market_data = market_data or MarketDataFetcher(self.client)
        self._positions: dict[str, Position] = {}
        self._last_update: Optional[datetime] = None

    async def refresh_positions(self) -> dict[str, Position]:
        """
        Refresh positions from the API.

        Returns:
            Dictionary of product_id -> Position
        """
        logger.debug("Refreshing positions...")
        response = await self.client.get_positions()

        self._positions.clear()
        for data in response:
            position = Position.from_api_response(data)
            if position.size != 0:  # Only track non-zero positions
                self._positions[position.product_id] = position

        self._last_update = datetime.now()
        logger.debug(f"Found {len(self._positions)} open positions")
        return self._positions

    async def get_position(self, product_id: str) -> Optional[Position]:
        """
        Get position for a specific product.

        Args:
            product_id: Product identifier

        Returns:
            Position or None if no position
        """
        # Refresh if stale
        if self._last_update is None:
            await self.refresh_positions()

        return self._positions.get(product_id)

    async def get_all_positions(self) -> dict[str, Position]:
        """
        Get all open positions.

        Returns:
            Dictionary of product_id -> Position
        """
        await self.refresh_positions()
        return self._positions.copy()

    async def get_account_summary(self) -> AccountSummary:
        """
        Get account summary with position information.

        Returns:
            AccountSummary with balance and position data
        """
        balance = await self.client.get_balance()
        positions = await self.get_all_positions()

        total_position_value = sum(p.notional_value for p in positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions.values())

        margin_balance = float(balance.get("margin_balance", 0))
        available_balance = float(balance.get("available_balance", 0))

        margin_usage = 0.0
        if margin_balance > 0:
            margin_usage = (margin_balance - available_balance) / margin_balance

        return AccountSummary(
            margin_balance=margin_balance,
            available_balance=available_balance,
            unrealized_pnl=total_unrealized_pnl,
            total_position_value=total_position_value,
            margin_usage_percent=margin_usage,
        )

    async def calculate_total_delta(self) -> float:
        """
        Calculate total portfolio delta in USD.

        Positive delta = net long exposure
        Negative delta = net short exposure

        Returns:
            Total delta in USD
        """
        positions = await self.get_all_positions()

        total_delta = 0.0
        for position in positions.values():
            # Delta = size * mark_price (positive for long, negative for short)
            total_delta += position.size * position.mark_price

        return total_delta

    async def calculate_product_delta(self, product_id: str) -> float:
        """
        Calculate delta for a specific product.

        Args:
            product_id: Product identifier

        Returns:
            Delta in USD
        """
        position = await self.get_position(product_id)
        if position is None:
            return 0.0

        return position.size * position.mark_price

    async def get_total_pnl(self) -> dict[str, float]:
        """
        Get total PnL across all positions.

        Returns:
            Dictionary with unrealized and realized PnL
        """
        positions = await self.get_all_positions()

        unrealized = sum(p.unrealized_pnl for p in positions.values())
        realized = sum(p.realized_pnl for p in positions.values())

        return {
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "total_pnl": unrealized + realized,
        }

    async def check_liquidation_risk(
        self,
        threshold_percent: float = 0.1,
    ) -> list[Position]:
        """
        Check for positions at risk of liquidation.

        Args:
            threshold_percent: Warn if price is within this % of liquidation

        Returns:
            List of positions at risk
        """
        positions = await self.get_all_positions()
        at_risk = []

        for position in positions.values():
            if position.liquidation_price <= 0:
                continue

            # Calculate distance to liquidation
            if position.is_long:
                distance = (position.mark_price - position.liquidation_price) / position.mark_price
            else:
                distance = (position.liquidation_price - position.mark_price) / position.mark_price

            if distance < threshold_percent:
                logger.warning(
                    f"LIQUIDATION RISK: {position.product_id} - "
                    f"{distance*100:.1f}% from liquidation"
                )
                at_risk.append(position)

        return at_risk

    def format_positions_table(self) -> str:
        """
        Format positions as a text table.

        Returns:
            Formatted table string
        """
        if not self._positions:
            return "No open positions"

        lines = [
            "Product    | Side  | Size       | Entry      | Mark       | PnL",
            "-" * 70,
        ]

        for position in self._positions.values():
            pnl_str = f"{position.unrealized_pnl:+.2f}"
            lines.append(
                f"{position.product_id:<10} | {position.side:<5} | "
                f"{position.size:<10.4f} | {format_price(position.entry_price):<10} | "
                f"{format_price(position.mark_price):<10} | {pnl_str}"
            )

        return "\n".join(lines)

    @property
    def positions(self) -> dict[str, Position]:
        """Get cached positions."""
        return self._positions.copy()

    @property
    def has_positions(self) -> bool:
        """Check if there are any open positions."""
        return len(self._positions) > 0
