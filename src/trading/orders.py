"""Order management for Ethereal DN Trader."""

import asyncio
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..client import EtherealClient, get_client
from ..market.data import MarketDataFetcher, ProductInfo
from ..utils.logger import get_logger
from ..utils.helpers import round_to_lot_size, round_to_tick_size

logger = get_logger("ethereal.orders")


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status."""
    NEW = "NEW"
    PENDING = "PENDING"
    FILLED_PARTIAL = "FILLED_PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force options."""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    product_id: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float]
    status: OrderStatus
    filled_size: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def remaining_size(self) -> float:
        """Get remaining unfilled size."""
        return self.size - self.filled_size

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in [OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.FILLED_PARTIAL]

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @classmethod
    def from_api_response(cls, data: dict) -> "Order":
        """Create Order from API response."""
        return cls(
            order_id=data.get("order_id", ""),
            product_id=data.get("product_id", ""),
            side=OrderSide(data.get("side", "buy").lower()),
            order_type=OrderType(data.get("order_type", "market").lower()),
            size=float(data.get("size", 0)),
            price=float(data.get("price")) if data.get("price") else None,
            status=OrderStatus(data.get("status", "NEW")),
            filled_size=float(data.get("filled_size", 0)),
            average_price=float(data.get("average_price", 0)),
        )


class OrderManager:
    """
    Manages order placement, tracking, and cancellation.

    Provides high-level order operations with validation and
    automatic lot/tick size rounding.
    """

    def __init__(
        self,
        client: Optional[EtherealClient] = None,
        market_data: Optional[MarketDataFetcher] = None,
    ):
        """
        Initialize the order manager.

        Args:
            client: Ethereal client instance
            market_data: Market data fetcher instance
        """
        self.client = client or get_client()
        self.market_data = market_data or MarketDataFetcher(self.client)
        self._open_orders: dict[str, Order] = {}

    async def _validate_order(
        self,
        product_id: str,
        size: float,
        price: Optional[float] = None,
    ) -> ProductInfo:
        """
        Validate order parameters against product specifications.

        Args:
            product_id: Product identifier
            size: Order size
            price: Order price (for limit orders)

        Returns:
            Product info for the product

        Raises:
            ValueError: If validation fails
        """
        product = await self.market_data.get_product(product_id)
        if product is None:
            raise ValueError(f"Unknown product: {product_id}")

        if size <= 0:
            raise ValueError("Order size must be positive")

        if size < product.min_order_size:
            raise ValueError(f"Size {size} below minimum {product.min_order_size}")

        if price is not None and price <= 0:
            raise ValueError("Price must be positive")

        return product

    async def place_market_order(
        self,
        product_id: str,
        side: OrderSide,
        size: float,
        reduce_only: bool = False,
    ) -> Order:
        """
        Place a market order.

        Args:
            product_id: Product to trade
            side: Buy or sell
            size: Order size
            reduce_only: Only reduce existing position

        Returns:
            Order object with status
        """
        product = await self._validate_order(product_id, size)

        # Round size to lot size
        rounded_size = round_to_lot_size(size, product.lot_size)
        if rounded_size <= 0:
            raise ValueError(f"Rounded size is zero. Min lot: {product.lot_size}")

        logger.info(f"Placing market {side.value} order: {rounded_size} {product_id}")

        response = await self.client.place_market_order(
            product_id=product_id,
            side=side.value,
            size=rounded_size,
            reduce_only=reduce_only,
        )

        order = Order.from_api_response(response)
        if order.is_open:
            self._open_orders[order.order_id] = order

        logger.info(f"Order placed: {order.order_id} - {order.status.value}")
        return order

    async def place_limit_order(
        self,
        product_id: str,
        side: OrderSide,
        size: float,
        price: float,
        reduce_only: bool = False,
        time_in_force: TimeInForce = TimeInForce.GTC,
        post_only: bool = False,
    ) -> Order:
        """
        Place a limit order.

        Args:
            product_id: Product to trade
            side: Buy or sell
            size: Order size
            price: Limit price
            reduce_only: Only reduce existing position
            time_in_force: Order time in force
            post_only: Only add liquidity (maker only)

        Returns:
            Order object with status
        """
        product = await self._validate_order(product_id, size, price)

        # Round to lot and tick sizes
        rounded_size = round_to_lot_size(size, product.lot_size)
        rounded_price = round_to_tick_size(price, product.tick_size)

        if rounded_size <= 0:
            raise ValueError(f"Rounded size is zero. Min lot: {product.lot_size}")

        logger.info(
            f"Placing limit {side.value} order: {rounded_size} {product_id} @ {rounded_price}"
        )

        response = await self.client.place_limit_order(
            product_id=product_id,
            side=side.value,
            size=rounded_size,
            price=rounded_price,
            reduce_only=reduce_only,
            time_in_force=time_in_force.value,
            post_only=post_only,
        )

        order = Order.from_api_response(response)
        if order.is_open:
            self._open_orders[order.order_id] = order

        logger.info(f"Order placed: {order.order_id} - {order.status.value}")
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        try:
            logger.info(f"Cancelling order: {order_id}")
            await self.client.cancel_order(order_id)

            if order_id in self._open_orders:
                self._open_orders[order_id].status = OrderStatus.CANCELED
                del self._open_orders[order_id]

            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(
        self,
        product_id: Optional[str] = None,
    ) -> int:
        """
        Cancel all open orders.

        Args:
            product_id: Optional product filter

        Returns:
            Number of orders cancelled
        """
        try:
            logger.info(f"Cancelling all orders for {product_id or 'all products'}")
            await self.client.cancel_all_orders(product_id=product_id)

            cancelled = 0
            for order_id in list(self._open_orders.keys()):
                order = self._open_orders[order_id]
                if product_id is None or order.product_id == product_id:
                    order.status = OrderStatus.CANCELED
                    del self._open_orders[order_id]
                    cancelled += 1

            logger.info(f"Cancelled {cancelled} orders")
            return cancelled
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return 0

    async def get_open_orders(
        self,
        product_id: Optional[str] = None,
    ) -> list[Order]:
        """
        Get all open orders.

        Args:
            product_id: Optional product filter

        Returns:
            List of open orders
        """
        response = await self.client.get_open_orders(product_id=product_id)

        orders = []
        self._open_orders.clear()

        for data in response:
            order = Order.from_api_response(data)
            orders.append(order)
            if order.is_open:
                self._open_orders[order.order_id] = order

        return orders

    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None if not found
        """
        # Check local cache first
        if order_id in self._open_orders:
            return self._open_orders[order_id]

        # Refresh from API
        orders = await self.get_open_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None

    async def wait_for_fill(
        self,
        order_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Optional[Order]:
        """
        Wait for an order to be filled.

        Args:
            order_id: Order ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Filled order or None if timeout/cancelled
        """
        import time
        start = time.time()

        while time.time() - start < timeout:
            order = await self.get_order(order_id)
            if order is None:
                logger.warning(f"Order {order_id} not found")
                return None

            if order.is_filled:
                return order

            if order.status in [OrderStatus.CANCELED, OrderStatus.EXPIRED]:
                logger.warning(f"Order {order_id} {order.status.value}")
                return order

            await asyncio.sleep(poll_interval)

        logger.warning(f"Timeout waiting for order {order_id}")
        return await self.get_order(order_id)

    @property
    def open_orders(self) -> dict[str, Order]:
        """Get cached open orders."""
        return self._open_orders.copy()
