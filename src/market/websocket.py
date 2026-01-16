"""WebSocket handler for real-time market data from Ethereal."""

import asyncio
import json
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import get_settings
from ..utils.logger import get_logger

logger = get_logger("ethereal.websocket")


class SubscriptionType(Enum):
    """WebSocket subscription types."""
    BOOK_DEPTH = "BOOK_DEPTH"
    MARKET_PRICE = "MARKET_PRICE"
    ORDER_FILL = "ORDER_FILL"
    TRADE_FILL = "TRADE_FILL"
    ORDER_UPDATE = "ORDER_UPDATE"
    SUBACCOUNT_LIQUIDATION = "SUBACCOUNT_LIQUIDATION"
    TOKEN_TRANSFER = "TOKEN_TRANSFER"


@dataclass
class BookDepthUpdate:
    """Order book depth update."""
    product_id: str
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketPriceUpdate:
    """Market price update."""
    product_id: str
    mark_price: float
    index_price: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderFillUpdate:
    """Order fill notification."""
    order_id: str
    product_id: str
    side: str
    size: float
    price: float
    fee: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderStatusUpdate:
    """Order status change notification."""
    order_id: str
    product_id: str
    status: str
    filled_size: float
    remaining_size: float
    timestamp: datetime = field(default_factory=datetime.now)


# Type alias for callbacks
Callback = Callable[[Any], None]


class WebSocketHandler:
    """
    Handles WebSocket connections for real-time Ethereal data.

    Uses Socket.IO protocol for communication with the Ethereal
    WebSocket gateway.
    """

    def __init__(self, ws_url: Optional[str] = None):
        """
        Initialize the WebSocket handler.

        Args:
            ws_url: WebSocket URL. Uses settings if not provided.
        """
        settings = get_settings()
        self.ws_url = ws_url or settings.ethereal_ws_url
        self.subaccount = settings.subaccount_name

        self._sio = None
        self._connected = False
        self._subscriptions: dict[str, set[str]] = {}  # type -> product_ids
        self._callbacks: dict[str, list[Callback]] = {}

        # Latest data
        self.prices: dict[str, MarketPriceUpdate] = {}
        self.orderbooks: dict[str, BookDepthUpdate] = {}

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to the WebSocket gateway."""
        if self._connected:
            logger.warning("WebSocket already connected")
            return

        try:
            import socketio
            self._sio = socketio.AsyncClient(
                reconnection=True,
                reconnection_attempts=5,
                reconnection_delay=1,
            )

            # Set up event handlers
            self._setup_handlers()

            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            await self._sio.connect(
                self.ws_url,
                transports=["websocket"],
            )
            self._connected = True
            logger.info("WebSocket connected")

        except ImportError:
            logger.error("python-socketio not installed. Install with: pip install python-socketio[asyncio_client]")
            raise
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket gateway."""
        if not self._connected or self._sio is None:
            return

        logger.info("Disconnecting WebSocket...")
        await self._sio.disconnect()
        self._connected = False
        self._subscriptions.clear()
        logger.info("WebSocket disconnected")

    def _setup_handlers(self) -> None:
        """Set up Socket.IO event handlers."""
        if self._sio is None:
            return

        @self._sio.event
        async def connect():
            logger.info("WebSocket connected event")
            self._connected = True

        @self._sio.event
        async def disconnect():
            logger.warning("WebSocket disconnected event")
            self._connected = False

        @self._sio.event
        async def connect_error(data):
            logger.error(f"WebSocket connection error: {data}")

        @self._sio.on("BOOK_DEPTH")
        async def on_book_depth(data):
            await self._handle_book_depth(data)

        @self._sio.on("MARKET_PRICE")
        async def on_market_price(data):
            await self._handle_market_price(data)

        @self._sio.on("ORDER_FILL")
        async def on_order_fill(data):
            await self._handle_order_fill(data)

        @self._sio.on("ORDER_UPDATE")
        async def on_order_update(data):
            await self._handle_order_update(data)

        @self._sio.on("SUBACCOUNT_LIQUIDATION")
        async def on_liquidation(data):
            await self._handle_liquidation(data)

        @self._sio.on("exception")
        async def on_exception(data):
            logger.error(f"WebSocket exception: {data}")

    async def _handle_book_depth(self, data: dict) -> None:
        """Handle order book depth updates."""
        try:
            product_id = data.get("product_id", "")
            update = BookDepthUpdate(
                product_id=product_id,
                bids=[(float(b[0]), float(b[1])) for b in data.get("bids", [])],
                asks=[(float(a[0]), float(a[1])) for a in data.get("asks", [])],
            )
            self.orderbooks[product_id] = update

            # Trigger callbacks
            for callback in self._callbacks.get("BOOK_DEPTH", []):
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Failed to handle book depth: {e}")

    async def _handle_market_price(self, data: dict) -> None:
        """Handle market price updates."""
        try:
            product_id = data.get("product_id", "")
            update = MarketPriceUpdate(
                product_id=product_id,
                mark_price=float(data.get("mark_price", 0)),
                index_price=float(data.get("index_price", 0)),
            )
            self.prices[product_id] = update

            for callback in self._callbacks.get("MARKET_PRICE", []):
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Failed to handle market price: {e}")

    async def _handle_order_fill(self, data: dict) -> None:
        """Handle order fill notifications."""
        try:
            update = OrderFillUpdate(
                order_id=data.get("order_id", ""),
                product_id=data.get("product_id", ""),
                side=data.get("side", ""),
                size=float(data.get("size", 0)),
                price=float(data.get("price", 0)),
                fee=float(data.get("fee", 0)),
            )
            logger.info(f"Order filled: {update.order_id} - {update.side} {update.size} @ {update.price}")

            for callback in self._callbacks.get("ORDER_FILL", []):
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Failed to handle order fill: {e}")

    async def _handle_order_update(self, data: dict) -> None:
        """Handle order status updates."""
        try:
            update = OrderStatusUpdate(
                order_id=data.get("order_id", ""),
                product_id=data.get("product_id", ""),
                status=data.get("status", ""),
                filled_size=float(data.get("filled_size", 0)),
                remaining_size=float(data.get("remaining_size", 0)),
            )
            logger.debug(f"Order update: {update.order_id} - {update.status}")

            for callback in self._callbacks.get("ORDER_UPDATE", []):
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Failed to handle order update: {e}")

    async def _handle_liquidation(self, data: dict) -> None:
        """Handle liquidation alerts."""
        logger.warning(f"LIQUIDATION ALERT: {data}")
        for callback in self._callbacks.get("SUBACCOUNT_LIQUIDATION", []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def subscribe(
        self,
        subscription_type: SubscriptionType,
        product_id: Optional[str] = None,
    ) -> None:
        """
        Subscribe to a data stream.

        Args:
            subscription_type: Type of subscription
            product_id: Product ID for product-specific subscriptions
        """
        if not self._connected or self._sio is None:
            raise RuntimeError("WebSocket not connected")

        sub_type = subscription_type.value

        message = {"type": sub_type}
        if product_id:
            message["product_id"] = product_id
        if sub_type in ["ORDER_FILL", "ORDER_UPDATE", "SUBACCOUNT_LIQUIDATION", "TOKEN_TRANSFER"]:
            message["subaccount"] = self.subaccount

        logger.debug(f"Subscribing to {sub_type} for {product_id or 'all'}")
        await self._sio.emit("subscribe", message)

        # Track subscription
        if sub_type not in self._subscriptions:
            self._subscriptions[sub_type] = set()
        if product_id:
            self._subscriptions[sub_type].add(product_id)

    async def unsubscribe(
        self,
        subscription_type: SubscriptionType,
        product_id: Optional[str] = None,
    ) -> None:
        """
        Unsubscribe from a data stream.

        Args:
            subscription_type: Type of subscription
            product_id: Product ID for product-specific subscriptions
        """
        if not self._connected or self._sio is None:
            return

        sub_type = subscription_type.value

        message = {"type": sub_type}
        if product_id:
            message["product_id"] = product_id

        logger.debug(f"Unsubscribing from {sub_type} for {product_id or 'all'}")
        await self._sio.emit("unsubscribe", message)

        # Update tracking
        if sub_type in self._subscriptions and product_id:
            self._subscriptions[sub_type].discard(product_id)

    def on(self, event_type: str, callback: Callback) -> None:
        """
        Register a callback for an event type.

        Args:
            event_type: Event type (e.g., "MARKET_PRICE", "ORDER_FILL")
            callback: Callback function to invoke
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def off(self, event_type: str, callback: Optional[Callback] = None) -> None:
        """
        Remove a callback for an event type.

        Args:
            event_type: Event type
            callback: Specific callback to remove. If None, removes all.
        """
        if event_type not in self._callbacks:
            return

        if callback is None:
            self._callbacks[event_type].clear()
        else:
            self._callbacks[event_type] = [
                cb for cb in self._callbacks[event_type] if cb != callback
            ]

    def get_price(self, product_id: str) -> Optional[float]:
        """Get latest cached price for a product."""
        update = self.prices.get(product_id)
        return update.mark_price if update else None

    def get_orderbook(self, product_id: str) -> Optional[BookDepthUpdate]:
        """Get latest cached orderbook for a product."""
        return self.orderbooks.get(product_id)

    async def wait_for_price(
        self,
        product_id: str,
        timeout: float = 10.0,
    ) -> Optional[float]:
        """
        Wait for a price update for a product.

        Args:
            product_id: Product ID
            timeout: Maximum time to wait in seconds

        Returns:
            Price or None if timeout
        """
        if product_id in self.prices:
            return self.prices[product_id].mark_price

        event = asyncio.Event()
        price_holder = {"price": None}

        def on_price(update: MarketPriceUpdate):
            if update.product_id == product_id:
                price_holder["price"] = update.mark_price
                event.set()

        self.on("MARKET_PRICE", on_price)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return price_holder["price"]
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for price: {product_id}")
            return None
        finally:
            self.off("MARKET_PRICE", on_price)
