"""Ethereal SDK client wrapper."""

from typing import Optional, Any
from contextlib import asynccontextmanager

from ethereal import AsyncRESTClient
from eth_account import Account

from .config import Settings, get_settings
from .utils.logger import get_logger

logger = get_logger("ethereal.client")


class EtherealClient:
    """
    Wrapper around the Ethereal SDK AsyncRESTClient.

    Provides a simplified interface for trading operations and
    handles connection lifecycle management.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the Ethereal client.

        Args:
            settings: Optional settings instance. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self._client: Optional[AsyncRESTClient] = None
        self._connected = False
        self._address: Optional[str] = None
        self._subaccount_id: Optional[str] = None

    @property
    def client(self) -> AsyncRESTClient:
        """Get the underlying SDK client."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Call connect() first.")
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None

    async def connect(self) -> None:
        """Initialize and connect the SDK client."""
        if self._connected:
            logger.warning("Client already connected")
            return

        logger.info("Connecting to Ethereal API...")
        try:
            # Use configured wallet address, or derive from private key if not set
            if self.settings.ethereal_wallet_address:
                self._address = self.settings.ethereal_wallet_address
            else:
                self._address = Account.from_key(self.settings.ethereal_private_key).address
                logger.warning("No ETHEREAL_WALLET_ADDRESS set, using address derived from private key")

            config = {
                "private_key": self.settings.ethereal_private_key,
                "base_url": self.settings.ethereal_api_url,
            }
            self._client = await AsyncRESTClient.create(config)
            self._connected = True
            logger.info(f"Connected to {self.settings.ethereal_api_url}")

            # Try to get default subaccount
            await self._load_subaccount()
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def _load_subaccount(self) -> None:
        """Load the default subaccount ID."""
        try:
            subaccounts = await self.client.list_subaccounts(sender=self._address)
            if subaccounts:
                # Find subaccount by name or use first one
                target_name = self.settings.subaccount_name
                for sub in subaccounts:
                    if hasattr(sub, 'name') and sub.name == target_name:
                        self._subaccount_id = str(sub.id)
                        logger.debug(f"Found subaccount '{target_name}': {self._subaccount_id}")
                        return
                # Use first subaccount if name not found
                self._subaccount_id = str(subaccounts[0].id)
                logger.debug(f"Using first subaccount: {self._subaccount_id}")
            else:
                logger.warning("No subaccounts found. Deposit funds to create one.")
        except Exception as e:
            logger.warning(f"Failed to load subaccount: {e}")

    async def disconnect(self) -> None:
        """Disconnect and cleanup the client."""
        if not self._connected:
            return

        logger.info("Disconnecting from Ethereal API...")
        if self._client:
            await self._client.close()
        self._client = None
        self._connected = False
        logger.info("Disconnected")

    @asynccontextmanager
    async def session(self):
        """Context manager for client session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    # Product Methods
    async def list_products(self) -> list[Any]:
        """
        Get list of available trading products.

        Returns:
            List of product objects/dictionaries with specifications
        """
        logger.debug("Fetching products...")
        response = await self.client.list_products()
        return response or []

    async def get_product(self, product_id: str) -> Optional[dict[str, Any]]:
        """
        Get specifications for a specific product.

        Args:
            product_id: Product identifier (e.g., "BTCUSD")

        Returns:
            Product specifications or None if not found
        """
        # Try to get from cached products
        if hasattr(self.client, 'products_by_ticker'):
            product = self.client.products_by_ticker.get(product_id)
            if product:
                return vars(product) if hasattr(product, '__dict__') else product

        products = await self.list_products()
        for product in products:
            ticker = product.get("ticker") or product.get("symbol") or product.get("product_id", "")
            if ticker == product_id:
                return product
        return None

    # Token Methods
    async def list_tokens(self) -> list[dict[str, Any]]:
        """
        Get list of supported tokens.

        Returns:
            List of token dictionaries
        """
        logger.debug("Fetching tokens...")
        response = await self.client.list_tokens()
        if response and hasattr(response[0], '__dict__'):
            return [vars(t) if hasattr(t, '__dict__') else t for t in response]
        return response

    # Account Methods
    async def get_subaccount(self, subaccount_name: Optional[str] = None) -> dict[str, Any]:
        """
        Get subaccount information.

        Args:
            subaccount_name: Subaccount name. Uses default from settings if not provided.

        Returns:
            Subaccount information including balances
        """
        name = subaccount_name or self.settings.subaccount_name
        logger.debug(f"Fetching subaccount: {name}")
        response = await self.client.get_subaccount(name)
        if hasattr(response, '__dict__'):
            return vars(response)
        return response

    async def get_balance(self, subaccount_name: Optional[str] = None) -> dict[str, Any]:
        """
        Get account balance.

        Args:
            subaccount_name: Subaccount name. Uses default from settings if not provided.

        Returns:
            Balance information
        """
        if not self._subaccount_id:
            return {
                "margin_balance": 0,
                "available_balance": 0,
                "unrealized_pnl": 0,
            }

        margin_balance = 0
        available_balance = 0
        unrealized_pnl = 0

        try:
            balances = await self.client.get_subaccount_balances(subaccount_id=self._subaccount_id)
            if balances:
                for b in balances:
                    # Handle pydantic model or dict
                    if hasattr(b, 'amount'):
                        margin_balance += float(b.amount or 0)
                        available_balance += float(b.available or 0)
                    else:
                        bal = vars(b) if hasattr(b, '__dict__') else b
                        margin_balance += float(bal.get("amount", bal.get("balance", 0)))
                        available_balance += float(bal.get("available", 0))
        except Exception as e:
            logger.warning(f"Failed to get balances: {e}")

        # Get unrealized PnL from positions
        try:
            positions = await self.get_positions(subaccount_name)
            for pos in positions:
                unrealized_pnl += float(pos.get("unrealized_pnl", 0))
        except:
            pass

        return {
            "margin_balance": margin_balance,
            "available_balance": available_balance if available_balance > 0 else margin_balance - abs(unrealized_pnl),
            "unrealized_pnl": unrealized_pnl,
        }

    # Order Methods
    async def place_market_order(
        self,
        product_id: str,
        side: str,
        size: float,
        subaccount_name: Optional[str] = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """
        Place a market order.

        Args:
            product_id: Product to trade (e.g., "BTCUSD")
            side: Order side ("buy" or "sell")
            size: Order size
            subaccount_name: Subaccount name
            reduce_only: Whether order should only reduce position

        Returns:
            Order response with status and details
        """
        name = subaccount_name or self.settings.subaccount_name
        logger.info(f"Placing market {side} order: {size} {product_id}")

        response = await self.client.create_order(
            subaccount_name=name,
            product_id=product_id,
            side=side.lower(),
            size=str(size),
            order_type="market",
            reduce_only=reduce_only,
        )

        result = vars(response) if hasattr(response, '__dict__') else response
        logger.info(f"Order placed: {result.get('order_id', result.get('id'))}")
        return result

    async def place_limit_order(
        self,
        product_id: str,
        side: str,
        size: float,
        price: float,
        subaccount_name: Optional[str] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        post_only: bool = False,
    ) -> dict[str, Any]:
        """
        Place a limit order.

        Args:
            product_id: Product to trade (e.g., "BTCUSD")
            side: Order side ("buy" or "sell")
            size: Order size
            price: Limit price
            subaccount_name: Subaccount name
            reduce_only: Whether order should only reduce position
            time_in_force: Time in force (GTC, IOC, FOK)
            post_only: Whether order should only add liquidity

        Returns:
            Order response with status and details
        """
        name = subaccount_name or self.settings.subaccount_name
        logger.info(f"Placing limit {side} order: {size} {product_id} @ {price}")

        response = await self.client.create_order(
            subaccount_name=name,
            product_id=product_id,
            side=side.lower(),
            size=str(size),
            price=str(price),
            order_type="limit",
            reduce_only=reduce_only,
            time_in_force=time_in_force,
            post_only=post_only,
        )

        result = vars(response) if hasattr(response, '__dict__') else response
        logger.info(f"Order placed: {result.get('order_id', result.get('id'))}")
        return result

    async def cancel_order(
        self,
        order_id: str,
        subaccount_name: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel
            subaccount_name: Subaccount name
            product_id: Product ID

        Returns:
            Cancellation response
        """
        name = subaccount_name or self.settings.subaccount_name
        logger.info(f"Cancelling order: {order_id}")

        response = await self.client.cancel_order(
            subaccount_name=name,
            order_id=order_id,
            product_id=product_id,
        )

        result = vars(response) if hasattr(response, '__dict__') else response
        logger.info(f"Order cancelled: {order_id}")
        return result

    async def cancel_all_orders(
        self,
        product_id: Optional[str] = None,
        subaccount_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Cancel all open orders.

        Args:
            product_id: Optional product filter
            subaccount_name: Subaccount name

        Returns:
            Cancellation response
        """
        name = subaccount_name or self.settings.subaccount_name
        logger.info(f"Cancelling all orders for {product_id or 'all products'}")

        response = await self.client.cancel_all_orders(
            subaccount_name=name,
            product_id=product_id,
        )

        logger.info("All orders cancelled")
        return vars(response) if hasattr(response, '__dict__') else response

    async def get_open_orders(
        self,
        product_id: Optional[str] = None,
        subaccount_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get all open orders.

        Args:
            product_id: Optional product filter
            subaccount_name: Subaccount name

        Returns:
            List of open orders
        """
        name = subaccount_name or self.settings.subaccount_name
        response = await self.client.list_orders(
            subaccount_name=name,
            product_id=product_id,
            status="open",
        )

        if response and hasattr(response[0], '__dict__'):
            return [vars(o) if hasattr(o, '__dict__') else o for o in response]
        return response or []

    # Position Methods
    async def get_positions(
        self,
        subaccount_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get all open positions.

        Args:
            subaccount_name: Subaccount name

        Returns:
            List of open positions
        """
        if not self._subaccount_id:
            return []

        response = await self.client.list_positions(subaccount_id=self._subaccount_id)

        if response and len(response) > 0 and hasattr(response[0], '__dict__'):
            return [vars(p) if hasattr(p, '__dict__') else p for p in response]
        return response or []

    async def get_position(
        self,
        product_id: str,
        subaccount_name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Get position for a specific product.

        Args:
            product_id: Product identifier
            subaccount_name: Subaccount name

        Returns:
            Position information or None if no position
        """
        if not self._subaccount_id:
            return None

        try:
            response = await self.client.get_position(
                subaccount_id=self._subaccount_id,
                product_id=product_id,
            )
            if response:
                return vars(response) if hasattr(response, '__dict__') else response
        except:
            pass
        return None

    # Market Data Methods
    async def get_orderbook(
        self,
        product_id: str,
        depth: int = 10,
    ) -> dict[str, Any]:
        """
        Get order book for a product.

        Args:
            product_id: Product identifier
            depth: Number of levels to fetch

        Returns:
            Order book with bids and asks
        """
        response = await self.client.get_market_liquidity(
            product_id=product_id,
            depth=depth,
        )
        return vars(response) if hasattr(response, '__dict__') else response

    async def get_mark_price(self, product_id: str) -> float:
        """
        Get current mark price for a product.

        Args:
            product_id: Product identifier

        Returns:
            Mark price
        """
        prices = await self.client.list_market_prices()
        for p in prices:
            price_data = vars(p) if hasattr(p, '__dict__') else p
            if price_data.get("product_id") == product_id or price_data.get("ticker") == product_id:
                return float(price_data.get("mark_price", 0))
        return 0.0

    async def get_funding_rate(self, product_id: str) -> dict[str, Any]:
        """
        Get funding rate information for a product.

        Args:
            product_id: Product identifier

        Returns:
            Funding rate information
        """
        response = await self.client.list_projected_funding()
        for f in response:
            fund_data = vars(f) if hasattr(f, '__dict__') else f
            if fund_data.get("product_id") == product_id or fund_data.get("ticker") == product_id:
                return fund_data
        return {"funding_rate": 0, "product_id": product_id}


# Global client instance
_client: Optional[EtherealClient] = None


def get_client() -> EtherealClient:
    """Get or create the global client instance."""
    global _client
    if _client is None:
        _client = EtherealClient()
    return _client
