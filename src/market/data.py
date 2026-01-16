"""Market data fetching and caching for Ethereal DN Trader."""

import asyncio
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..client import EtherealClient, get_client
from ..utils.logger import get_logger

logger = get_logger("ethereal.market")


@dataclass
class ProductInfo:
    """Product specification data."""
    product_id: str
    symbol: str
    base_currency: str
    quote_currency: str
    lot_size: float
    tick_size: float
    max_leverage: int
    maker_fee: float
    taker_fee: float
    max_position_notional: float
    min_order_size: float
    funding_rate_1h: float = 0.0


@dataclass
class MarketPrice:
    """Current market price data."""
    product_id: str
    mark_price: float
    best_bid: float
    best_ask: float
    last_price: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FundingInfo:
    """Funding rate information."""
    product_id: str
    funding_rate: float
    next_funding_time: datetime
    predicted_rate: Optional[float] = None


class MarketDataFetcher:
    """
    Fetches and caches market data from Ethereal.

    Provides methods to retrieve product info, prices, and funding rates
    with optional caching to reduce API calls.
    """

    def __init__(
        self,
        client: Optional[EtherealClient] = None,
        cache_duration: int = 60,
    ):
        """
        Initialize the market data fetcher.

        Args:
            client: Ethereal client instance
            cache_duration: Cache duration in seconds
        """
        self.client = client or get_client()
        self.cache_duration = timedelta(seconds=cache_duration)

        # Caches
        self._products_cache: dict[str, ProductInfo] = {}
        self._products_cache_time: Optional[datetime] = None
        self._prices_cache: dict[str, MarketPrice] = {}
        self._funding_cache: dict[str, FundingInfo] = {}

    def _is_cache_valid(self, cache_time: Optional[datetime]) -> bool:
        """Check if cache is still valid."""
        if cache_time is None:
            return False
        return datetime.now() - cache_time < self.cache_duration

    async def get_all_products(self, force_refresh: bool = False) -> dict[str, ProductInfo]:
        """
        Get all available trading products.

        Args:
            force_refresh: Force refresh from API even if cached

        Returns:
            Dictionary of product_id -> ProductInfo
        """
        if not force_refresh and self._is_cache_valid(self._products_cache_time):
            return self._products_cache

        logger.debug("Fetching all products from API...")
        raw_products = await self.client.list_products()

        self._products_cache = {}
        for product in raw_products:
            try:
                # Handle both dict and pydantic model responses
                if hasattr(product, 'ticker'):
                    # Pydantic model
                    info = ProductInfo(
                        product_id=product.ticker,
                        symbol=product.display_ticker or product.ticker,
                        base_currency=product.base_token_name or "",
                        quote_currency=product.quote_token_name or "USD",
                        lot_size=float(product.lot_size or 0.0001),
                        tick_size=float(product.tick_size or 1),
                        max_leverage=int(product.max_leverage or 10),
                        maker_fee=float(product.maker_fee or 0),
                        taker_fee=float(product.taker_fee or 0.0003),
                        max_position_notional=float(product.max_position_notional_usd or 0),
                        min_order_size=float(product.min_quantity or 0),
                        funding_rate_1h=float(product.funding_rate1h or 0),
                    )
                else:
                    # Dict response
                    info = ProductInfo(
                        product_id=product.get("ticker", product.get("product_id", "")),
                        symbol=product.get("display_ticker", product.get("symbol", "")),
                        base_currency=product.get("base_token_name", ""),
                        quote_currency=product.get("quote_token_name", "USD"),
                        lot_size=float(product.get("lot_size", 0.0001)),
                        tick_size=float(product.get("tick_size", 1)),
                        max_leverage=int(float(product.get("max_leverage", 10))),
                        maker_fee=float(product.get("maker_fee", 0)),
                        taker_fee=float(product.get("taker_fee", 0.0003)),
                        max_position_notional=float(product.get("max_position_notional_usd", 0)),
                        min_order_size=float(product.get("min_quantity", 0)),
                        funding_rate_1h=float(product.get("funding_rate1h", 0)),
                    )
                self._products_cache[info.product_id] = info
            except (KeyError, ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse product: {e}")

        self._products_cache_time = datetime.now()
        logger.info(f"Loaded {len(self._products_cache)} products")
        return self._products_cache

    async def get_product(self, product_id: str) -> Optional[ProductInfo]:
        """
        Get product information by ID.

        Args:
            product_id: Product identifier (e.g., "BTCUSD")

        Returns:
            ProductInfo or None if not found
        """
        products = await self.get_all_products()
        return products.get(product_id)

    async def get_price(
        self,
        product_id: str,
        force_refresh: bool = False,
    ) -> Optional[MarketPrice]:
        """
        Get current price for a product.

        Args:
            product_id: Product identifier
            force_refresh: Force refresh from API

        Returns:
            MarketPrice or None if not available
        """
        cached = self._prices_cache.get(product_id)
        if not force_refresh and cached:
            cache_age = datetime.now() - cached.timestamp
            if cache_age < timedelta(seconds=5):  # Short cache for prices
                return cached

        try:
            mark_price = await self.client.get_mark_price(product_id)
            orderbook = await self.client.get_orderbook(product_id, depth=1)

            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])

            price = MarketPrice(
                product_id=product_id,
                mark_price=mark_price,
                best_bid=float(bids[0][0]) if bids else mark_price,
                best_ask=float(asks[0][0]) if asks else mark_price,
                last_price=mark_price,
                timestamp=datetime.now(),
            )
            self._prices_cache[product_id] = price
            return price
        except Exception as e:
            logger.error(f"Failed to fetch price for {product_id}: {e}")
            return cached

    async def get_prices(
        self,
        product_ids: list[str],
    ) -> dict[str, MarketPrice]:
        """
        Get prices for multiple products concurrently.

        Args:
            product_ids: List of product identifiers

        Returns:
            Dictionary of product_id -> MarketPrice
        """
        tasks = [self.get_price(pid) for pid in product_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for pid, result in zip(product_ids, results):
            if isinstance(result, MarketPrice):
                prices[pid] = result
            elif isinstance(result, Exception):
                logger.error(f"Failed to get price for {pid}: {result}")

        return prices

    async def get_funding_rate(
        self,
        product_id: str,
        force_refresh: bool = False,
    ) -> Optional[FundingInfo]:
        """
        Get funding rate for a product.

        Args:
            product_id: Product identifier
            force_refresh: Force refresh from API

        Returns:
            FundingInfo or None if not available
        """
        cached = self._funding_cache.get(product_id)
        if not force_refresh and cached:
            cache_age = datetime.now() - cached.next_funding_time
            if cache_age < timedelta(minutes=5):
                return cached

        try:
            # Get funding from product list (funding_rate1h is available there)
            products = await self.get_all_products(force_refresh=force_refresh)
            product = products.get(product_id)

            if product:
                # Calculate next funding time (every hour on the hour)
                now = datetime.now()
                next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

                funding = FundingInfo(
                    product_id=product_id,
                    funding_rate=product.funding_rate_1h,
                    next_funding_time=next_hour,
                    predicted_rate=None,
                )
                self._funding_cache[product_id] = funding
                return funding
            return cached
        except Exception as e:
            logger.error(f"Failed to fetch funding rate for {product_id}: {e}")
            return cached

    async def get_all_funding_rates(
        self,
        product_ids: Optional[list[str]] = None,
    ) -> dict[str, FundingInfo]:
        """
        Get funding rates for multiple products.

        Args:
            product_ids: List of product IDs. If None, fetches all products.

        Returns:
            Dictionary of product_id -> FundingInfo
        """
        if product_ids is None:
            products = await self.get_all_products()
            product_ids = list(products.keys())

        tasks = [self.get_funding_rate(pid) for pid in product_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        funding_rates = {}
        for pid, result in zip(product_ids, results):
            if isinstance(result, FundingInfo):
                funding_rates[pid] = result

        return funding_rates

    async def get_orderbook(
        self,
        product_id: str,
        depth: int = 10,
    ) -> dict[str, Any]:
        """
        Get order book for a product.

        Args:
            product_id: Product identifier
            depth: Number of levels

        Returns:
            Order book with bids and asks
        """
        return await self.client.get_orderbook(product_id, depth)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._products_cache.clear()
        self._products_cache_time = None
        self._prices_cache.clear()
        self._funding_cache.clear()
        logger.debug("Cache cleared")
