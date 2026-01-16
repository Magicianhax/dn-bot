"""Risk management for Ethereal DN Trader."""

import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from ..config import Settings, get_settings
from ..client import EtherealClient, get_client
from ..market.data import MarketDataFetcher, ProductInfo
from ..utils.logger import get_logger

from .positions import PositionManager, Position

logger = get_logger("ethereal.risk")


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_notional: float  # Max notional per position
    max_total_notional: float  # Max total portfolio notional
    max_leverage: float  # Maximum leverage allowed
    max_delta_percent: float  # Max delta as % of margin
    liquidation_warning_percent: float  # Warn when this close to liquidation
    max_drawdown_percent: float  # Maximum allowed drawdown


@dataclass
class RiskCheck:
    """Result of a risk check."""
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error, critical


class RiskManager:
    """
    Manages risk checks and limits for trading.

    Performs pre-trade validation, monitors position risks,
    and enforces trading limits.
    """

    def __init__(
        self,
        client: Optional[EtherealClient] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the risk manager.

        Args:
            client: Ethereal client instance
            settings: Application settings
        """
        self.client = client or get_client()
        self.settings = settings or get_settings()
        self.market_data = MarketDataFetcher(self.client)
        self.position_manager = PositionManager(self.client, self.market_data)

        # Default limits
        self.limits = RiskLimits(
            max_position_notional=self.settings.max_position_size,
            max_total_notional=self.settings.max_position_size * 5,
            max_leverage=float(self.settings.default_leverage),
            max_delta_percent=self.settings.delta_threshold * 2,
            liquidation_warning_percent=0.15,  # 15% from liquidation
            max_drawdown_percent=0.1,  # 10% max drawdown
        )

        # Tracking
        self.peak_balance: Optional[float] = None
        self._alert_callbacks: list[Callable] = []

    async def check_pre_trade(
        self,
        product_id: str,
        size: float,
        side: str,
    ) -> RiskCheck:
        """
        Perform pre-trade risk checks.

        Args:
            product_id: Product to trade
            size: Order size
            side: Order side (buy/sell)

        Returns:
            RiskCheck result
        """
        checks = []

        # Check 1: Product exists and get specs
        product = await self.market_data.get_product(product_id)
        if product is None:
            return RiskCheck(
                passed=False,
                message=f"Unknown product: {product_id}",
                severity="error",
            )

        # Check 2: Get current price
        price_info = await self.market_data.get_price(product_id)
        if price_info is None:
            return RiskCheck(
                passed=False,
                message=f"Cannot get price for {product_id}",
                severity="error",
            )

        notional = size * price_info.mark_price

        # Check 3: Position notional limit
        if notional > self.limits.max_position_notional:
            return RiskCheck(
                passed=False,
                message=f"Order notional ${notional:,.2f} exceeds limit ${self.limits.max_position_notional:,.2f}",
                severity="error",
            )

        # Check 4: Product max notional
        if product.max_position_notional > 0 and notional > product.max_position_notional:
            return RiskCheck(
                passed=False,
                message=f"Order notional exceeds product limit ${product.max_position_notional:,.2f}",
                severity="error",
            )

        # Check 5: Leverage limit
        if product.max_leverage > 0 and self.limits.max_leverage > product.max_leverage:
            return RiskCheck(
                passed=False,
                message=f"Leverage exceeds product max {product.max_leverage}x",
                severity="warning",
            )

        # Check 6: Account balance
        account = await self.position_manager.get_account_summary()
        required_margin = notional / self.limits.max_leverage

        if required_margin > account.available_balance:
            return RiskCheck(
                passed=False,
                message=f"Insufficient margin. Required: ${required_margin:,.2f}, Available: ${account.available_balance:,.2f}",
                severity="error",
            )

        # Check 7: Total portfolio notional
        current_notional = account.total_position_value
        if current_notional + notional > self.limits.max_total_notional:
            return RiskCheck(
                passed=False,
                message=f"Total notional ${current_notional + notional:,.2f} exceeds limit",
                severity="error",
            )

        return RiskCheck(
            passed=True,
            message="All pre-trade checks passed",
            severity="info",
        )

    async def check_position_risks(self) -> list[RiskCheck]:
        """
        Check all current positions for risk issues.

        Returns:
            List of RiskCheck results for each issue found
        """
        checks = []

        positions = await self.position_manager.get_all_positions()
        account = await self.position_manager.get_account_summary()

        # Update peak balance for drawdown tracking
        if self.peak_balance is None or account.margin_balance > self.peak_balance:
            self.peak_balance = account.margin_balance

        # Check 1: Drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - account.margin_balance) / self.peak_balance
            if drawdown > self.limits.max_drawdown_percent:
                checks.append(RiskCheck(
                    passed=False,
                    message=f"Drawdown {drawdown*100:.1f}% exceeds limit {self.limits.max_drawdown_percent*100:.1f}%",
                    severity="critical",
                ))

        # Check 2: Delta exposure
        if account.margin_balance > 0:
            total_delta = await self.position_manager.calculate_total_delta()
            delta_percent = abs(total_delta) / account.margin_balance

            if delta_percent > self.limits.max_delta_percent:
                checks.append(RiskCheck(
                    passed=False,
                    message=f"Delta exposure {delta_percent*100:.1f}% exceeds limit",
                    severity="warning",
                ))

        # Check 3: Individual position risks
        for position in positions.values():
            # Liquidation proximity
            if position.liquidation_price > 0:
                if position.is_long:
                    distance = (position.mark_price - position.liquidation_price) / position.mark_price
                else:
                    distance = (position.liquidation_price - position.mark_price) / position.mark_price

                if distance < self.limits.liquidation_warning_percent:
                    checks.append(RiskCheck(
                        passed=False,
                        message=f"{position.product_id}: {distance*100:.1f}% from liquidation",
                        severity="critical",
                    ))

            # Position size relative to limit
            notional = position.notional_value
            if notional > self.limits.max_position_notional * 1.1:  # 10% buffer
                checks.append(RiskCheck(
                    passed=False,
                    message=f"{position.product_id}: Position notional ${notional:,.2f} exceeds limit",
                    severity="warning",
                ))

        return checks

    async def monitor_risks(
        self,
        interval: float = 30.0,
        on_risk_alert: Optional[Callable] = None,
    ) -> None:
        """
        Continuously monitor risks.

        Args:
            interval: Check interval in seconds
            on_risk_alert: Callback for risk alerts
        """
        logger.info("Starting risk monitoring...")

        while True:
            try:
                checks = await self.check_position_risks()

                for check in checks:
                    if not check.passed:
                        logger.warning(f"RISK ALERT [{check.severity.upper()}]: {check.message}")

                        if on_risk_alert:
                            on_risk_alert(check)

                        for callback in self._alert_callbacks:
                            callback(check)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(interval)

    def on_risk_alert(self, callback: Callable) -> None:
        """Register a callback for risk alerts."""
        self._alert_callbacks.append(callback)

    async def get_risk_summary(self) -> dict:
        """
        Get a summary of current risk metrics.

        Returns:
            Dictionary with risk metrics
        """
        account = await self.position_manager.get_account_summary()
        positions = await self.position_manager.get_all_positions()
        total_delta = await self.position_manager.calculate_total_delta()

        # Calculate drawdown
        drawdown = 0.0
        if self.peak_balance and self.peak_balance > 0:
            drawdown = (self.peak_balance - account.margin_balance) / self.peak_balance

        # Calculate margin usage
        margin_used = account.margin_balance - account.available_balance
        margin_usage = margin_used / account.margin_balance if account.margin_balance > 0 else 0

        # Calculate delta percentage
        delta_percent = abs(total_delta) / account.margin_balance if account.margin_balance > 0 else 0

        # Find closest liquidation
        closest_liquidation = float("inf")
        for position in positions.values():
            if position.liquidation_price > 0:
                if position.is_long:
                    distance = (position.mark_price - position.liquidation_price) / position.mark_price
                else:
                    distance = (position.liquidation_price - position.mark_price) / position.mark_price
                closest_liquidation = min(closest_liquidation, distance)

        return {
            "margin_balance": account.margin_balance,
            "available_balance": account.available_balance,
            "margin_usage_percent": margin_usage * 100,
            "total_notional": account.total_position_value,
            "total_delta": total_delta,
            "delta_percent": delta_percent * 100,
            "drawdown_percent": drawdown * 100,
            "closest_liquidation_percent": closest_liquidation * 100 if closest_liquidation != float("inf") else None,
            "position_count": len(positions),
            "limits": {
                "max_position_notional": self.limits.max_position_notional,
                "max_total_notional": self.limits.max_total_notional,
                "max_leverage": self.limits.max_leverage,
                "max_delta_percent": self.limits.max_delta_percent * 100,
                "max_drawdown_percent": self.limits.max_drawdown_percent * 100,
            },
        }

    def update_limits(
        self,
        max_position_notional: Optional[float] = None,
        max_total_notional: Optional[float] = None,
        max_leverage: Optional[float] = None,
        max_delta_percent: Optional[float] = None,
        max_drawdown_percent: Optional[float] = None,
    ) -> None:
        """
        Update risk limits.

        Args:
            max_position_notional: New max position notional
            max_total_notional: New max total notional
            max_leverage: New max leverage
            max_delta_percent: New max delta percent
            max_drawdown_percent: New max drawdown percent
        """
        if max_position_notional is not None:
            self.limits.max_position_notional = max_position_notional
        if max_total_notional is not None:
            self.limits.max_total_notional = max_total_notional
        if max_leverage is not None:
            self.limits.max_leverage = max_leverage
        if max_delta_percent is not None:
            self.limits.max_delta_percent = max_delta_percent
        if max_drawdown_percent is not None:
            self.limits.max_drawdown_percent = max_drawdown_percent

        logger.info("Risk limits updated")
