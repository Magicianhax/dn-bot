"""Configuration management for Ethereal Points Farming Tool."""

import json
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# Path to persist runtime settings
SETTINGS_FILE = Path(__file__).parent.parent / "data" / "settings.json"
TRADES_FILE = Path(__file__).parent.parent / "data" / "active_trades.json"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Account 1 Configuration (Long Side)
    account1_private_key: str = Field(
        default="",
        description="Linked signer private key for Account 1"
    )
    account1_wallet_address: str = Field(
        default="",
        description="Main wallet address for Account 1"
    )

    # Account 2 Configuration (Short Side)
    account2_private_key: str = Field(
        default="",
        description="Linked signer private key for Account 2"
    )
    account2_wallet_address: str = Field(
        default="",
        description="Main wallet address for Account 2"
    )

    # Legacy single account support
    ethereal_private_key: str = Field(
        default="",
        description="Legacy: Single account private key"
    )
    ethereal_wallet_address: str = Field(
        default="",
        description="Legacy: Single account wallet address"
    )

    # Trading Parameters
    trading_pairs: str = Field(
        default="BTCUSD,ETHUSD",
        description="Comma-separated trading pairs"
    )
    leverage: int = Field(
        default=10,
        description="Default trading leverage (used if no market-specific limit)"
    )
    max_concurrent_trades: int = Field(
        default=2,
        description="Number of trades to open simultaneously (each uses different pair)"
    )
    market_max_leverage: str = Field(
        default="BTCUSD:20,ETHUSD:20",
        description="Market-specific max leverage limits (format: PAIR:LEV,PAIR:LEV). Markets not listed default to 10x."
    )
    position_size: float = Field(
        default=0.0,
        description="Position size in USD (0 = use full balance)"
    )
    use_full_balance: bool = Field(
        default=True,
        description="Use full available balance for trading"
    )
    min_balance_threshold: float = Field(
        default=10.0,
        description="Stop trading when balance falls below this amount"
    )
    max_daily_trades: int = Field(
        default=100,
        description="Maximum number of trades per day (resets at midnight)",
        alias="max_trades",
    )

    # Risk Management
    stop_loss_percent: float = Field(
        default=0.05,
        description="Stop loss percentage (0.05 = 5%)"
    )
    take_profit_percent: float = Field(
        default=0.05,
        description="Take profit percentage (0.05 = 5%)"
    )

    # Time Settings
    min_hold_time_minutes: int = Field(
        default=30,
        description="Minimum hold time in minutes"
    )
    max_hold_time_minutes: int = Field(
        default=120,
        description="Maximum hold time in minutes"
    )
    min_trade_delay_seconds: int = Field(
        default=60,
        description="Minimum delay between trades in seconds",
        alias="trade_delay_seconds",
    )
    max_trade_delay_seconds: int = Field(
        default=300,
        description="Maximum delay between trades in seconds"
    )

    # API Configuration
    ethereal_api_url: str = Field(
        default="https://api.ethereal.trade",
        description="Ethereal API base URL"
    )
    ethereal_ws_url: str = Field(
        default="wss://ws.ethereal.trade/v1/stream",
        description="Ethereal WebSocket URL"
    )
    ethereal_rpc_url: str = Field(
        default="https://rpc.ethereal.trade",
        description="Blockchain RPC URL"
    )

    # Account Configuration
    subaccount_name: str = Field(
        default="primary",
        description="Subaccount identifier"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    @field_validator("leverage")
    @classmethod
    def validate_leverage(cls, v: int) -> int:
        """Validate leverage is within acceptable range."""
        if v < 1 or v > 20:
            raise ValueError("Leverage must be between 1 and 20")
        return v

    @field_validator("max_concurrent_trades")
    @classmethod
    def validate_max_concurrent_trades(cls, v: int) -> int:
        """Validate max concurrent trades."""
        if v < 1 or v > 10:
            raise ValueError("Max concurrent trades must be between 1 and 10")
        return v

    @field_validator("stop_loss_percent", "take_profit_percent")
    @classmethod
    def validate_percent(cls, v: float) -> float:
        """Validate percentage is reasonable."""
        if v < 0.001 or v > 0.5:
            raise ValueError("Percentage must be between 0.1% and 50%")
        return v

    @property
    def trading_pairs_list(self) -> List[str]:
        """Get trading pairs as a list."""
        return [p.strip().upper() for p in self.trading_pairs.split(",") if p.strip()]

    @property
    def market_leverage_limits(self) -> dict[str, int]:
        """Parse market_max_leverage string into a dict.

        Format: "BTCUSD:20,ETHUSD:20,SOLUSD:10"
        Returns: {"BTCUSD": 20, "ETHUSD": 20, "SOLUSD": 10}
        """
        limits = {}
        if not self.market_max_leverage:
            return limits
        for item in self.market_max_leverage.split(","):
            item = item.strip()
            if ":" in item:
                pair, lev = item.split(":", 1)
                try:
                    limits[pair.strip().upper()] = int(lev.strip())
                except ValueError:
                    pass
        return limits

    def get_leverage_for_market(self, market: str) -> int:
        """Get effective leverage for a specific market.

        Returns market-specific limit if defined, otherwise default leverage.
        Ensures requested leverage doesn't exceed market max.
        """
        market = market.upper()
        limits = self.market_leverage_limits
        max_lev = limits.get(market, 10)  # Default max is 10x for unlisted markets
        return min(self.leverage, max_lev)

    @property
    def has_dual_accounts(self) -> bool:
        """Check if dual accounts are configured."""
        return bool(
            self.account1_private_key and self.account1_wallet_address and
            self.account2_private_key and self.account2_wallet_address
        )

    @property
    def account1_config(self) -> dict:
        """Get Account 1 configuration."""
        return {
            "private_key": self.account1_private_key,
            "wallet_address": self.account1_wallet_address,
            "base_url": self.ethereal_api_url,
        }

    @property
    def account2_config(self) -> dict:
        """Get Account 2 configuration."""
        return {
            "private_key": self.account2_private_key,
            "wallet_address": self.account2_wallet_address,
            "base_url": self.ethereal_api_url,
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields for backward compatibility
        "populate_by_name": True,  # Allow both alias and field name
        "validate_assignment": True,  # Allow attribute assignment after creation
    }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Load saved runtime settings on top of env defaults
        _load_saved_settings(_settings)
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings()
    _load_saved_settings(_settings)
    return _settings


def save_settings(settings: Settings) -> bool:
    """Save current settings to JSON file for persistence."""
    try:
        # Ensure data directory exists
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Save only the runtime-configurable settings
        data = {
            "trading_pairs": settings.trading_pairs,
            "leverage": settings.leverage,
            "max_concurrent_trades": settings.max_concurrent_trades,
            "market_max_leverage": settings.market_max_leverage,
            "position_size": settings.position_size,
            "use_full_balance": settings.use_full_balance,
            "min_balance_threshold": settings.min_balance_threshold,
            "stop_loss_percent": settings.stop_loss_percent,
            "take_profit_percent": settings.take_profit_percent,
            "min_hold_time_minutes": settings.min_hold_time_minutes,
            "max_hold_time_minutes": settings.max_hold_time_minutes,
            "min_trade_delay_seconds": settings.min_trade_delay_seconds,
            "max_trade_delay_seconds": settings.max_trade_delay_seconds,
            "max_daily_trades": settings.max_daily_trades,
        }
        
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Failed to save settings: {e}")
        return False


def _load_saved_settings(settings: Settings) -> None:
    """Load saved settings from JSON file and apply to settings object."""
    if not SETTINGS_FILE.exists():
        return
    
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        
        # Apply saved values
        if "trading_pairs" in data:
            settings.trading_pairs = data["trading_pairs"]
        if "leverage" in data:
            settings.leverage = data["leverage"]
        if "max_concurrent_trades" in data:
            settings.max_concurrent_trades = data["max_concurrent_trades"]
        if "market_max_leverage" in data:
            settings.market_max_leverage = data["market_max_leverage"]
        if "position_size" in data:
            settings.position_size = data["position_size"]
        if "use_full_balance" in data:
            settings.use_full_balance = data["use_full_balance"]
        if "min_balance_threshold" in data:
            settings.min_balance_threshold = data["min_balance_threshold"]
        if "stop_loss_percent" in data:
            settings.stop_loss_percent = data["stop_loss_percent"]
        if "take_profit_percent" in data:
            settings.take_profit_percent = data["take_profit_percent"]
        if "min_hold_time_minutes" in data:
            settings.min_hold_time_minutes = data["min_hold_time_minutes"]
        if "max_hold_time_minutes" in data:
            settings.max_hold_time_minutes = data["max_hold_time_minutes"]
        if "min_trade_delay_seconds" in data:
            settings.min_trade_delay_seconds = data["min_trade_delay_seconds"]
        if "max_trade_delay_seconds" in data:
            settings.max_trade_delay_seconds = data["max_trade_delay_seconds"]
        if "max_daily_trades" in data:
            settings.max_daily_trades = data["max_daily_trades"]
        
        print(f"Loaded saved settings from {SETTINGS_FILE}")
    except Exception as e:
        print(f"Failed to load saved settings: {e}")


def save_active_trades(trades: dict) -> bool:
    """Save active trades to JSON file for persistence across restarts."""
    try:
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert trades to serializable format
        data = {}
        for trade_id, trade in trades.items():
            data[trade_id] = {
                "id": trade.id,
                "product_id": trade.product_id,
                "size": trade.size,
                "leverage": trade.leverage,
                "entry_price": trade.entry_price,
                "long_entry_price": trade.long_entry_price,
                "short_entry_price": trade.short_entry_price,
                "account1_is_long": trade.account1_is_long,
                "opened_at": trade.opened_at.isoformat(),
                "target_hold_minutes": trade.target_hold_minutes,
            }
        
        with open(TRADES_FILE, "w") as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Failed to save active trades: {e}")
        return False


def load_active_trades() -> dict:
    """Load saved active trades from JSON file."""
    if not TRADES_FILE.exists():
        return {}
    
    try:
        with open(TRADES_FILE, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} saved trade(s) from {TRADES_FILE}")
        return data
    except Exception as e:
        print(f"Failed to load active trades: {e}")
        return {}


def clear_active_trades() -> None:
    """Clear the saved active trades file."""
    try:
        if TRADES_FILE.exists():
            TRADES_FILE.unlink()
    except Exception as e:
        print(f"Failed to clear active trades: {e}")
