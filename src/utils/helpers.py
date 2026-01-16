"""Helper utilities for Ethereal DN Trader."""

from decimal import Decimal, ROUND_DOWN
from typing import Union


def format_price(price: Union[float, Decimal, str], decimals: int = 2) -> str:
    """
    Format a price value for display.

    Args:
        price: Price value to format
        decimals: Number of decimal places

    Returns:
        Formatted price string with dollar sign
    """
    if isinstance(price, str):
        price = Decimal(price)
    elif isinstance(price, float):
        price = Decimal(str(price))

    formatted = f"${price:,.{decimals}f}"
    return formatted


def format_size(size: Union[float, Decimal, str], decimals: int = 4) -> str:
    """
    Format a position size for display.

    Args:
        size: Size value to format
        decimals: Number of decimal places

    Returns:
        Formatted size string
    """
    if isinstance(size, str):
        size = Decimal(size)
    elif isinstance(size, float):
        size = Decimal(str(size))

    return f"{size:.{decimals}f}"


def calculate_notional(size: float, price: float) -> float:
    """
    Calculate notional value of a position.

    Args:
        size: Position size
        price: Current price

    Returns:
        Notional value in USD
    """
    return abs(size * price)


def round_to_lot_size(size: float, lot_size: float) -> float:
    """
    Round a size down to the nearest lot size.

    Args:
        size: Desired size
        lot_size: Minimum lot size for the product

    Returns:
        Rounded size
    """
    if lot_size <= 0:
        return size

    lots = Decimal(str(size)) / Decimal(str(lot_size))
    rounded_lots = lots.quantize(Decimal("1"), rounding=ROUND_DOWN)
    return float(rounded_lots * Decimal(str(lot_size)))


def round_to_tick_size(price: float, tick_size: float) -> float:
    """
    Round a price to the nearest tick size.

    Args:
        price: Desired price
        tick_size: Minimum tick size for the product

    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price

    ticks = Decimal(str(price)) / Decimal(str(tick_size))
    rounded_ticks = ticks.quantize(Decimal("1"), rounding=ROUND_DOWN)
    return float(rounded_ticks * Decimal(str(tick_size)))


def calculate_pnl(
    entry_price: float,
    current_price: float,
    size: float,
    is_long: bool
) -> float:
    """
    Calculate unrealized PnL for a position.

    Args:
        entry_price: Position entry price
        current_price: Current market price
        size: Position size (absolute value)
        is_long: Whether position is long

    Returns:
        Unrealized PnL in USD
    """
    price_diff = current_price - entry_price
    if not is_long:
        price_diff = -price_diff
    return price_diff * abs(size)


def calculate_pnl_percent(
    entry_price: float,
    current_price: float,
    is_long: bool
) -> float:
    """
    Calculate PnL as a percentage.

    Args:
        entry_price: Position entry price
        current_price: Current market price
        is_long: Whether position is long

    Returns:
        PnL percentage (0.01 = 1%)
    """
    if entry_price == 0:
        return 0.0

    price_diff = current_price - entry_price
    if not is_long:
        price_diff = -price_diff
    return price_diff / entry_price


def bytes32_encode(name: str) -> str:
    """
    Encode a string to bytes32 hex format.

    Args:
        name: String to encode (e.g., "primary")

    Returns:
        Hex string representation of bytes32
    """
    encoded = name.encode("utf-8")
    padded = encoded.ljust(32, b"\x00")[:32]
    return "0x" + padded.hex()
