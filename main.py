"""Main entry point for Ethereal DN Trader."""

import asyncio
import sys
from typing import Optional
from functools import wraps

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config import get_settings, Settings
from src.client import EtherealClient, get_client
from src.market.data import MarketDataFetcher
from src.trading.orders import OrderManager, OrderSide
from src.trading.positions import PositionManager
from src.trading.dn_strategy import DNStrategy
from src.trading.points_strategy import PointsFarmingStrategy
from src.trading.risk import RiskManager
from src.utils.logger import setup_logger, get_logger

console = Console()
logger = get_logger("ethereal")


def async_command(f):
    """Decorator to run async click commands."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool):
    """Ethereal DN Trading Tool - Delta-neutral trading on Ethereal DEX."""
    level = "DEBUG" if debug else "INFO"
    setup_logger("ethereal", level=level)


@cli.command()
@async_command
async def start():
    """Start the DN trading bot."""
    console.print(Panel.fit(
        "[bold blue]Ethereal DN Trader[/bold blue]\n"
        "Starting delta-neutral trading strategy...",
        border_style="blue",
    ))

    try:
        settings = get_settings()
        client = get_client()

        async with client.session():
            strategy = DNStrategy(client, settings)
            await strategy.initialize()

            # Register callbacks
            strategy.on_trade(lambda p, s, sz, pr: console.print(
                f"[green]Trade:[/green] {s.value} {sz:.4f} {p} @ ${pr:,.2f}"
            ))
            strategy.on_rebalance(lambda p, sz: console.print(
                f"[yellow]Rebalance:[/yellow] {p} by {sz:.4f}"
            ))

            console.print("[green]Strategy started. Press Ctrl+C to stop.[/green]")

            try:
                await strategy.start()
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping strategy...[/yellow]")
                await strategy.stop()

            await strategy.shutdown()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command("farm")
@async_command
async def farm():
    """Start the points farming bot (dual account long/short strategy)."""
    console.print(Panel.fit(
        "[bold green]Ethereal Points Farmer[/bold green]\n"
        "Starting dual-account points farming strategy...",
        border_style="green",
    ))

    try:
        settings = get_settings()

        if not settings.has_dual_accounts:
            console.print("[red]Error: Dual accounts not configured![/red]")
            console.print("Set ACCOUNT1_* and ACCOUNT2_* in your .env file")
            sys.exit(1)

        # Show configuration
        config_table = Table(title="Configuration", show_header=False)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Trading Pairs", ", ".join(settings.trading_pairs_list))
        config_table.add_row("Leverage", f"{settings.leverage}x")
        config_table.add_row("Position Size", f"${settings.position_size}")
        config_table.add_row("Max Daily Trades", str(settings.max_daily_trades))
        config_table.add_row("Stop Loss", f"{settings.stop_loss_percent*100:.1f}%")
        config_table.add_row("Take Profit", f"{settings.take_profit_percent*100:.1f}%")
        config_table.add_row("Min Hold Time", f"{settings.min_hold_time_minutes} min")
        config_table.add_row("Max Hold Time", f"{settings.max_hold_time_minutes} min")

        console.print(config_table)

        strategy = PointsFarmingStrategy(settings)

        if not await strategy.initialize():
            console.print("[red]Failed to initialize strategy[/red]")
            sys.exit(1)

        # Register callbacks
        strategy.on_trade_open(lambda t: console.print(
            f"[green]OPENED:[/green] {t.product_id} | Long+Short @ ${t.entry_price:,.2f}"
        ))
        strategy.on_trade_close(lambda t: console.print(
            f"[yellow]CLOSED:[/yellow] {t.product_id} | {t.close_reason} | "
            f"PnL: ${t.total_pnl:+,.2f} | Hold: {t.hold_time_minutes:.1f}m"
        ))

        console.print("\n[green]Strategy started. Press Ctrl+C to stop.[/green]")

        try:
            await strategy.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping strategy...[/yellow]")
            await strategy.stop()

        await strategy.shutdown()

        # Show final stats
        status = await strategy.get_status()
        console.print(f"\n[bold]Final Stats:[/bold]")
        console.print(f"  Total Trades: {status['completed_trades']}")
        console.print(f"  Total PnL: ${status['total_pnl']:+,.2f}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command("farm-status")
@async_command
async def farm_status():
    """Show points farming status and configuration."""
    try:
        settings = get_settings()

        # Account status table
        table = Table(title="Points Farming Configuration", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        # Check dual accounts
        if settings.has_dual_accounts:
            table.add_row("Account 1 (Long)", f"{settings.account1_wallet_address[:10]}...{settings.account1_wallet_address[-6:]}")
            table.add_row("Account 2 (Short)", f"{settings.account2_wallet_address[:10]}...{settings.account2_wallet_address[-6:]}")
        else:
            table.add_row("Dual Accounts", "[red]Not configured[/red]")

        table.add_row("", "")
        table.add_row("[bold]Trading Settings[/bold]", "")
        table.add_row("Trading Pairs", ", ".join(settings.trading_pairs_list))
        table.add_row("Leverage", f"{settings.leverage}x")
        table.add_row("Position Size", f"${settings.position_size}")
        table.add_row("Max Trades", str(settings.max_daily_trades))

        table.add_row("", "")
        table.add_row("[bold]Risk Management[/bold]", "")
        table.add_row("Stop Loss", f"{settings.stop_loss_percent*100:.1f}%")
        table.add_row("Take Profit", f"{settings.take_profit_percent*100:.1f}%")

        table.add_row("", "")
        table.add_row("[bold]Time Settings[/bold]", "")
        table.add_row("Min Hold Time", f"{settings.min_hold_time_minutes} minutes")
        table.add_row("Max Hold Time", f"{settings.max_hold_time_minutes} minutes")
        table.add_row("Trade Delay", f"{settings.trade_delay_seconds} seconds")

        console.print(table)

        if not settings.has_dual_accounts:
            console.print("\n[yellow]To use points farming, configure these in .env:[/yellow]")
            console.print("  ACCOUNT1_PRIVATE_KEY=<linked signer key>")
            console.print("  ACCOUNT1_WALLET_ADDRESS=<main wallet>")
            console.print("  ACCOUNT2_PRIVATE_KEY=<linked signer key>")
            console.print("  ACCOUNT2_WALLET_ADDRESS=<main wallet>")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command("farm-close")
@async_command
async def farm_close():
    """Close all positions on both farming accounts."""
    try:
        settings = get_settings()

        if not settings.has_dual_accounts:
            console.print("[red]Error: Dual accounts not configured![/red]")
            sys.exit(1)

        console.print("Initializing strategy to close positions...")
        strategy = PointsFarmingStrategy(settings)
        await strategy.initialize()

        console.print("Closing all active trades...")
        await strategy.close_all()

        await strategy.shutdown()
        console.print("[green]All positions closed[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@async_command
async def status():
    """Show current positions and account status."""
    try:
        client = get_client()

        async with client.session():
            position_manager = PositionManager(client)
            account = await position_manager.get_account_summary()
            positions = await position_manager.get_all_positions()
            delta = await position_manager.calculate_total_delta()

            # Account table
            account_table = Table(title="Account Summary", show_header=False)
            account_table.add_column("Metric", style="cyan")
            account_table.add_column("Value", style="green")

            account_table.add_row("Margin Balance", f"${account.margin_balance:,.2f}")
            account_table.add_row("Available Balance", f"${account.available_balance:,.2f}")
            account_table.add_row("Unrealized PnL", f"${account.unrealized_pnl:+,.2f}")
            account_table.add_row("Total Position Value", f"${account.total_position_value:,.2f}")
            account_table.add_row("Margin Usage", f"{account.margin_usage_percent*100:.1f}%")
            account_table.add_row("Net Delta", f"${delta:+,.2f}")

            console.print(account_table)

            # Positions table
            if positions:
                pos_table = Table(title="Open Positions")
                pos_table.add_column("Product", style="cyan")
                pos_table.add_column("Side", style="bold")
                pos_table.add_column("Size", justify="right")
                pos_table.add_column("Entry", justify="right")
                pos_table.add_column("Mark", justify="right")
                pos_table.add_column("Liq. Price", justify="right")
                pos_table.add_column("PnL", justify="right")

                for pos in positions.values():
                    side_color = "green" if pos.is_long else "red"
                    pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"

                    pos_table.add_row(
                        pos.product_id,
                        f"[{side_color}]{pos.side}[/{side_color}]",
                        f"{pos.size:.4f}",
                        f"${pos.entry_price:,.2f}",
                        f"${pos.mark_price:,.2f}",
                        f"${pos.liquidation_price:,.2f}",
                        f"[{pnl_color}]${pos.unrealized_pnl:+,.2f}[/{pnl_color}]",
                    )

                console.print(pos_table)
            else:
                console.print("[dim]No open positions[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@async_command
async def balance():
    """Show account balance."""
    try:
        client = get_client()

        async with client.session():
            balance = await client.get_balance()

            table = Table(title="Account Balance", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Margin Balance", f"${float(balance.get('margin_balance', 0)):,.2f}")
            table.add_row("Available Balance", f"${float(balance.get('available_balance', 0)):,.2f}")
            table.add_row("Unrealized PnL", f"${float(balance.get('unrealized_pnl', 0)):+,.2f}")

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("product", required=False)
@async_command
async def close(product: Optional[str]):
    """Close position(s). Specify PRODUCT or close all if not specified."""
    try:
        client = get_client()

        async with client.session():
            position_manager = PositionManager(client)
            order_manager = OrderManager(client)

            if product:
                # Close specific product
                position = await position_manager.get_position(product.upper())
                if position is None or position.size == 0:
                    console.print(f"[yellow]No position in {product.upper()}[/yellow]")
                    return

                side = OrderSide.SELL if position.size > 0 else OrderSide.BUY
                size = abs(position.size)

                console.print(f"Closing {product.upper()}: {side.value} {size:.4f}")
                order = await order_manager.place_market_order(
                    product_id=product.upper(),
                    side=side,
                    size=size,
                    reduce_only=True,
                )
                console.print(f"[green]Position closed. Order ID: {order.order_id}[/green]")

            else:
                # Close all positions
                positions = await position_manager.get_all_positions()
                if not positions:
                    console.print("[yellow]No open positions[/yellow]")
                    return

                console.print(f"Closing {len(positions)} position(s)...")
                for pos in positions.values():
                    side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                    size = abs(pos.size)

                    console.print(f"  Closing {pos.product_id}: {side.value} {size:.4f}")
                    await order_manager.place_market_order(
                        product_id=pos.product_id,
                        side=side,
                        size=size,
                        reduce_only=True,
                    )

                console.print("[green]All positions closed[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@async_command
async def products():
    """List available trading products."""
    try:
        client = get_client()

        async with client.session():
            market_data = MarketDataFetcher(client)
            prods = await market_data.get_all_products()

            table = Table(title="Available Products")
            table.add_column("Product", style="cyan")
            table.add_column("Lot Size", justify="right")
            table.add_column("Tick Size", justify="right")
            table.add_column("Max Lev.", justify="right")
            table.add_column("Maker Fee", justify="right")
            table.add_column("Taker Fee", justify="right")

            for product in prods.values():
                table.add_row(
                    product.product_id,
                    str(product.lot_size),
                    str(product.tick_size),
                    f"{product.max_leverage}x",
                    f"{product.maker_fee*100:.2f}%",
                    f"{product.taker_fee*100:.2f}%",
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@async_command
async def funding():
    """Show current funding rates."""
    try:
        client = get_client()

        async with client.session():
            market_data = MarketDataFetcher(client)
            prods = await market_data.get_all_products()
            funding_rates = await market_data.get_all_funding_rates(list(prods.keys()))

            table = Table(title="Funding Rates")
            table.add_column("Product", style="cyan")
            table.add_column("Funding Rate", justify="right")
            table.add_column("Predicted", justify="right")
            table.add_column("Next Funding", justify="right")

            for product_id, fund in sorted(
                funding_rates.items(),
                key=lambda x: abs(x[1].funding_rate),
                reverse=True,
            ):
                rate = fund.funding_rate * 100
                rate_color = "green" if rate < 0 else "red" if rate > 0 else "white"

                predicted = ""
                if fund.predicted_rate is not None:
                    predicted = f"{fund.predicted_rate*100:+.4f}%"

                table.add_row(
                    product_id,
                    f"[{rate_color}]{rate:+.4f}%[/{rate_color}]",
                    predicted,
                    fund.next_funding_time.strftime("%H:%M:%S"),
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@async_command
async def risk():
    """Show risk metrics and limits."""
    try:
        client = get_client()

        async with client.session():
            risk_manager = RiskManager(client)
            summary = await risk_manager.get_risk_summary()

            table = Table(title="Risk Summary", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Limit", style="yellow")

            table.add_row(
                "Margin Balance",
                f"${summary['margin_balance']:,.2f}",
                "-",
            )
            table.add_row(
                "Margin Usage",
                f"{summary['margin_usage_percent']:.1f}%",
                "-",
            )
            table.add_row(
                "Total Notional",
                f"${summary['total_notional']:,.2f}",
                f"${summary['limits']['max_total_notional']:,.2f}",
            )
            table.add_row(
                "Delta Exposure",
                f"{summary['delta_percent']:.1f}%",
                f"{summary['limits']['max_delta_percent']:.1f}%",
            )
            table.add_row(
                "Drawdown",
                f"{summary['drawdown_percent']:.1f}%",
                f"{summary['limits']['max_drawdown_percent']:.1f}%",
            )

            if summary['closest_liquidation_percent'] is not None:
                table.add_row(
                    "Closest Liquidation",
                    f"{summary['closest_liquidation_percent']:.1f}%",
                    "-",
                )

            console.print(table)

            # Check for active risk alerts
            checks = await risk_manager.check_position_risks()
            if checks:
                console.print("\n[bold red]Active Risk Alerts:[/bold red]")
                for check in checks:
                    console.print(f"  [{check.severity.upper()}] {check.message}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@async_command
async def config():
    """Show current configuration."""
    try:
        settings = get_settings()

        table = Table(title="Configuration", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("[bold]API Settings[/bold]", "")
        table.add_row("API URL", settings.ethereal_api_url)
        table.add_row("WebSocket URL", settings.ethereal_ws_url)
        table.add_row("Subaccount", settings.subaccount_name)

        table.add_row("", "")
        table.add_row("[bold]Points Farming[/bold]", "")
        table.add_row("Dual Accounts", "[green]Configured[/green]" if settings.has_dual_accounts else "[red]Not configured[/red]")
        table.add_row("Trading Pairs", ", ".join(settings.trading_pairs_list))
        table.add_row("Leverage", f"{settings.leverage}x")
        table.add_row("Position Size", f"${settings.position_size}")
        table.add_row("Max Trades", str(settings.max_daily_trades))

        table.add_row("", "")
        table.add_row("[bold]Risk Management[/bold]", "")
        table.add_row("Take Profit", f"{settings.take_profit_percent*100:.1f}%")
        table.add_row("Stop Loss", f"{settings.stop_loss_percent*100:.1f}%")

        table.add_row("", "")
        table.add_row("[bold]Time Settings[/bold]", "")
        table.add_row("Min Hold Time", f"{settings.min_hold_time_minutes} min")
        table.add_row("Max Hold Time", f"{settings.max_hold_time_minutes} min")
        table.add_row("Trade Delay", f"{settings.trade_delay_seconds} sec")

        table.add_row("", "")
        table.add_row("Log Level", settings.log_level)

        # Show account info if configured
        if settings.has_dual_accounts:
            table.add_row("", "")
            table.add_row("[bold]Accounts[/bold]", "")
            table.add_row("Account 1 (Long)", f"{settings.account1_wallet_address[:10]}...{settings.account1_wallet_address[-6:]}")
            table.add_row("Account 2 (Short)", f"{settings.account2_wallet_address[:10]}...{settings.account2_wallet_address[-6:]}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("product")
@click.argument("side", type=click.Choice(["buy", "sell"]))
@click.argument("size", type=float)
@click.option("--price", "-p", type=float, help="Limit price (market order if not specified)")
@async_command
async def order(product: str, side: str, size: float, price: Optional[float]):
    """Place a manual order. Example: order BTCUSD buy 0.01"""
    try:
        client = get_client()

        async with client.session():
            order_manager = OrderManager(client)
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            if price:
                console.print(f"Placing limit {side} order: {size} {product.upper()} @ ${price:,.2f}")
                result = await order_manager.place_limit_order(
                    product_id=product.upper(),
                    side=order_side,
                    size=size,
                    price=price,
                )
            else:
                console.print(f"Placing market {side} order: {size} {product.upper()}")
                result = await order_manager.place_market_order(
                    product_id=product.upper(),
                    side=order_side,
                    size=size,
                )

            status_color = "green" if result.is_filled else "yellow"
            console.print(f"[{status_color}]Order {result.status.value}[/{status_color}]")
            console.print(f"  Order ID: {result.order_id}")
            console.print(f"  Filled: {result.filled_size:.4f}")
            if result.average_price:
                console.print(f"  Avg Price: ${result.average_price:,.2f}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command("cancel-all")
@click.option("--product", "-p", help="Only cancel orders for this product")
@async_command
async def cancel_all(product: Optional[str]):
    """Cancel all open orders."""
    try:
        client = get_client()

        async with client.session():
            order_manager = OrderManager(client)

            console.print(f"Cancelling orders for {product.upper() if product else 'all products'}...")
            count = await order_manager.cancel_all_orders(
                product_id=product.upper() if product else None
            )
            console.print(f"[green]Cancelled {count} order(s)[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8080, help="Port to run on")
@async_command
async def dashboard(host: str, port: int):
    """Launch the web dashboard for controlling the bot."""
    console.print(Panel.fit(
        "[bold cyan]Ethereal Trading Dashboard[/bold cyan]\n"
        f"Starting web server on http://{host}:{port}",
        border_style="cyan",
    ))

    try:
        from src.api.server import run_server
        await run_server(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
