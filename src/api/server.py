"""FastAPI server for the trading dashboard."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..database import get_database
from ..trading.points_strategy import PointsFarmingStrategy, TradePair, TradeStatus
from ..utils.logger import get_logger

logger = get_logger("ethereal.api")

# Global state
_strategy: Optional[PointsFarmingStrategy] = None
_strategy_task: Optional[asyncio.Task] = None
_websocket_clients: List[WebSocket] = []


class BotConfig(BaseModel):
    """Bot configuration model."""
    trading_pairs: Optional[str] = None
    position_size: Optional[float] = None
    leverage: Optional[int] = None
    stop_loss_percent: Optional[float] = None
    take_profit_percent: Optional[float] = None
    min_hold_time_minutes: Optional[int] = None
    max_hold_time_minutes: Optional[int] = None
    max_trades: Optional[int] = None


class OrderRequest(BaseModel):
    """Manual order request."""
    ticker: str
    side: str  # "buy" or "sell"
    size: float
    account: int = 1  # 1 or 2


class SettingsUpdate(BaseModel):
    """Settings update request."""
    trading_pairs: Optional[str] = None
    position_size: Optional[float] = None
    use_full_balance: Optional[bool] = None
    min_balance_threshold: Optional[float] = None
    leverage: Optional[int] = None
    max_concurrent_trades: Optional[int] = None
    market_max_leverage: Optional[str] = None
    stop_loss_percent: Optional[float] = None
    take_profit_percent: Optional[float] = None
    min_hold_time_minutes: Optional[int] = None
    max_hold_time_minutes: Optional[int] = None
    min_trade_delay_seconds: Optional[int] = None
    max_trade_delay_seconds: Optional[int] = None
    max_daily_trades: Optional[int] = None


# Log buffer for live logs
_log_buffer: List[Dict[str, Any]] = []
MAX_LOG_BUFFER = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting API server...")
    db = await get_database()
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    global _strategy, _strategy_task
    if _strategy:
        await _strategy.shutdown()
    if _strategy_task:
        _strategy_task.cancel()
    await db.close()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Ethereal Trading Dashboard",
        description="Control panel for the Ethereal points farming bot",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Static files
    static_path = Path(__file__).parent.parent.parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # Register routes
    register_routes(app)
    
    return app


def register_routes(app: FastAPI):
    """Register all API routes."""
    
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the dashboard."""
        index_path = Path(__file__).parent.parent.parent / "static" / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse("<h1>Dashboard not found. Run setup first.</h1>")
    
    # ============ Bot Control ============
    
    @app.get("/api/status")
    async def get_status():
        """Get bot status and statistics."""
        global _strategy
        db = await get_database()
        settings = get_settings()
        
        stats = await db.get_total_stats()
        volume = await db.get_volume_summary()
        
        bot_connected = _strategy is not None and _strategy.account1 is not None and _strategy.account1.is_connected
        bot_trading = _strategy is not None and _strategy.running
        
        status = {
            "bot_running": bot_trading,  # Legacy compatibility
            "bot_connected": bot_connected,
            "bot_trading": bot_trading,
            "settings": {
                "trading_pairs": settings.trading_pairs,
                "position_size": settings.position_size,
                "leverage": settings.leverage,
                "max_concurrent_trades": getattr(settings, 'max_concurrent_trades', 2),
                "market_max_leverage": getattr(settings, 'market_max_leverage', 'BTCUSD:20,ETHUSD:20'),
                "stop_loss_percent": settings.stop_loss_percent * 100,
                "take_profit_percent": settings.take_profit_percent * 100,
                "min_hold_time_minutes": settings.min_hold_time_minutes,
                "max_hold_time_minutes": settings.max_hold_time_minutes,
                "max_daily_trades": settings.max_daily_trades,
            },
            "stats": {
                "total_trades": stats.get("total_trades", 0) or 0,
                "open_trades": stats.get("open_trades", 0) or 0,
                "closed_trades": stats.get("closed_trades", 0) or 0,
                "total_pnl": float(stats.get("total_pnl", 0) or 0),
            },
            "volume": {
                "total_buy_volume": float(volume.get("total_buy_volume", 0) or 0),
                "total_sell_volume": float(volume.get("total_sell_volume", 0) or 0),
                "total_volume": float(volume.get("total_volume", 0) or 0),
            },
            "accounts": {},
        }
        
        # Get account balances if strategy is running
        if _strategy and _strategy.account1 and _strategy.account2:
            try:
                bal1 = await _strategy.account1.get_balance()
                bal2 = await _strategy.account2.get_balance()
                status["accounts"] = {
                    "account1_balance": bal1,
                    "account2_balance": bal2,
                }
            except:
                pass
        
        return status
    
    @app.post("/api/bot/connect")
    async def connect_bot():
        """Connect to accounts without starting trading."""
        global _strategy
        
        if _strategy and _strategy.account1 and _strategy.account1.is_connected:
            return {"success": True, "message": "Already connected"}
        
        try:
            settings = get_settings()
            _strategy = PointsFarmingStrategy(settings)
            
            # Initialize (connects to accounts)
            if not await _strategy.initialize():
                return {"success": False, "message": "Failed to initialize strategy"}
            
            # Set up trade callbacks
            _strategy.on_trade_open(on_trade_open)
            _strategy.on_trade_close(on_trade_close)
            
            logger.info("Bot connected via API (not trading yet)")
            await broadcast({"type": "bot_status", "connected": True, "trading": False})
            
            return {"success": True, "message": "Bot connected"}
        except Exception as e:
            logger.error(f"Failed to connect bot: {e}")
            return {"success": False, "message": str(e)}

    @app.post("/api/bot/start-trading")
    async def start_trading():
        """Start trading (bot must be connected first)."""
        global _strategy, _strategy_task
        
        if not _strategy or not _strategy.account1 or not _strategy.account1.is_connected:
            return {"success": False, "message": "Bot not connected. Click Connect first."}
        
        if _strategy.running:
            return {"success": False, "message": "Already trading"}
        
        try:
            # Start trading in background
            _strategy_task = asyncio.create_task(_strategy.start())
            
            logger.info("Trading started via API")
            await broadcast({"type": "bot_status", "connected": True, "trading": True})
            
            return {"success": True, "message": "Trading started"}
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            return {"success": False, "message": str(e)}

    @app.post("/api/bot/start")
    async def start_bot():
        """Start the trading bot (legacy - connects and starts trading)."""
        global _strategy, _strategy_task
        
        if _strategy and _strategy.running:
            return {"success": False, "message": "Bot is already running"}
        
        try:
            settings = get_settings()
            _strategy = PointsFarmingStrategy(settings)
            
            # Initialize
            if not await _strategy.initialize():
                return {"success": False, "message": "Failed to initialize strategy"}
            
            # Set up trade callbacks
            _strategy.on_trade_open(on_trade_open)
            _strategy.on_trade_close(on_trade_close)
            
            # Start in background
            _strategy_task = asyncio.create_task(_strategy.start())
            
            logger.info("Bot started via API")
            await broadcast({"type": "bot_status", "connected": True, "trading": True})
            
            return {"success": True, "message": "Bot started"}
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            return {"success": False, "message": str(e)}
    
    @app.post("/api/bot/stop")
    async def stop_bot():
        """Stop the trading bot and disconnect."""
        global _strategy, _strategy_task
        
        if not _strategy:
            return {"success": False, "message": "Bot is not running"}
        
        try:
            # Stop trading if running
            if _strategy.running:
                await _strategy.stop()
            if _strategy_task:
                _strategy_task.cancel()
                _strategy_task = None
            
            # Disconnect accounts
            await _strategy.shutdown()
            _strategy = None
            
            logger.info("Bot stopped and disconnected via API")
            await broadcast({"type": "bot_status", "connected": False, "trading": False})
            
            return {"success": True, "message": "Bot stopped"}
        except Exception as e:
            logger.error(f"Failed to stop bot: {e}")
            return {"success": False, "message": str(e)}
    
    @app.post("/api/bot/close-all")
    async def close_all_positions():
        """Close all open positions."""
        global _strategy
        
        if not _strategy:
            return {"success": False, "message": "Bot not initialized"}
        
        try:
            await _strategy.close_all()
            return {"success": True, "message": "All positions closed"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    # ============ Settings ============
    
    @app.get("/api/settings")
    async def get_settings_api():
        """Get current settings."""
        settings = get_settings()
        return {
            "trading_pairs": settings.trading_pairs,
            "position_size": settings.position_size,
            "use_full_balance": getattr(settings, 'use_full_balance', True),
            "min_balance_threshold": getattr(settings, 'min_balance_threshold', 10.0),
            "leverage": settings.leverage,
            "max_concurrent_trades": getattr(settings, 'max_concurrent_trades', 2),
            "market_max_leverage": getattr(settings, 'market_max_leverage', 'BTCUSD:20,ETHUSD:20'),
            "stop_loss_percent": settings.stop_loss_percent * 100,
            "take_profit_percent": settings.take_profit_percent * 100,
            "min_hold_time_minutes": settings.min_hold_time_minutes,
            "max_hold_time_minutes": settings.max_hold_time_minutes,
            "min_trade_delay_seconds": getattr(settings, 'min_trade_delay_seconds', 60),
            "max_trade_delay_seconds": getattr(settings, 'max_trade_delay_seconds', 300),
            "max_daily_trades": getattr(settings, 'max_daily_trades', 100),
        }
    
    @app.get("/api/products/available")
    async def get_available_products():
        """Get list of all available trading products with their max leverage."""
        global _strategy

        products_info = []

        # Try to get products from strategy's account client
        if _strategy and _strategy.account1 and _strategy.account1._products:
            products_info = _strategy.account1.get_all_products()
        else:
            # Fallback to hardcoded defaults with known leverage limits
            products_info = [
                {"ticker": "BTCUSD", "max_leverage": 20, "lot_size": 0.00001, "tick_size": 1},
                {"ticker": "ETHUSD", "max_leverage": 20, "lot_size": 0.0001, "tick_size": 0.1},
                {"ticker": "SOLUSD", "max_leverage": 10, "lot_size": 0.001, "tick_size": 0.01},
                {"ticker": "HYPEUSD", "max_leverage": 10, "lot_size": 0.01, "tick_size": 0.001},
                {"ticker": "SUIUSD", "max_leverage": 10, "lot_size": 0.1, "tick_size": 0.0001},
                {"ticker": "XRPUSD", "max_leverage": 10, "lot_size": 0.1, "tick_size": 0.0001},
                {"ticker": "AAVEUSD", "max_leverage": 5, "lot_size": 0.001, "tick_size": 0.01},
                {"ticker": "ENAUSD", "max_leverage": 5, "lot_size": 1, "tick_size": 0.00001},
                {"ticker": "FARTCOINUSD", "max_leverage": 5, "lot_size": 1, "tick_size": 0.00001},
                {"ticker": "PUMPUSD", "max_leverage": 5, "lot_size": 100, "tick_size": 0.0000001},
                {"ticker": "ZECUSD", "max_leverage": 5, "lot_size": 0.001, "tick_size": 0.01},
                {"ticker": "MONUSD", "max_leverage": 5, "lot_size": 10, "tick_size": 0.000001},
            ]

        # Sort by max leverage (highest first), then by ticker
        products_info.sort(key=lambda x: (-x.get("max_leverage", 10), x.get("ticker", "")))

        # Get current selected pairs
        settings = get_settings()
        selected_pairs = settings.trading_pairs_list

        # Add selected flag
        for p in products_info:
            p["selected"] = p["ticker"] in selected_pairs

        return {"products": products_info}
    
    @app.get("/api/logs")
    async def get_logs():
        """Get recent log messages."""
        return {"logs": _log_buffer[-50:]}
    
    @app.post("/api/settings")
    async def update_settings(update: SettingsUpdate):
        """Update settings (runtime only, not persisted to .env)."""
        global _strategy
        settings = get_settings()
        
        try:
            # Update settings object
            if update.trading_pairs is not None:
                settings.trading_pairs = update.trading_pairs
            if update.position_size is not None:
                settings.position_size = update.position_size
            if update.use_full_balance is not None:
                settings.use_full_balance = update.use_full_balance
            if update.min_balance_threshold is not None:
                settings.min_balance_threshold = update.min_balance_threshold
            if update.leverage is not None:
                settings.leverage = update.leverage
            if update.max_concurrent_trades is not None:
                settings.max_concurrent_trades = update.max_concurrent_trades
            if update.market_max_leverage is not None:
                settings.market_max_leverage = update.market_max_leverage
            if update.stop_loss_percent is not None:
                settings.stop_loss_percent = update.stop_loss_percent / 100  # Convert from %
            if update.take_profit_percent is not None:
                settings.take_profit_percent = update.take_profit_percent / 100
            if update.min_hold_time_minutes is not None:
                settings.min_hold_time_minutes = update.min_hold_time_minutes
            if update.max_hold_time_minutes is not None:
                settings.max_hold_time_minutes = update.max_hold_time_minutes
            if update.min_trade_delay_seconds is not None:
                settings.min_trade_delay_seconds = update.min_trade_delay_seconds
            if update.max_trade_delay_seconds is not None:
                settings.max_trade_delay_seconds = update.max_trade_delay_seconds
            if update.max_daily_trades is not None:
                settings.max_daily_trades = update.max_daily_trades
            
            # If strategy is running, update its settings reference
            if _strategy:
                _strategy.settings = settings
            
            logger.info(f"Settings updated via API")
            await broadcast({"type": "settings_updated"})
            add_log("INFO", "Settings updated")
            
            return {"success": True, "message": "Settings updated"}
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            return {"success": False, "message": str(e)}


    # ============ Trades ============
    
    @app.get("/api/trades")
    async def get_trades(limit: int = 50, offset: int = 0, status: Optional[str] = None):
        """Get trade history."""
        db = await get_database()
        trades = await db.get_trades(limit=limit, offset=offset, status=status)
        return {"trades": trades}
    
    @app.get("/api/trades/active")
    async def get_active_trades():
        """Get active trades."""
        global _strategy
        
        active = []
        if _strategy:
            for trade_id, trade in _strategy.active_trades.items():
                active.append({
                    "id": trade.id,
                    "product_id": trade.product_id,
                    "size": trade.size,
                    "entry_price": trade.entry_price,
                    "pnl_long": trade.pnl_long,
                    "pnl_short": trade.pnl_short,
                    "total_pnl": trade.total_pnl,
                    "hold_time_minutes": trade.hold_time_minutes,
                    "target_hold_minutes": trade.target_hold_minutes,
                    "account1_is_long": trade.account1_is_long,
                    "opened_at": trade.opened_at.isoformat(),
                })
        
        return {"trades": active}
    
    # ============ Positions ============
    
    @app.get("/api/positions")
    async def get_positions():
        """Get current positions for both accounts with detailed info."""
        global _strategy
        settings = get_settings()

        positions = {
            "account1": [],
            "account2": [],
            "account1_address": "",
            "account2_address": "",
            "leverage": settings.leverage,
        }

        if _strategy and _strategy.account1 and _strategy.account2:
            try:
                # Get wallet addresses (shortened)
                addr1 = _strategy.account1.wallet_address
                addr2 = _strategy.account2.wallet_address
                positions["account1_address"] = f"{addr1[:6]}...{addr1[-4:]}" if addr1 else ""
                positions["account2_address"] = f"{addr2[:6]}...{addr2[-4:]}" if addr2 else ""

                pos1 = await _strategy.account1.list_positions()
                pos2 = await _strategy.account2.list_positions()

                # Positions already have all calculated fields
                for p in pos1:
                    ticker = p.get("ticker", "")
                    # Get per-pair leverage from settings, fallback to market max
                    pair_lev = settings.market_leverage_limits.get(ticker.upper(), 0)
                    if pair_lev == 0:
                        pair_lev = _strategy.account1.get_max_leverage(ticker) if _strategy.account1 else 10
                    positions["account1"].append({
                        "ticker": ticker,
                        "size": float(p.get("size", 0)),
                        "entry_price": float(p.get("entry_price", 0)),
                        "mark_price": float(p.get("mark_price", 0)),
                        "unrealized_pnl": float(p.get("unrealized_pnl", 0)),
                        "side": p.get("side", "LONG"),
                        "notional": float(p.get("notional", 0)),
                        "leverage": pair_lev,
                    })

                for p in pos2:
                    ticker = p.get("ticker", "")
                    # Get per-pair leverage from settings, fallback to market max
                    pair_lev = settings.market_leverage_limits.get(ticker.upper(), 0)
                    if pair_lev == 0:
                        pair_lev = _strategy.account2.get_max_leverage(ticker) if _strategy.account2 else 10
                    positions["account2"].append({
                        "ticker": ticker,
                        "size": float(p.get("size", 0)),
                        "entry_price": float(p.get("entry_price", 0)),
                        "mark_price": float(p.get("mark_price", 0)),
                        "unrealized_pnl": float(p.get("unrealized_pnl", 0)),
                        "side": p.get("side", "LONG"),
                        "notional": float(p.get("notional", 0)),
                        "leverage": pair_lev,
                    })
            except Exception as e:
                logger.error(f"Failed to get positions: {e}")

        return positions
    
    # ============ Volume Stats ============
    
    @app.get("/api/volume")
    async def get_volume_stats(days: int = 30):
        """Get volume statistics."""
        db = await get_database()
        stats = await db.get_volume_stats(days=days)
        summary = await db.get_volume_summary()
        
        return {
            "daily": stats,
            "summary": {
                "total_buy_volume": float(summary.get("total_buy_volume", 0) or 0),
                "total_sell_volume": float(summary.get("total_sell_volume", 0) or 0),
                "total_volume": float(summary.get("total_volume", 0) or 0),
                "total_trades": int(summary.get("total_trades", 0) or 0),
                "total_pnl": float(summary.get("total_pnl", 0) or 0),
            }
        }
    
    # ============ Market Data ============
    
    @app.get("/api/prices")
    async def get_prices():
        """Get current prices for trading pairs."""
        global _strategy
        
        prices = {}
        settings = get_settings()
        
        if _strategy and _strategy.account1:
            for ticker in settings.trading_pairs_list:
                try:
                    price = await _strategy.account1.get_mark_price(ticker)
                    prices[ticker] = price
                except:
                    prices[ticker] = 0
        
        return {"prices": prices}
    
    @app.get("/api/products")
    async def get_products():
        """Get available products."""
        global _strategy
        
        if _strategy and _strategy.account1:
            return {"products": list(_strategy.account1._products.keys())}
        
        return {"products": []}
    
    # ============ WebSocket ============
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates."""
        await websocket.accept()
        _websocket_clients.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(_websocket_clients)}")
        
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
        except WebSocketDisconnect:
            _websocket_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(_websocket_clients)}")


def add_log(level: str, message: str):
    """Add a log message to the buffer and broadcast."""
    global _log_buffer
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
    }
    _log_buffer.append(log_entry)
    if len(_log_buffer) > MAX_LOG_BUFFER:
        _log_buffer = _log_buffer[-MAX_LOG_BUFFER:]
    
    # Broadcast to WebSocket clients
    asyncio.create_task(broadcast({"type": "log", "log": log_entry}))


async def broadcast(message: Dict[str, Any]):
    """Broadcast message to all WebSocket clients."""
    if not _websocket_clients:
        return
    
    for client in _websocket_clients[:]:
        try:
            await client.send_json(message)
        except:
            try:
                _websocket_clients.remove(client)
            except:
                pass


def on_trade_open(trade: TradePair):
    """Callback when trade opens."""
    asyncio.create_task(_handle_trade_open(trade))


async def _handle_trade_open(trade: TradePair):
    """Handle trade open event."""
    db = await get_database()
    
    trade_data = {
        "id": trade.id,
        "product_id": trade.product_id,
        "size": trade.size,
        "entry_price": trade.entry_price,
        "status": trade.status.value,
        "account1_is_long": trade.account1_is_long,
        "opened_at": trade.opened_at.isoformat(),
    }
    
    await db.save_trade(trade_data)
    await broadcast({
        "type": "trade_open",
        "trade": trade_data,
    })
    
    direction = "Acc1=LONG" if trade.account1_is_long else "Acc1=SHORT"
    add_log("INFO", f"Trade opened: {trade.product_id} @ ${trade.entry_price:.2f} ({direction})")


def on_trade_close(trade: TradePair):
    """Callback when trade closes."""
    asyncio.create_task(_handle_trade_close(trade))


async def _handle_trade_close(trade: TradePair):
    """Handle trade close event."""
    db = await get_database()
    
    trade_data = {
        "id": trade.id,
        "product_id": trade.product_id,
        "size": trade.size,
        "entry_price": trade.entry_price,
        "exit_price": trade.entry_price,  # Would need actual exit price
        "pnl_long": trade.pnl_long,
        "pnl_short": trade.pnl_short,
        "total_pnl": trade.total_pnl,
        "status": trade.status.value,
        "close_reason": trade.close_reason,
        "account1_is_long": trade.account1_is_long,
        "opened_at": trade.opened_at.isoformat(),
        "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
    }
    
    await db.save_trade(trade_data)
    await broadcast({
        "type": "trade_close",
        "trade": trade_data,
    })
    
    pnl_str = f"+${trade.total_pnl:.2f}" if trade.total_pnl >= 0 else f"-${abs(trade.total_pnl):.2f}"
    add_log("INFO", f"Trade closed: {trade.product_id} | PnL: {pnl_str} | Reason: {trade.close_reason}")


async def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the API server."""
    import uvicorn
    
    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
