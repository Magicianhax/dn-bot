"""SQLite database for trade history and statistics."""

import aiosqlite
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent.parent.parent / "data" / "trades.db"

_db_instance: Optional["TradeDatabase"] = None


async def get_database() -> "TradeDatabase":
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradeDatabase()
        await _db_instance.initialize()
    return _db_instance


class TradeDatabase:
    """Database for storing trade history and statistics."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize database and create tables."""
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl_long REAL DEFAULT 0,
                pnl_short REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                status TEXT NOT NULL,
                close_reason TEXT,
                account1_is_long INTEGER DEFAULT 1,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS volume_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                buy_volume REAL DEFAULT 0,
                sell_volume REAL DEFAULT 0,
                total_volume REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades(opened_at);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_volume_date ON volume_stats(date);
        """)
        await self._conn.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # Trade operations
    async def save_trade(self, trade: Dict[str, Any]) -> None:
        """Save a trade to database."""
        await self._conn.execute("""
            INSERT OR REPLACE INTO trades 
            (id, product_id, size, entry_price, exit_price, pnl_long, pnl_short, 
             total_pnl, status, close_reason, account1_is_long, opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["id"],
            trade["product_id"],
            trade["size"],
            trade["entry_price"],
            trade.get("exit_price"),
            trade.get("pnl_long", 0),
            trade.get("pnl_short", 0),
            trade.get("total_pnl", 0),
            trade["status"],
            trade.get("close_reason"),
            1 if trade.get("account1_is_long", True) else 0,
            trade["opened_at"],
            trade.get("closed_at"),
        ))
        await self._conn.commit()
        
        # Update volume stats
        await self._update_volume_stats(trade)

    async def _update_volume_stats(self, trade: Dict[str, Any]) -> None:
        """Update daily volume statistics."""
        date = trade["opened_at"][:10]  # YYYY-MM-DD
        notional = trade["size"] * trade["entry_price"]
        
        # Both long and short count as volume
        buy_vol = notional
        sell_vol = notional
        
        await self._conn.execute("""
            INSERT INTO volume_stats (date, buy_volume, sell_volume, total_volume, trade_count, total_pnl)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(date) DO UPDATE SET
                buy_volume = buy_volume + ?,
                sell_volume = sell_volume + ?,
                total_volume = total_volume + ?,
                trade_count = trade_count + 1,
                total_pnl = total_pnl + ?
        """, (
            date, buy_vol, sell_vol, buy_vol + sell_vol, trade.get("total_pnl", 0),
            buy_vol, sell_vol, buy_vol + sell_vol, trade.get("total_pnl", 0)
        ))
        await self._conn.commit()

    async def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a single trade by ID."""
        async with self._conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_trades(
        self, 
        limit: int = 50, 
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trades with pagination."""
        query = "SELECT * FROM trades"
        params = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        query += " ORDER BY opened_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades."""
        return await self.get_trades(status="OPEN")

    # Volume statistics
    async def get_volume_stats(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get volume stats for the last N days."""
        async with self._conn.execute("""
            SELECT * FROM volume_stats 
            ORDER BY date DESC 
            LIMIT ?
        """, (days,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_total_stats(self) -> Dict[str, Any]:
        """Get total statistics."""
        async with self._conn.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN status LIKE 'CLOSED%' THEN 1 ELSE 0 END) as closed_trades,
                SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_trades,
                SUM(total_pnl) as total_pnl,
                SUM(size * entry_price) as total_notional
            FROM trades
        """) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else {}

    async def get_volume_summary(self) -> Dict[str, Any]:
        """Get volume summary."""
        async with self._conn.execute("""
            SELECT 
                SUM(buy_volume) as total_buy_volume,
                SUM(sell_volume) as total_sell_volume,
                SUM(total_volume) as total_volume,
                SUM(trade_count) as total_trades,
                SUM(total_pnl) as total_pnl
            FROM volume_stats
        """) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else {}

    # Bot state
    async def set_state(self, key: str, value: Any) -> None:
        """Set bot state value."""
        await self._conn.execute("""
            INSERT OR REPLACE INTO bot_state (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.now().isoformat()))
        await self._conn.commit()

    async def get_state(self, key: str, default: Any = None) -> Any:
        """Get bot state value."""
        async with self._conn.execute(
            "SELECT value FROM bot_state WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
            return json.loads(row["value"]) if row else default
