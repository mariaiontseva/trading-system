"""
Trade Logger - logs all trades to SQLite for analysis
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from loguru import logger

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trades.db')


@dataclass
class TradeLog:
    """Single trade record"""
    id: Optional[int] = None
    symbol: str = ''
    side: str = ''  # LONG or SHORT

    # Signal info
    signal_time: str = ''
    signal_price: float = 0.0
    signal_rsi: float = 0.0
    signal_adx: float = 0.0

    # Entry info
    entry_time: str = ''
    entry_price: float = 0.0
    entry_slippage: float = 0.0  # % difference from signal price

    # Position info
    position_size_usd: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Exit info
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP, SL, TRAILING, SIGNAL, MANUAL
    exit_slippage: Optional[float] = None

    # Results
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_hours: Optional[float] = None

    # Source
    source: str = 'testnet'  # testnet, paper, backtest

    # DB metadata
    created_at: Optional[str] = None


class TradeLogger:
    """SQLite-based trade logger"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if not exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,

                signal_time TEXT,
                signal_price REAL,
                signal_rsi REAL,
                signal_adx REAL,

                entry_time TEXT,
                entry_price REAL,
                entry_slippage REAL,

                position_size_usd REAL,
                stop_loss REAL,
                take_profit REAL,

                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                exit_slippage REAL,

                pnl_usd REAL,
                pnl_pct REAL,
                duration_hours REAL,

                source TEXT DEFAULT 'testnet',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Index for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_source ON trades(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)')

        conn.commit()
        conn.close()
        logger.info(f"Trade logger initialized: {self.db_path}")

    def log_entry(self, trade: TradeLog) -> int:
        """Log trade entry, return trade ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (
                symbol, side, signal_time, signal_price, signal_rsi, signal_adx,
                entry_time, entry_price, entry_slippage,
                position_size_usd, stop_loss, take_profit, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.symbol, trade.side, trade.signal_time, trade.signal_price,
            trade.signal_rsi, trade.signal_adx, trade.entry_time, trade.entry_price,
            trade.entry_slippage, trade.position_size_usd, trade.stop_loss,
            trade.take_profit, trade.source
        ))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Trade #{trade_id} ENTRY logged: {trade.side} {trade.symbol} @ {trade.entry_price}")
        return trade_id

    def log_exit(self, trade_id: int, exit_time: str, exit_price: float,
                 exit_reason: str, exit_slippage: float = 0.0) -> Optional[TradeLog]:
        """Log trade exit, calculate P&L"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get entry data
        cursor.execute('SELECT * FROM trades WHERE id = ?', (trade_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        columns = [desc[0] for desc in cursor.description]
        trade_data = dict(zip(columns, row))

        entry_price = trade_data['entry_price']
        side = trade_data['side']
        position_size = trade_data['position_size_usd']
        entry_time = trade_data['entry_time']

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # SHORT
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        pnl_usd = position_size * pnl_pct / 100

        # Calculate duration
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            exit_dt = datetime.fromisoformat(exit_time)
            duration_hours = (exit_dt - entry_dt).total_seconds() / 3600
        except:
            duration_hours = 0

        # Update record
        cursor.execute('''
            UPDATE trades SET
                exit_time = ?,
                exit_price = ?,
                exit_reason = ?,
                exit_slippage = ?,
                pnl_usd = ?,
                pnl_pct = ?,
                duration_hours = ?
            WHERE id = ?
        ''', (exit_time, exit_price, exit_reason, exit_slippage,
              pnl_usd, pnl_pct, duration_hours, trade_id))

        conn.commit()
        conn.close()

        logger.info(f"Trade #{trade_id} EXIT logged: {exit_reason} @ {exit_price}, P&L: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")

        return self.get_trade(trade_id)

    def get_trade(self, trade_id: int) -> Optional[TradeLog]:
        """Get single trade by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM trades WHERE id = ?', (trade_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return TradeLog(**dict(row))

    def get_open_trades(self, source: str = None) -> List[TradeLog]:
        """Get all open trades (no exit)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if source:
            cursor.execute('SELECT * FROM trades WHERE exit_time IS NULL AND source = ? ORDER BY entry_time DESC', (source,))
        else:
            cursor.execute('SELECT * FROM trades WHERE exit_time IS NULL ORDER BY entry_time DESC')

        rows = cursor.fetchall()
        conn.close()

        return [TradeLog(**dict(row)) for row in rows]

    def get_recent_trades(self, limit: int = 50, source: str = None) -> List[TradeLog]:
        """Get recent closed trades"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if source:
            cursor.execute('''
                SELECT * FROM trades
                WHERE exit_time IS NOT NULL AND source = ?
                ORDER BY exit_time DESC LIMIT ?
            ''', (source, limit))
        else:
            cursor.execute('''
                SELECT * FROM trades
                WHERE exit_time IS NOT NULL
                ORDER BY exit_time DESC LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [TradeLog(**dict(row)) for row in rows]

    def get_today_trades(self, source: str = None) -> List[TradeLog]:
        """Get today's trades"""
        today = datetime.now().strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if source:
            cursor.execute('''
                SELECT * FROM trades
                WHERE entry_time LIKE ? AND source = ?
                ORDER BY entry_time DESC
            ''', (f'{today}%', source))
        else:
            cursor.execute('''
                SELECT * FROM trades
                WHERE entry_time LIKE ?
                ORDER BY entry_time DESC
            ''', (f'{today}%',))

        rows = cursor.fetchall()
        conn.close()

        return [TradeLog(**dict(row)) for row in rows]

    def get_statistics(self, source: str = None, days: int = 30) -> Dict:
        """Get trading statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Base query
        where_clause = "WHERE exit_time IS NOT NULL"
        params = []

        if source:
            where_clause += " AND source = ?"
            params.append(source)

        if days:
            where_clause += " AND entry_time >= date('now', ?)"
            params.append(f'-{days} days')

        # Total trades
        cursor.execute(f'SELECT COUNT(*) FROM trades {where_clause}', params)
        total_trades = cursor.fetchone()[0]

        if total_trades == 0:
            conn.close()
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_duration': 0,
                'avg_slippage': 0
            }

        # Win rate
        cursor.execute(f'SELECT COUNT(*) FROM trades {where_clause} AND pnl_usd > 0', params)
        wins = cursor.fetchone()[0]

        # P&L
        cursor.execute(f'SELECT SUM(pnl_usd), AVG(pnl_usd) FROM trades {where_clause}', params)
        row = cursor.fetchone()
        total_pnl = row[0] or 0
        avg_pnl = row[1] or 0

        # Profit Factor
        cursor.execute(f'SELECT SUM(pnl_usd) FROM trades {where_clause} AND pnl_usd > 0', params)
        total_profit = cursor.fetchone()[0] or 0

        cursor.execute(f'SELECT SUM(ABS(pnl_usd)) FROM trades {where_clause} AND pnl_usd < 0', params)
        total_loss = cursor.fetchone()[0] or 0.001

        # Averages
        cursor.execute(f'SELECT AVG(duration_hours), AVG(entry_slippage) FROM trades {where_clause}', params)
        row = cursor.fetchone()
        avg_duration = row[0] or 0
        avg_slippage = row[1] or 0

        conn.close()

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': wins / total_trades * 100 if total_trades > 0 else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_duration': avg_duration,
            'avg_slippage': avg_slippage
        }

    def compare_with_backtest(self, backtest_stats: Dict) -> Dict:
        """Compare live results with backtest"""
        live_stats = self.get_statistics(source='testnet')

        return {
            'backtest': backtest_stats,
            'live': live_stats,
            'comparison': {
                'win_rate_diff': live_stats['win_rate'] - backtest_stats.get('win_rate', 0),
                'pf_diff': live_stats['profit_factor'] - backtest_stats.get('profit_factor', 0),
                'avg_pnl_diff': live_stats['avg_pnl'] - backtest_stats.get('avg_pnl', 0),
            }
        }


# Global instance
trade_logger = TradeLogger()
