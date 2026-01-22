"""
Paper Trading Engine - виртуальная торговля в реальном времени
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import json
import sqlite3
from pathlib import Path
from loguru import logger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.binance_client import BinanceDataLoader, BinanceWebSocket
from bots.strategies import BaseStrategy, get_strategy


@dataclass
class PaperPosition:
    """Виртуальная позиция"""
    id: int
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    quantity: float
    size_usd: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    # Trailing stop fields
    initial_stop_loss: Optional[float] = None
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    break_even_triggered: bool = False

    def update_price(self, price: float):
        self.current_price = price
        if self.side == 'LONG':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            # Track highest price for trailing stop
            if self.highest_price is None or price > self.highest_price:
                self.highest_price = price
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            # Track lowest price for trailing stop
            if self.lowest_price is None or price < self.lowest_price:
                self.lowest_price = price
        self.unrealized_pnl_pct = self.unrealized_pnl / self.size_usd
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'size_usd': self.size_usd,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'current_price': self.current_price,
            'unrealized_pnl': round(self.unrealized_pnl, 2),
            'unrealized_pnl_pct': round(self.unrealized_pnl_pct * 100, 2)
        }


@dataclass 
class PaperTrade:
    """Закрытая виртуальная сделка"""
    id: int
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    size_usd: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    fees: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': round(self.pnl, 2),
            'pnl_pct': round(self.pnl_pct * 100, 2),
            'exit_reason': self.exit_reason
        }


class PaperTradingEngine:
    """
    Paper Trading Engine для виртуальной торговли
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        db_path: str = 'data/paper_trading.db'
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.trade_counter = 0
        
        self.strategies: Dict[str, BaseStrategy] = {}
        self.price_data: Dict[str, float] = {}
        
        self.is_running = False
        self.callbacks: List[Callable] = []
        
        self._init_database()
        self._load_state()
        
        logger.info(f"PaperTradingEngine initialized with ${initial_capital:,.2f}")
    
    def _init_database(self):
        """Инициализация базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # State
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Positions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                entry_time TEXT,
                entry_price REAL,
                quantity REAL,
                size_usd REAL,
                stop_loss REAL,
                take_profit REAL
            )
        ''')
        
        # Trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                size_usd REAL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                fees REAL
            )
        ''')
        
        # Equity history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equity REAL,
                capital REAL,
                unrealized_pnl REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Загрузка состояния из БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load capital
        cursor.execute("SELECT value FROM state WHERE key = 'capital'")
        row = cursor.fetchone()
        if row:
            self.capital = float(row[0])
        
        # Load trade counter
        cursor.execute("SELECT value FROM state WHERE key = 'trade_counter'")
        row = cursor.fetchone()
        if row:
            self.trade_counter = int(row[0])
        
        # Load positions
        cursor.execute("SELECT * FROM positions")
        for row in cursor.fetchall():
            pos = PaperPosition(
                id=row[0],
                symbol=row[1],
                side=row[2],
                entry_time=datetime.fromisoformat(row[3]),
                entry_price=row[4],
                quantity=row[5],
                size_usd=row[6],
                stop_loss=row[7],
                take_profit=row[8]
            )
            self.positions[pos.symbol] = pos
        
        # Load trades
        cursor.execute("SELECT * FROM trades ORDER BY exit_time DESC LIMIT 100")
        for row in cursor.fetchall():
            trade = PaperTrade(
                id=row[0],
                symbol=row[1],
                side=row[2],
                entry_time=datetime.fromisoformat(row[3]),
                exit_time=datetime.fromisoformat(row[4]),
                entry_price=row[5],
                exit_price=row[6],
                quantity=row[7],
                size_usd=row[8],
                pnl=row[9],
                pnl_pct=row[10],
                exit_reason=row[11],
                fees=row[12]
            )
            self.trades.append(trade)
        
        conn.close()
        logger.info(f"Loaded state: capital=${self.capital:,.2f}, {len(self.positions)} positions, {len(self.trades)} trades")
    
    def _save_state(self):
        """Сохранение состояния в БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save capital
        cursor.execute("INSERT OR REPLACE INTO state (key, value) VALUES ('capital', ?)", (str(self.capital),))
        cursor.execute("INSERT OR REPLACE INTO state (key, value) VALUES ('trade_counter', ?)", (str(self.trade_counter),))
        
        conn.commit()
        conn.close()
    
    def _save_position(self, position: PaperPosition):
        """Сохранение позиции"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO positions 
            (id, symbol, side, entry_time, entry_price, quantity, size_usd, stop_loss, take_profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position.id, position.symbol, position.side,
            position.entry_time.isoformat(), position.entry_price,
            position.quantity, position.size_usd,
            position.stop_loss, position.take_profit
        ))
        
        conn.commit()
        conn.close()
    
    def _delete_position(self, symbol: str):
        """Удаление позиции"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()
    
    def _save_trade(self, trade: PaperTrade):
        """Сохранение сделки"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (id, symbol, side, entry_time, exit_time, entry_price, exit_price, 
             quantity, size_usd, pnl, pnl_pct, exit_reason, fees)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.id, trade.symbol, trade.side,
            trade.entry_time.isoformat(), trade.exit_time.isoformat(),
            trade.entry_price, trade.exit_price,
            trade.quantity, trade.size_usd,
            trade.pnl, trade.pnl_pct, trade.exit_reason, trade.fees
        ))
        
        conn.commit()
        conn.close()
    
    def _save_equity(self):
        """Сохранение equity snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        equity = self.capital + unrealized
        
        cursor.execute('''
            INSERT INTO equity_history (timestamp, equity, capital, unrealized_pnl)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), equity, self.capital, unrealized))
        
        conn.commit()
        conn.close()
    
    def add_strategy(self, symbol: str, strategy: BaseStrategy):
        """Добавление стратегии для символа"""
        self.strategies[symbol] = strategy
        logger.info(f"Added strategy {strategy.name} for {symbol}")
    
    def update_price(self, symbol: str, price: float):
        """Обновление цены"""
        self.price_data[symbol] = price
        
        if symbol in self.positions:
            position = self.positions[symbol]
            position.update_price(price)
            
            # Check stop-loss / take-profit
            self._check_exit_conditions(symbol, price)
        
        # Notify callbacks
        for callback in self.callbacks:
            callback('price_update', {'symbol': symbol, 'price': price})
    
    def _check_exit_conditions(self, symbol: str, price: float):
        """Проверка условий выхода с trailing stop и break-even"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate initial risk (1R)
        if position.initial_stop_loss and position.entry_price:
            risk_1r = abs(position.entry_price - position.initial_stop_loss)
        else:
            risk_1r = 0

        if position.side == 'LONG':
            # Break-even: move SL to entry when price reaches +1R
            if risk_1r > 0 and not position.break_even_triggered:
                if price >= position.entry_price + risk_1r:
                    position.stop_loss = position.entry_price
                    position.break_even_triggered = True
                    logger.info(f"{symbol}: Break-even triggered, SL moved to {position.stop_loss:.4f}")

            # Trailing stop: after break-even, trail at 1.5R behind highest
            if position.break_even_triggered and position.highest_price:
                trailing_sl = position.highest_price - (risk_1r * 1.5)
                if trailing_sl > position.stop_loss:
                    position.stop_loss = trailing_sl
                    logger.debug(f"{symbol}: Trailing SL updated to {position.stop_loss:.4f}")

            # Check exit conditions
            if position.stop_loss and price <= position.stop_loss:
                reason = "TRAILING_STOP" if position.break_even_triggered else "STOP_LOSS"
                self.close_position(symbol, price, reason)
            elif position.take_profit and price >= position.take_profit:
                self.close_position(symbol, price, "TAKE_PROFIT")
        else:
            # SHORT position
            # Break-even: move SL to entry when price drops -1R
            if risk_1r > 0 and not position.break_even_triggered:
                if price <= position.entry_price - risk_1r:
                    position.stop_loss = position.entry_price
                    position.break_even_triggered = True
                    logger.info(f"{symbol}: Break-even triggered, SL moved to {position.stop_loss:.4f}")

            # Trailing stop: after break-even, trail at 1.5R above lowest
            if position.break_even_triggered and position.lowest_price:
                trailing_sl = position.lowest_price + (risk_1r * 1.5)
                if trailing_sl < position.stop_loss:
                    position.stop_loss = trailing_sl
                    logger.debug(f"{symbol}: Trailing SL updated to {position.stop_loss:.4f}")

            # Check exit conditions
            if position.stop_loss and price >= position.stop_loss:
                reason = "TRAILING_STOP" if position.break_even_triggered else "STOP_LOSS"
                self.close_position(symbol, price, reason)
            elif position.take_profit and price <= position.take_profit:
                self.close_position(symbol, price, "TAKE_PROFIT")
    
    def open_position(
        self,
        symbol: str,
        side: str,
        size_usd: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[PaperPosition]:
        """Открытие позиции"""
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return None
        
        price = self.price_data.get(symbol)
        if not price:
            logger.error(f"No price data for {symbol}")
            return None
        
        # Default size: 2% of capital
        if not size_usd:
            size_usd = self.capital * 0.02
        
        # Check if we have enough capital
        if size_usd > self.capital:
            logger.warning(f"Insufficient capital for {symbol}")
            return None
        
        # Calculate quantity
        fee = size_usd * self.fee_rate
        quantity = (size_usd - fee) / price
        
        # Create position with trailing stop support
        self.trade_counter += 1
        position = PaperPosition(
            id=self.trade_counter,
            symbol=symbol,
            side=side,
            entry_time=datetime.now(),
            entry_price=price,
            quantity=quantity,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=price,
            initial_stop_loss=stop_loss,  # Save for trailing stop calculation
            highest_price=price if side == 'LONG' else None,
            lowest_price=price if side == 'SHORT' else None,
            break_even_triggered=False
        )
        
        self.positions[symbol] = position
        self._save_position(position)
        self._save_state()
        
        logger.info(f"Opened {side} position: {symbol} @ {price:.4f}, size=${size_usd:.2f}")
        
        for callback in self.callbacks:
            callback('position_opened', position.to_dict())
        
        return position
    
    def close_position(
        self,
        symbol: str,
        price: Optional[float] = None,
        reason: str = "MANUAL"
    ) -> Optional[PaperTrade]:
        """Закрытие позиции"""
        if symbol not in self.positions:
            logger.warning(f"No position for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        if not price:
            price = self.price_data.get(symbol, position.current_price)
        
        # Calculate P&L
        if position.side == 'LONG':
            pnl = (price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - price) * position.quantity
        
        # Fees
        exit_fee = position.quantity * price * self.fee_rate
        entry_fee = position.size_usd * self.fee_rate
        total_fees = entry_fee + exit_fee
        
        net_pnl = pnl - total_fees
        pnl_pct = net_pnl / position.size_usd
        
        # Create trade record
        trade = PaperTrade(
            id=position.id,
            symbol=symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            size_usd=position.size_usd,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            fees=total_fees
        )
        
        # Update capital
        self.capital += net_pnl
        
        # Save and remove
        self._save_trade(trade)
        self._delete_position(symbol)
        self._save_state()
        
        del self.positions[symbol]
        self.trades.insert(0, trade)
        
        logger.info(f"Closed {position.side} position: {symbol} @ {price:.4f}, "
                   f"PnL: ${net_pnl:.2f} ({pnl_pct:.2%}), Reason: {reason}")
        
        for callback in self.callbacks:
            callback('position_closed', trade.to_dict())
        
        return trade
    
    def process_signal(self, symbol: str, signal: Dict):
        """Обработка сигнала от стратегии"""
        action = signal.get('action')
        
        if action == 'HOLD':
            return
        
        # Close opposite position
        if symbol in self.positions:
            position = self.positions[symbol]
            if (action == 'BUY' and position.side == 'SHORT') or \
               (action == 'SELL' and position.side == 'LONG'):
                self.close_position(symbol, reason="SIGNAL_REVERSE")
        
        # Open new position
        if symbol not in self.positions:
            side = 'LONG' if action == 'BUY' else 'SHORT'
            position_size = signal.get('position_size', 0.02)
            
            self.open_position(
                symbol=symbol,
                side=side,
                size_usd=self.capital * position_size,
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit')
            )
    
    def get_status(self) -> Dict:
        """Получить текущий статус"""
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        equity = self.capital + unrealized_pnl
        
        # Statistics
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            win_rate = len(wins) / len(self.trades)
            total_pnl = sum(t.pnl for t in self.trades)
        else:
            win_rate = 0
            total_pnl = 0
        
        return {
            'capital': round(self.capital, 2),
            'equity': round(equity, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'total_return_pct': round((equity - self.initial_capital) / self.initial_capital * 100, 2),
            'positions_count': len(self.positions),
            'trades_count': len(self.trades),
            'win_rate': round(win_rate * 100, 2),
            'total_realized_pnl': round(total_pnl, 2),
            'positions': [p.to_dict() for p in self.positions.values()],
            'recent_trades': [t.to_dict() for t in self.trades[:10]]
        }
    
    def get_equity_history(self, days: int = 30) -> List[Dict]:
        """Получить историю equity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute('''
            SELECT timestamp, equity, capital, unrealized_pnl 
            FROM equity_history 
            WHERE timestamp > ?
            ORDER BY timestamp
        ''', (since,))
        
        history = [
            {
                'timestamp': row[0],
                'equity': row[1],
                'capital': row[2],
                'unrealized_pnl': row[3]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return history
    
    def reset(self):
        """Сброс к начальному состоянию"""
        # Close all positions
        for symbol in list(self.positions.keys()):
            self.close_position(symbol, reason="RESET")
        
        # Reset capital
        self.capital = self.initial_capital
        self.trade_counter = 0
        self.trades = []
        
        # Clear database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions")
        cursor.execute("DELETE FROM trades")
        cursor.execute("DELETE FROM equity_history")
        cursor.execute("DELETE FROM state")
        conn.commit()
        conn.close()
        
        self._save_state()
        logger.info("Paper trading engine reset")
    
    def add_callback(self, callback: Callable):
        """Добавить callback для событий"""
        self.callbacks.append(callback)
    
    def deposit(self, amount: float):
        """Пополнение баланса"""
        self.capital += amount
        self._save_state()
        logger.info(f"Deposited ${amount:.2f}. New capital: ${self.capital:.2f}")
    
    def withdraw(self, amount: float) -> bool:
        """Вывод средств"""
        if amount > self.capital:
            logger.warning("Insufficient funds for withdrawal")
            return False
        
        self.capital -= amount
        self._save_state()
        logger.info(f"Withdrawn ${amount:.2f}. New capital: ${self.capital:.2f}")
        return True
