"""
Bot 8: Risk Manager
Управление рисками и защита капитала

Правила:
- Максимум 1-2% риска на сделку
- Максимум 5% капитала в одной позиции
- Максимум 20% общего капитала в позициях
- Stop Loss обязателен для каждой сделки
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RiskManager')


class RiskLevel(Enum):
    """Уровни риска"""
    LOW = 0.01      # 1% на сделку
    MEDIUM = 0.015  # 1.5% на сделку
    HIGH = 0.02     # 2% на сделку


@dataclass
class RiskConfig:
    """Конфигурация риск-менеджмента"""
    risk_per_trade: float = 0.01       # 1% риска на сделку
    max_position_size: float = 0.05    # Макс 5% в одной позиции
    max_total_exposure: float = 0.20   # Макс 20% в позициях
    max_daily_loss: float = 0.05       # Макс 5% дневной убыток
    max_drawdown: float = 0.15         # Макс 15% просадка
    min_risk_reward: float = 1.5       # Минимальное R:R соотношение


@dataclass
class PositionSize:
    """Результат расчета размера позиции"""
    symbol: str
    side: str  # 'LONG' или 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float       # В базовой валюте (BTC, ETH, etc.)
    position_value: float      # В USDT
    risk_amount: float         # Сколько рискуем в USDT
    risk_reward_ratio: float   # R:R
    is_valid: bool
    rejection_reason: Optional[str] = None


class RiskManager:
    """Менеджер рисков"""

    def __init__(self, initial_balance: float = 10000.0, config: RiskConfig = None):
        self.balance = initial_balance
        self.config = config or RiskConfig()
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.peak_balance = initial_balance
        self.current_drawdown = 0.0

        # База данных для истории
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'risk.db')
        self._init_database()

    def _init_database(self):
        """Инициализация базы данных"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Таблица для позиций
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                size REAL NOT NULL,
                value REAL NOT NULL,
                risk_amount REAL NOT NULL,
                status TEXT DEFAULT 'open',
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                exit_price REAL,
                pnl REAL,
                pnl_percent REAL
            )
        ''')

        # Таблица для дневной статистики
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                starting_balance REAL,
                ending_balance REAL,
                pnl REAL,
                pnl_percent REAL,
                trades_count INTEGER,
                wins INTEGER,
                losses INTEGER,
                max_drawdown REAL
            )
        ''')

        # Таблица для конфигурации
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Risk database initialized at {self.db_path}")

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        side: str = 'LONG'
    ) -> PositionSize:
        """
        Расчет оптимального размера позиции

        Args:
            symbol: Торговая пара
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            side: 'LONG' или 'SHORT'

        Returns:
            PositionSize с результатом расчета
        """
        # Проверяем корректность параметров
        if side == 'LONG':
            if stop_loss >= entry_price:
                return PositionSize(
                    symbol=symbol, side=side, entry_price=entry_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    position_size=0, position_value=0, risk_amount=0,
                    risk_reward_ratio=0, is_valid=False,
                    rejection_reason="Stop Loss должен быть ниже цены входа для LONG"
                )
            if take_profit <= entry_price:
                return PositionSize(
                    symbol=symbol, side=side, entry_price=entry_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    position_size=0, position_value=0, risk_amount=0,
                    risk_reward_ratio=0, is_valid=False,
                    rejection_reason="Take Profit должен быть выше цены входа для LONG"
                )
        else:  # SHORT
            if stop_loss <= entry_price:
                return PositionSize(
                    symbol=symbol, side=side, entry_price=entry_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    position_size=0, position_value=0, risk_amount=0,
                    risk_reward_ratio=0, is_valid=False,
                    rejection_reason="Stop Loss должен быть выше цены входа для SHORT"
                )
            if take_profit >= entry_price:
                return PositionSize(
                    symbol=symbol, side=side, entry_price=entry_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    position_size=0, position_value=0, risk_amount=0,
                    risk_reward_ratio=0, is_valid=False,
                    rejection_reason="Take Profit должен быть ниже цены входа для SHORT"
                )

        # Расчет риска на единицу
        risk_per_unit = abs(entry_price - stop_loss)
        reward_per_unit = abs(take_profit - entry_price)

        # Risk:Reward ratio
        risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0

        # Проверка минимального R:R
        if risk_reward_ratio < self.config.min_risk_reward:
            return PositionSize(
                symbol=symbol, side=side, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                position_size=0, position_value=0, risk_amount=0,
                risk_reward_ratio=risk_reward_ratio, is_valid=False,
                rejection_reason=f"R:R ({risk_reward_ratio:.2f}) ниже минимума ({self.config.min_risk_reward})"
            )

        # Проверка дневного лимита убытков
        if self.daily_pnl < -self.balance * self.config.max_daily_loss:
            return PositionSize(
                symbol=symbol, side=side, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                position_size=0, position_value=0, risk_amount=0,
                risk_reward_ratio=risk_reward_ratio, is_valid=False,
                rejection_reason="Достигнут дневной лимит убытков"
            )

        # Проверка максимальной просадки
        if self.current_drawdown >= self.config.max_drawdown:
            return PositionSize(
                symbol=symbol, side=side, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                position_size=0, position_value=0, risk_amount=0,
                risk_reward_ratio=risk_reward_ratio, is_valid=False,
                rejection_reason="Достигнута максимальная просадка"
            )

        # Проверка существующей позиции
        if symbol in self.positions:
            return PositionSize(
                symbol=symbol, side=side, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                position_size=0, position_value=0, risk_amount=0,
                risk_reward_ratio=risk_reward_ratio, is_valid=False,
                rejection_reason=f"Уже есть открытая позиция по {symbol}"
            )

        # Проверка общей экспозиции
        current_exposure = self.get_total_exposure()
        if current_exposure >= self.config.max_total_exposure:
            return PositionSize(
                symbol=symbol, side=side, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                position_size=0, position_value=0, risk_amount=0,
                risk_reward_ratio=risk_reward_ratio, is_valid=False,
                rejection_reason=f"Превышена максимальная экспозиция ({current_exposure:.1%})"
            )

        # Расчет суммы риска
        risk_amount = self.balance * self.config.risk_per_trade

        # Расчет размера позиции
        position_size = risk_amount / risk_per_unit
        position_value = position_size * entry_price

        # Ограничение по максимальному размеру позиции
        max_position_value = self.balance * self.config.max_position_size
        if position_value > max_position_value:
            position_value = max_position_value
            position_size = position_value / entry_price
            risk_amount = position_size * risk_per_unit

        # Ограничение по оставшейся экспозиции
        remaining_exposure = (self.config.max_total_exposure - current_exposure) * self.balance
        if position_value > remaining_exposure:
            position_value = remaining_exposure
            position_size = position_value / entry_price
            risk_amount = position_size * risk_per_unit

        return PositionSize(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=round(position_size, 8),
            position_value=round(position_value, 2),
            risk_amount=round(risk_amount, 2),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            is_valid=True
        )

    def get_total_exposure(self) -> float:
        """Получение текущей общей экспозиции"""
        total_value = sum(pos.get('value', 0) for pos in self.positions.values())
        return total_value / self.balance if self.balance > 0 else 0

    def open_position(self, position: PositionSize) -> bool:
        """Открытие позиции"""
        if not position.is_valid:
            logger.warning(f"Cannot open invalid position: {position.rejection_reason}")
            return False

        self.positions[position.symbol] = {
            'side': position.side,
            'entry_price': position.entry_price,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'size': position.position_size,
            'value': position.position_value,
            'risk_amount': position.risk_amount,
            'entry_time': datetime.now()
        }

        # Сохраняем в базу
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO positions (symbol, side, entry_price, stop_loss, take_profit,
                                   size, value, risk_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position.symbol, position.side, position.entry_price,
            position.stop_loss, position.take_profit,
            position.position_size, position.position_value, position.risk_amount
        ))
        conn.commit()
        conn.close()

        logger.info(f"Opened {position.side} position for {position.symbol}: "
                   f"size={position.position_size}, value=${position.position_value}")
        return True

    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Закрытие позиции"""
        if symbol not in self.positions:
            logger.warning(f"No open position for {symbol}")
            return None

        pos = self.positions[symbol]

        # Расчет P&L
        if pos['side'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']

        pnl_percent = pnl / pos['value'] * 100 if pos['value'] > 0 else 0

        # Обновляем баланс
        self.balance += pnl
        self.daily_pnl += pnl

        # Обновляем drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_balance - self.balance) / self.peak_balance

        # Обновляем в базе
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE positions
            SET status = 'closed', exit_time = ?, exit_price = ?, pnl = ?, pnl_percent = ?
            WHERE symbol = ? AND status = 'open'
        ''', (datetime.now().isoformat(), exit_price, pnl, pnl_percent, symbol))
        conn.commit()
        conn.close()

        # Удаляем из активных
        del self.positions[symbol]

        logger.info(f"Closed {symbol} at ${exit_price}: P&L = ${pnl:.2f} ({pnl_percent:+.2f}%)")
        return pnl

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Проверка срабатывания стоп-лосса"""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        if pos['side'] == 'LONG':
            return current_price <= pos['stop_loss']
        else:
            return current_price >= pos['stop_loss']

    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Проверка срабатывания тейк-профита"""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        if pos['side'] == 'LONG':
            return current_price >= pos['take_profit']
        else:
            return current_price <= pos['take_profit']

    def get_position_pnl(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Получение текущего P&L позиции"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        if pos['side'] == 'LONG':
            pnl = (current_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - current_price) * pos['size']

        pnl_percent = pnl / pos['value'] * 100 if pos['value'] > 0 else 0

        return {
            'symbol': symbol,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'current_price': current_price,
            'size': pos['size'],
            'value': pos['value'],
            'pnl': round(pnl, 2),
            'pnl_percent': round(pnl_percent, 2),
            'stop_loss': pos['stop_loss'],
            'take_profit': pos['take_profit']
        }

    def get_portfolio_status(self) -> Dict:
        """Получение статуса портфеля"""
        return {
            'balance': round(self.balance, 2),
            'peak_balance': round(self.peak_balance, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'current_drawdown': round(self.current_drawdown * 100, 2),
            'total_exposure': round(self.get_total_exposure() * 100, 2),
            'open_positions': len(self.positions),
            'positions': list(self.positions.keys())
        }

    def reset_daily_stats(self):
        """Сброс дневной статистики"""
        # Сохраняем дневную статистику в базу
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            INSERT OR REPLACE INTO daily_stats
            (date, starting_balance, ending_balance, pnl, pnl_percent)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            today,
            self.balance - self.daily_pnl,
            self.balance,
            self.daily_pnl,
            (self.daily_pnl / (self.balance - self.daily_pnl)) * 100 if self.balance != self.daily_pnl else 0
        ))
        conn.commit()
        conn.close()

        # Сбрасываем дневной P&L
        self.daily_pnl = 0.0
        logger.info("Daily stats reset")

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Получение истории сделок"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT symbol, side, entry_price, exit_price, size, pnl, pnl_percent,
                   entry_time, exit_time, status
            FROM positions
            ORDER BY entry_time DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'symbol': row[0],
                'side': row[1],
                'entry_price': row[2],
                'exit_price': row[3],
                'size': row[4],
                'pnl': row[5],
                'pnl_percent': row[6],
                'entry_time': row[7],
                'exit_time': row[8],
                'status': row[9]
            }
            for row in rows
        ]


def main():
    """Демонстрация работы Risk Manager"""
    print("=" * 60)
    print("BOT 8: RISK MANAGER")
    print("Управление рисками и защита капитала")
    print("=" * 60)

    # Создаем менеджер с балансом $10,000
    rm = RiskManager(initial_balance=10000.0)

    print(f"\nНачальный баланс: ${rm.balance:,.2f}")
    print(f"Риск на сделку: {rm.config.risk_per_trade:.1%}")
    print(f"Макс. размер позиции: {rm.config.max_position_size:.1%}")
    print(f"Макс. экспозиция: {rm.config.max_total_exposure:.1%}")
    print(f"Минимальный R:R: {rm.config.min_risk_reward}")

    # Тестовые расчеты для BTC
    print("\n" + "-" * 60)
    print("ТЕСТ: Расчет позиции для BTCUSDT")
    print("-" * 60)

    position = rm.calculate_position_size(
        symbol='BTCUSDT',
        entry_price=95000,
        stop_loss=93000,      # -2.1%
        take_profit=99000,    # +4.2% (R:R = 2:1)
        side='LONG'
    )

    if position.is_valid:
        print(f"✅ Позиция валидна")
        print(f"   Размер: {position.position_size:.6f} BTC")
        print(f"   Стоимость: ${position.position_value:,.2f}")
        print(f"   Риск: ${position.risk_amount:,.2f}")
        print(f"   R:R: {position.risk_reward_ratio:.2f}")
        print(f"   Stop Loss: ${position.stop_loss:,.2f}")
        print(f"   Take Profit: ${position.take_profit:,.2f}")
    else:
        print(f"❌ Позиция отклонена: {position.rejection_reason}")

    # Тест с плохим R:R
    print("\n" + "-" * 60)
    print("ТЕСТ: Позиция с плохим R:R")
    print("-" * 60)

    bad_position = rm.calculate_position_size(
        symbol='ETHUSDT',
        entry_price=3300,
        stop_loss=3200,       # -3%
        take_profit=3400,     # +3% (R:R = 1:1)
        side='LONG'
    )

    if bad_position.is_valid:
        print(f"✅ Позиция валидна")
    else:
        print(f"❌ Позиция отклонена: {bad_position.rejection_reason}")

    # Симуляция открытия и закрытия
    if position.is_valid:
        print("\n" + "-" * 60)
        print("СИМУЛЯЦИЯ: Открытие и закрытие позиции")
        print("-" * 60)

        rm.open_position(position)
        print(f"\nПортфель после открытия:")
        status = rm.get_portfolio_status()
        print(f"   Экспозиция: {status['total_exposure']:.1f}%")
        print(f"   Открытые позиции: {status['open_positions']}")

        # Симулируем движение цены
        current_price = 96500  # Цена выросла
        pnl_info = rm.get_position_pnl('BTCUSDT', current_price)
        print(f"\nТекущий P&L при цене ${current_price:,}:")
        print(f"   P&L: ${pnl_info['pnl']:+,.2f} ({pnl_info['pnl_percent']:+.2f}%)")

        # Закрываем с прибылью
        pnl = rm.close_position('BTCUSDT', 98000)
        print(f"\nЗакрыто с P&L: ${pnl:+,.2f}")
        print(f"Новый баланс: ${rm.balance:,.2f}")

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ СТАТУС ПОРТФЕЛЯ")
    print("=" * 60)
    status = rm.get_portfolio_status()
    for key, value in status.items():
        print(f"   {key}: {value}")


if __name__ == '__main__':
    main()
