"""
Bot 10: Position Manager
Управление открытыми позициями

Функционал:
- Trailing Stop (подтягивание стопа)
- Частичное закрытие (TP1, TP2, TP3)
- Break-even перенос стопа
- Time-based exit (макс. время в позиции)
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PositionManager')


@dataclass
class TakeProfitLevel:
    """Уровень тейк-профита"""
    price: float
    percentage: float  # Какой % позиции закрыть
    triggered: bool = False


@dataclass
class Position:
    """Структура позиции"""
    symbol: str
    side: str  # 'LONG' или 'SHORT'
    entry_price: float
    current_price: float
    quantity: float
    stop_loss: float
    take_profits: List[TakeProfitLevel] = field(default_factory=list)
    trailing_stop_enabled: bool = False
    trailing_stop_distance: float = 0.0  # В процентах
    trailing_stop_activated: bool = False
    trailing_stop_price: float = 0.0
    break_even_triggered: bool = False
    break_even_threshold: float = 0.01  # 1% profit для активации
    entry_time: datetime = None
    max_hold_time: timedelta = None  # Максимальное время удержания

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
        if self.max_hold_time is None:
            self.max_hold_time = timedelta(hours=24)

    @property
    def unrealized_pnl(self) -> float:
        """Нереализованный P&L"""
        if self.side == 'LONG':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> float:
        """Нереализованный P&L в процентах"""
        if self.entry_price == 0:
            return 0
        if self.side == 'LONG':
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100

    @property
    def hold_time(self) -> timedelta:
        """Время удержания позиции"""
        return datetime.now() - self.entry_time

    @property
    def is_expired(self) -> bool:
        """Проверка истечения времени удержания"""
        return self.hold_time > self.max_hold_time


class PositionManager:
    """Менеджер позиций"""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'positions.db')
        self._init_database()

    def _init_database(self):
        """Инициализация базы данных"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                price REAL,
                quantity REAL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Position database initialized at {self.db_path}")

    def _log_event(self, symbol: str, event_type: str, price: float = 0, quantity: float = 0, details: str = ""):
        """Логирование события"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO position_events (symbol, event_type, price, quantity, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, event_type, price, quantity, details))
        conn.commit()
        conn.close()

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit_levels: List[Dict] = None,
        trailing_stop_percent: float = 0,
        max_hold_hours: int = 24
    ) -> Position:
        """
        Открытие новой позиции

        Args:
            symbol: Торговая пара
            side: 'LONG' или 'SHORT'
            entry_price: Цена входа
            quantity: Размер позиции
            stop_loss: Уровень стоп-лосса
            take_profit_levels: Список уровней TP [{price, percentage}, ...]
            trailing_stop_percent: Процент для трейлинг-стопа (0 = выкл)
            max_hold_hours: Максимальное время удержания в часах

        Returns:
            Position
        """
        # Создаем уровни TP
        tp_levels = []
        if take_profit_levels:
            for tp in take_profit_levels:
                tp_levels.append(TakeProfitLevel(
                    price=tp['price'],
                    percentage=tp['percentage'],
                    triggered=False
                ))

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profits=tp_levels,
            trailing_stop_enabled=trailing_stop_percent > 0,
            trailing_stop_distance=trailing_stop_percent,
            max_hold_time=timedelta(hours=max_hold_hours)
        )

        self.positions[symbol] = position
        self._log_event(symbol, 'OPEN', entry_price, quantity, f"Side: {side}, SL: {stop_loss}")

        logger.info(f"Opened {side} position: {symbol} @ ${entry_price} qty={quantity}")
        return position

    def update_price(self, symbol: str, current_price: float) -> Dict:
        """
        Обновление цены и проверка условий

        Returns:
            Dict с действиями, которые нужно выполнить
        """
        if symbol not in self.positions:
            return {'action': 'none'}

        position = self.positions[symbol]
        position.current_price = current_price

        actions = {'action': 'none', 'details': []}

        # Проверка стоп-лосса
        if self._check_stop_loss(position):
            actions['action'] = 'close_all'
            actions['reason'] = 'stop_loss'
            actions['price'] = position.stop_loss
            return actions

        # Проверка тейк-профитов
        tp_action = self._check_take_profits(position)
        if tp_action:
            actions['action'] = 'partial_close'
            actions['details'].append(tp_action)

        # Обновление трейлинг-стопа
        if position.trailing_stop_enabled:
            self._update_trailing_stop(position)

        # Проверка break-even
        if not position.break_even_triggered:
            self._check_break_even(position)

        # Проверка времени удержания
        if position.is_expired:
            actions['action'] = 'close_all'
            actions['reason'] = 'time_expired'
            actions['hold_time'] = str(position.hold_time)

        return actions

    def _check_stop_loss(self, position: Position) -> bool:
        """Проверка срабатывания стоп-лосса"""
        if position.side == 'LONG':
            triggered = position.current_price <= position.stop_loss
        else:
            triggered = position.current_price >= position.stop_loss

        if triggered:
            logger.warning(f"Stop Loss triggered for {position.symbol} at ${position.current_price}")
            self._log_event(position.symbol, 'STOP_LOSS', position.current_price, position.quantity)
        return triggered

    def _check_take_profits(self, position: Position) -> Optional[Dict]:
        """Проверка срабатывания тейк-профитов"""
        for tp in position.take_profits:
            if tp.triggered:
                continue

            triggered = False
            if position.side == 'LONG':
                triggered = position.current_price >= tp.price
            else:
                triggered = position.current_price <= tp.price

            if triggered:
                tp.triggered = True
                close_qty = position.quantity * tp.percentage
                logger.info(f"Take Profit triggered for {position.symbol}: "
                           f"close {tp.percentage*100:.0f}% at ${tp.price}")
                self._log_event(position.symbol, 'TAKE_PROFIT', tp.price, close_qty,
                              f"Partial close: {tp.percentage*100:.0f}%")
                return {
                    'price': tp.price,
                    'percentage': tp.percentage,
                    'quantity': close_qty
                }
        return None

    def _update_trailing_stop(self, position: Position):
        """Обновление трейлинг-стопа"""
        distance = position.trailing_stop_distance / 100

        if position.side == 'LONG':
            # Для лонга: стоп подтягивается вверх при росте цены
            new_trailing = position.current_price * (1 - distance)

            # Активируем только если цена выросла выше entry
            if position.current_price > position.entry_price:
                if not position.trailing_stop_activated:
                    position.trailing_stop_activated = True
                    position.trailing_stop_price = new_trailing
                    logger.info(f"Trailing stop activated for {position.symbol} at ${new_trailing:.2f}")
                elif new_trailing > position.trailing_stop_price:
                    old_price = position.trailing_stop_price
                    position.trailing_stop_price = new_trailing
                    logger.info(f"Trailing stop updated for {position.symbol}: "
                               f"${old_price:.2f} -> ${new_trailing:.2f}")

                # Обновляем основной стоп
                if position.trailing_stop_price > position.stop_loss:
                    position.stop_loss = position.trailing_stop_price
        else:
            # Для шорта: стоп подтягивается вниз при падении цены
            new_trailing = position.current_price * (1 + distance)

            if position.current_price < position.entry_price:
                if not position.trailing_stop_activated:
                    position.trailing_stop_activated = True
                    position.trailing_stop_price = new_trailing
                    logger.info(f"Trailing stop activated for {position.symbol} at ${new_trailing:.2f}")
                elif new_trailing < position.trailing_stop_price:
                    old_price = position.trailing_stop_price
                    position.trailing_stop_price = new_trailing
                    logger.info(f"Trailing stop updated for {position.symbol}: "
                               f"${old_price:.2f} -> ${new_trailing:.2f}")

                if position.trailing_stop_price < position.stop_loss:
                    position.stop_loss = position.trailing_stop_price

    def _check_break_even(self, position: Position):
        """Проверка и активация break-even"""
        pnl_percent = position.unrealized_pnl_percent / 100

        if pnl_percent >= position.break_even_threshold:
            old_sl = position.stop_loss
            # Переносим стоп на точку входа + небольшая прибыль
            position.stop_loss = position.entry_price * (1.001 if position.side == 'LONG' else 0.999)
            position.break_even_triggered = True

            logger.info(f"Break-even activated for {position.symbol}: "
                       f"SL moved from ${old_sl:.2f} to ${position.stop_loss:.2f}")
            self._log_event(position.symbol, 'BREAK_EVEN', position.stop_loss, 0,
                          f"SL moved from ${old_sl:.2f}")

    def close_position(self, symbol: str, exit_price: float, partial_percent: float = 1.0) -> Optional[Dict]:
        """
        Закрытие позиции (полное или частичное)

        Args:
            symbol: Торговая пара
            exit_price: Цена выхода
            partial_percent: Доля позиции для закрытия (1.0 = полное)

        Returns:
            Dict с результатом
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        close_qty = position.quantity * partial_percent

        # Расчет P&L
        if position.side == 'LONG':
            pnl = (exit_price - position.entry_price) * close_qty
        else:
            pnl = (position.entry_price - exit_price) * close_qty

        pnl_percent = pnl / (position.entry_price * close_qty) * 100

        result = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': close_qty,
            'pnl': round(pnl, 2),
            'pnl_percent': round(pnl_percent, 2),
            'hold_time': str(position.hold_time)
        }

        if partial_percent >= 1.0:
            # Полное закрытие
            del self.positions[symbol]
            self._log_event(symbol, 'CLOSE', exit_price, close_qty,
                          f"P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")
            logger.info(f"Closed {symbol}: P&L = ${pnl:.2f} ({pnl_percent:+.2f}%)")
        else:
            # Частичное закрытие
            position.quantity -= close_qty
            self._log_event(symbol, 'PARTIAL_CLOSE', exit_price, close_qty,
                          f"Remaining: {position.quantity}, P&L: ${pnl:.2f}")
            logger.info(f"Partial close {symbol}: {partial_percent*100:.0f}% at ${exit_price}, "
                       f"P&L = ${pnl:.2f}")

        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        """Получение позиции"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """Получение всех позиций"""
        result = {}
        for symbol, pos in self.positions.items():
            result[symbol] = {
                'symbol': symbol,
                'side': pos.side,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'quantity': pos.quantity,
                'stop_loss': pos.stop_loss,
                'unrealized_pnl': round(pos.unrealized_pnl, 2),
                'unrealized_pnl_percent': round(pos.unrealized_pnl_percent, 2),
                'trailing_stop_enabled': pos.trailing_stop_enabled,
                'trailing_stop_price': pos.trailing_stop_price if pos.trailing_stop_activated else None,
                'break_even_triggered': pos.break_even_triggered,
                'hold_time': str(pos.hold_time),
                'is_expired': pos.is_expired
            }
        return result

    def get_position_events(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Получение истории событий"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if symbol:
            cursor.execute('''
                SELECT symbol, event_type, price, quantity, details, timestamp
                FROM position_events
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT symbol, event_type, price, quantity, details, timestamp
                FROM position_events
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'symbol': row[0],
                'event_type': row[1],
                'price': row[2],
                'quantity': row[3],
                'details': row[4],
                'timestamp': row[5]
            }
            for row in rows
        ]


def main():
    """Демонстрация работы Position Manager"""
    print("=" * 60)
    print("BOT 10: POSITION MANAGER")
    print("Управление открытыми позициями")
    print("=" * 60)

    pm = PositionManager()

    # Открываем позицию с трейлинг-стопом и несколькими TP
    print("\n--- Открытие позиции ---")
    position = pm.open_position(
        symbol='BTCUSDT',
        side='LONG',
        entry_price=95000,
        quantity=0.01,
        stop_loss=93000,
        take_profit_levels=[
            {'price': 97000, 'percentage': 0.33},  # TP1: 33% @ $97k
            {'price': 99000, 'percentage': 0.33},  # TP2: 33% @ $99k
            {'price': 101000, 'percentage': 0.34}, # TP3: 34% @ $101k
        ],
        trailing_stop_percent=2.0,  # 2% трейлинг
        max_hold_hours=48
    )

    print(f"✅ Позиция открыта:")
    print(f"   Символ: {position.symbol}")
    print(f"   Сторона: {position.side}")
    print(f"   Entry: ${position.entry_price:,}")
    print(f"   Количество: {position.quantity}")
    print(f"   Stop Loss: ${position.stop_loss:,}")
    print(f"   Trailing: {position.trailing_stop_distance}%")

    # Симуляция движения цены
    print("\n--- Симуляция движения цены ---")
    price_moves = [95500, 96000, 96500, 97000, 97500, 96800, 98000, 99000, 98500]

    for price in price_moves:
        print(f"\nЦена: ${price:,}")
        actions = pm.update_price('BTCUSDT', price)

        pos = pm.get_position('BTCUSDT')
        if pos:
            print(f"   P&L: ${pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_percent:+.2f}%)")
            print(f"   Stop Loss: ${pos.stop_loss:,.2f}")
            if pos.trailing_stop_activated:
                print(f"   Trailing Stop: ${pos.trailing_stop_price:,.2f}")

        if actions['action'] == 'partial_close':
            print(f"   📈 Take Profit сработал!")
            for detail in actions['details']:
                print(f"      Закрыть {detail['percentage']*100:.0f}% @ ${detail['price']:,}")

        if actions['action'] == 'close_all':
            print(f"   🛑 {actions['reason'].upper()}")
            break

    # Финальный статус
    print("\n--- Все позиции ---")
    for symbol, info in pm.get_all_positions().items():
        print(f"\n{symbol}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

    # Закрытие позиции
    print("\n--- Закрытие позиции ---")
    if 'BTCUSDT' in pm.positions:
        result = pm.close_position('BTCUSDT', 98500)
        if result:
            print(f"✅ Позиция закрыта:")
            print(f"   Entry: ${result['entry_price']:,}")
            print(f"   Exit: ${result['exit_price']:,}")
            print(f"   P&L: ${result['pnl']:+.2f} ({result['pnl_percent']:+.2f}%)")
            print(f"   Hold time: {result['hold_time']}")

    # История событий
    print("\n--- История событий ---")
    events = pm.get_position_events(limit=10)
    for event in events:
        print(f"  {event['timestamp']}: {event['event_type']} {event['symbol']} "
              f"@ ${event['price']:.2f} - {event['details']}")


if __name__ == '__main__':
    main()
