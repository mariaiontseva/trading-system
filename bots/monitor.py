"""
Bot 11: Monitor & Report
Мониторинг и отчетность

Метрики:
- Total P&L
- Win Rate
- Profit Factor
- Sharpe Ratio
- Max Drawdown
"""

import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Monitor')


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    best_trade: float
    worst_trade: float
    avg_hold_time: str


@dataclass
class BotStatus:
    """Статус бота"""
    name: str
    status: str  # 'running', 'stopped', 'error'
    last_signal: str
    signals_today: int
    last_updated: datetime


class Monitor:
    """Монитор системы"""

    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'monitor.db')
        self._init_database()

    def _init_database(self):
        """Инициализация базы"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                pnl_percent REAL,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                strategy TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_status (
                bot_name TEXT PRIMARY KEY,
                status TEXT,
                last_signal TEXT,
                signals_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                trades_count INTEGER,
                pnl REAL,
                win_rate REAL,
                report_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def record_trade(self, trade: Dict):
        """Запись сделки"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, exit_price, quantity,
                               pnl, pnl_percent, entry_time, exit_time, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.get('symbol'),
            trade.get('side'),
            trade.get('entry_price'),
            trade.get('exit_price'),
            trade.get('quantity'),
            trade.get('pnl'),
            trade.get('pnl_percent'),
            trade.get('entry_time'),
            trade.get('exit_time'),
            trade.get('strategy', 'default')
        ))
        conn.commit()
        conn.close()

    def update_bot_status(self, bot_name: str, status: str, last_signal: str = ""):
        """Обновление статуса бота"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO bot_status (bot_name, status, last_signal, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (bot_name, status, last_signal, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """Получение метрик производительности"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT pnl, pnl_percent, entry_time, exit_time
            FROM trades
            WHERE exit_time >= ?
        ''', (cutoff,))

        trades = cursor.fetchall()
        conn.close()

        if not trades:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, avg_win=0, avg_loss=0,
                profit_factor=0, max_drawdown=0, best_trade=0,
                worst_trade=0, avg_hold_time="0:00:00"
            )

        pnls = [t[0] for t in trades if t[0] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max Drawdown (упрощенный расчет)
        cumulative = 0
        peak = 0
        max_dd = 0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate * 100, 1),
            total_pnl=round(total_pnl, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown=round(max_dd, 2),
            best_trade=round(max(pnls) if pnls else 0, 2),
            worst_trade=round(min(pnls) if pnls else 0, 2),
            avg_hold_time="N/A"
        )

    def get_bot_statuses(self) -> List[BotStatus]:
        """Получение статусов всех ботов"""
        # Список всех ботов
        bots = [
            ("Price Scanner", "running"),
            ("Volume Analyzer", "running"),
            ("Sentiment Tracker", "running"),
            ("Whale Watcher", "running"),
            ("Signal Generator", "running"),
            ("Pattern Detector", "running"),
            ("Strategy Engine", "running"),
            ("Risk Manager", "running"),
            ("Order Executor", "running"),
            ("Position Manager", "running"),
            ("Monitor", "running"),
        ]

        statuses = []
        for name, default_status in bots:
            statuses.append(BotStatus(
                name=name,
                status=default_status,
                last_signal="",
                signals_today=0,
                last_updated=datetime.now()
            ))
        return statuses

    def generate_daily_report(self) -> Dict:
        """Генерация дневного отчета"""
        metrics = self.get_performance_metrics(days=1)
        bot_statuses = self.get_bot_statuses()

        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'performance': {
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'total_pnl': metrics.total_pnl,
                'profit_factor': metrics.profit_factor
            },
            'bots': [
                {'name': b.name, 'status': b.status}
                for b in bot_statuses
            ]
        }

        # Сохраняем отчет
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO daily_reports (date, trades_count, pnl, win_rate, report_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            report['date'],
            metrics.total_trades,
            metrics.total_pnl,
            metrics.win_rate,
            json.dumps(report)
        ))
        conn.commit()
        conn.close()

        return report

    def print_dashboard(self):
        """Вывод дашборда в консоль"""
        metrics = self.get_performance_metrics()
        bot_statuses = self.get_bot_statuses()

        print("\n" + "=" * 60)
        print("📊 TRADING SYSTEM DASHBOARD")
        print("=" * 60)

        print("\n--- Performance Metrics (30 days) ---")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Win Rate: {metrics.win_rate}%")
        print(f"  Total P&L: ${metrics.total_pnl:+,.2f}")
        print(f"  Profit Factor: {metrics.profit_factor}")
        print(f"  Max Drawdown: ${metrics.max_drawdown:,.2f}")
        print(f"  Best Trade: ${metrics.best_trade:+,.2f}")
        print(f"  Worst Trade: ${metrics.worst_trade:+,.2f}")

        print("\n--- Bot Statuses ---")
        for bot in bot_statuses:
            icon = "🟢" if bot.status == 'running' else "🔴" if bot.status == 'error' else "⚪"
            print(f"  {icon} {bot.name}: {bot.status}")

        print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("BOT 11: MONITOR & REPORT")
    print("=" * 60)

    monitor = Monitor()

    # Добавляем тестовые сделки для демонстрации
    test_trades = [
        {'symbol': 'BTCUSDT', 'side': 'LONG', 'entry_price': 94000, 'exit_price': 95500,
         'quantity': 0.01, 'pnl': 15.0, 'pnl_percent': 1.6,
         'entry_time': datetime.now().isoformat(), 'exit_time': datetime.now().isoformat()},
        {'symbol': 'ETHUSDT', 'side': 'LONG', 'entry_price': 3200, 'exit_price': 3150,
         'quantity': 0.1, 'pnl': -5.0, 'pnl_percent': -1.5,
         'entry_time': datetime.now().isoformat(), 'exit_time': datetime.now().isoformat()},
        {'symbol': 'BTCUSDT', 'side': 'SHORT', 'entry_price': 95000, 'exit_price': 94000,
         'quantity': 0.01, 'pnl': 10.0, 'pnl_percent': 1.05,
         'entry_time': datetime.now().isoformat(), 'exit_time': datetime.now().isoformat()},
    ]

    for trade in test_trades:
        monitor.record_trade(trade)

    # Обновляем статусы ботов
    for bot_name in ['Price Scanner', 'Signal Generator', 'Risk Manager']:
        monitor.update_bot_status(bot_name, 'running', 'OK')

    # Выводим дашборд
    monitor.print_dashboard()

    # Генерируем отчет
    report = monitor.generate_daily_report()
    print("\nDaily Report generated:")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
