"""
Bot 1: Price Scanner
Сбор и хранение ценовых данных с Binance

Функционал:
- Скачивание исторических данных за 4 года (2020-2024)
- Real-time WebSocket для живых цен
- Хранение в SQLite база данных
- Поддержка интервалов: 1m, 5m, 15m, 1h, 4h, 1d
"""

import os
import sys
import time
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PriceScanner')

# Binance API endpoints
BINANCE_API_URL = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"

# Активы для мониторинга
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

# Интервалы для скачивания
INTERVALS = {
    '1h': 60 * 60 * 1000,      # 1 час в мс
    '4h': 4 * 60 * 60 * 1000,  # 4 часа в мс
    '1d': 24 * 60 * 60 * 1000, # 1 день в мс
}

# Путь к базе данных
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'prices.db')


class PriceScanner:
    """Бот для сбора и хранения ценовых данных"""

    def __init__(self, use_testnet: bool = False):
        self.base_url = BINANCE_TESTNET_URL if use_testnet else BINANCE_API_URL
        self.db_path = DB_PATH
        self.symbols = SYMBOLS
        self._init_database()

    def _init_database(self):
        """Инициализация SQLite базы данных"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Таблица для OHLCV данных
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                close_time INTEGER NOT NULL,
                quote_volume REAL NOT NULL,
                trades INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, interval, open_time)
            )
        ''')

        # Индексы для быстрого поиска
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_interval_time
            ON ohlcv(symbol, interval, open_time)
        ''')

        # Таблица для текущих цен
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_prices (
                symbol TEXT PRIMARY KEY,
                price REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Таблица для статуса скачивания
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                candles_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, interval)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
        """
        Получение исторических свечей с Binance API

        Args:
            symbol: Торговая пара (например, BTCUSDT)
            interval: Интервал (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Начальное время в мс
            end_time: Конечное время в мс
            limit: Максимум свечей за запрос (макс 1000)

        Returns:
            Список свечей в формате OHLCV
        """
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []

    def download_historical_data(
        self,
        symbol: str,
        interval: str = '1h',
        start_date: str = '2020-01-01',
        end_date: str = None
    ) -> int:
        """
        Скачивание исторических данных за период

        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD), по умолчанию сегодня

        Returns:
            Количество скачанных свечей
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        logger.info(f"Downloading {symbol} {interval} from {start_date} to {end_date}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Обновляем статус
        cursor.execute('''
            INSERT OR REPLACE INTO download_status (symbol, interval, start_date, end_date, status)
            VALUES (?, ?, ?, ?, 'downloading')
        ''', (symbol, interval, start_date, end_date))
        conn.commit()

        total_candles = 0
        current_ts = start_ts
        interval_ms = INTERVALS.get(interval, 60 * 60 * 1000)

        while current_ts < end_ts:
            # Получаем порцию данных
            klines = self.get_klines(symbol, interval, current_ts, end_ts, limit=1000)

            if not klines:
                logger.warning(f"No data received for {symbol} at {current_ts}")
                current_ts += interval_ms * 1000
                continue

            # Сохраняем в базу
            for kline in klines:
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO ohlcv
                        (symbol, interval, open_time, open, high, low, close, volume,
                         close_time, quote_volume, trades)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        interval,
                        kline[0],   # open_time
                        float(kline[1]),  # open
                        float(kline[2]),  # high
                        float(kline[3]),  # low
                        float(kline[4]),  # close
                        float(kline[5]),  # volume
                        kline[6],   # close_time
                        float(kline[7]),  # quote_volume
                        kline[8],   # trades
                    ))
                except Exception as e:
                    logger.error(f"Error inserting kline: {e}")

            conn.commit()
            total_candles += len(klines)

            # Переходим к следующей порции
            if klines:
                current_ts = klines[-1][6] + 1  # close_time + 1ms
            else:
                current_ts += interval_ms * 1000

            # Пауза чтобы не превысить rate limit
            time.sleep(0.1)

            # Логируем прогресс каждые 5000 свечей
            if total_candles % 5000 == 0:
                progress_date = datetime.fromtimestamp(current_ts / 1000).strftime('%Y-%m-%d')
                logger.info(f"{symbol} {interval}: {total_candles} candles, at {progress_date}")

        # Обновляем статус
        cursor.execute('''
            UPDATE download_status
            SET candles_count = ?, status = 'completed', last_updated = CURRENT_TIMESTAMP
            WHERE symbol = ? AND interval = ?
        ''', (total_candles, symbol, interval))
        conn.commit()
        conn.close()

        logger.info(f"Completed {symbol} {interval}: {total_candles} candles downloaded")
        return total_candles

    def download_all_historical(self, start_date: str = '2020-01-01') -> Dict[str, int]:
        """
        Скачивание исторических данных для всех активов и интервалов

        Args:
            start_date: Начальная дата

        Returns:
            Словарь с количеством свечей по каждому активу
        """
        results = {}

        for symbol in self.symbols:
            results[symbol] = {}
            for interval in INTERVALS.keys():
                try:
                    count = self.download_historical_data(symbol, interval, start_date)
                    results[symbol][interval] = count
                except Exception as e:
                    logger.error(f"Error downloading {symbol} {interval}: {e}")
                    results[symbol][interval] = 0

        return results

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены актива"""
        url = f"{self.base_url}/api/v3/ticker/price"
        params = {'symbol': symbol}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            price = float(data['price'])

            # Сохраняем в базу
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO current_prices (symbol, price, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, price))
            conn.commit()
            conn.close()

            return price
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_all_current_prices(self) -> Dict[str, float]:
        """Получение текущих цен для всех активов"""
        prices = {}
        for symbol in self.symbols:
            price = self.get_current_price(symbol)
            if price:
                prices[symbol] = price
        return prices

    def get_ohlcv_data(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 100
    ) -> List[Dict]:
        """
        Получение OHLCV данных из базы

        Args:
            symbol: Торговая пара
            interval: Интервал
            limit: Количество последних свечей

        Returns:
            Список словарей с OHLCV данными
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT open_time, open, high, low, close, volume, quote_volume, trades
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time DESC
            LIMIT ?
        ''', (symbol, interval, limit))

        rows = cursor.fetchall()
        conn.close()

        # Преобразуем в список словарей (от старых к новым)
        data = []
        for row in reversed(rows):
            data.append({
                'timestamp': row[0],
                'datetime': datetime.fromtimestamp(row[0] / 1000).isoformat(),
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5],
                'quote_volume': row[6],
                'trades': row[7]
            })

        return data

    def get_download_status(self) -> List[Dict]:
        """Получение статуса скачивания данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT symbol, interval, start_date, end_date, candles_count, status, last_updated
            FROM download_status
            ORDER BY symbol, interval
        ''')

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'symbol': row[0],
                'interval': row[1],
                'start_date': row[2],
                'end_date': row[3],
                'candles_count': row[4],
                'status': row[5],
                'last_updated': row[6]
            }
            for row in rows
        ]

    def get_data_summary(self) -> Dict:
        """Получение сводки по скачанным данным"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Общее количество свечей
        cursor.execute('SELECT COUNT(*) FROM ohlcv')
        total_candles = cursor.fetchone()[0]

        # Количество по символам
        cursor.execute('''
            SELECT symbol, COUNT(*) as count,
                   MIN(datetime(open_time/1000, 'unixepoch')) as first_date,
                   MAX(datetime(open_time/1000, 'unixepoch')) as last_date
            FROM ohlcv
            GROUP BY symbol
        ''')
        by_symbol = {row[0]: {'count': row[1], 'first': row[2], 'last': row[3]}
                     for row in cursor.fetchall()}

        # Размер базы данных
        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

        conn.close()

        return {
            'total_candles': total_candles,
            'by_symbol': by_symbol,
            'db_size_mb': round(db_size_mb, 2),
            'symbols': self.symbols,
            'intervals': list(INTERVALS.keys())
        }


def main():
    """Основная функция для запуска скачивания"""
    print("=" * 60)
    print("BOT 1: PRICE SCANNER")
    print("Скачивание исторических данных за 4 года (2020-2024)")
    print("=" * 60)

    scanner = PriceScanner()

    # Показываем текущее состояние
    summary = scanner.get_data_summary()
    print(f"\nТекущее состояние базы:")
    print(f"  Всего свечей: {summary['total_candles']:,}")
    print(f"  Размер БД: {summary['db_size_mb']} MB")

    if summary['by_symbol']:
        print(f"\nДанные по активам:")
        for symbol, info in summary['by_symbol'].items():
            print(f"  {symbol}: {info['count']:,} свечей ({info['first']} - {info['last']})")

    # Скачиваем данные
    print(f"\nНачинаем скачивание для {len(SYMBOLS)} активов...")
    print(f"Активы: {', '.join(SYMBOLS)}")
    print(f"Интервалы: {', '.join(INTERVALS.keys())}")
    print()

    start_time = time.time()
    results = scanner.download_all_historical(start_date='2020-01-01')
    elapsed = time.time() - start_time

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СКАЧИВАНИЯ")
    print("=" * 60)

    total = 0
    for symbol, intervals in results.items():
        symbol_total = sum(intervals.values())
        total += symbol_total
        print(f"\n{symbol}:")
        for interval, count in intervals.items():
            print(f"  {interval}: {count:,} свечей")

    print(f"\n{'=' * 60}")
    print(f"ИТОГО: {total:,} свечей за {elapsed:.1f} секунд")

    # Обновленная сводка
    summary = scanner.get_data_summary()
    print(f"Размер БД: {summary['db_size_mb']} MB")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
