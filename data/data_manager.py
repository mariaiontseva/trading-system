"""
Data Manager - загрузка, хранение и управление историческими данными
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.binance_client import BinanceDataLoader


class DataManager:
    """
    Управление историческими данными
    - Загрузка с Binance
    - Хранение в SQLite/Parquet
    - Кэширование
    """
    
    def __init__(self, data_dir: str = 'data/historical'):
        """
        Args:
            data_dir: Директория для хранения данных
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / 'market_data.db'
        self.loader = BinanceDataLoader()
        
        self._init_database()
        logger.info(f"DataManager initialized. Data dir: {self.data_dir}")
    
    def _init_database(self):
        """
        Инициализация SQLite базы данных
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица для метаданных о загруженных данных
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                candles_count INTEGER,
                file_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, interval)
            )
        ''')
        
        # Таблица для OHLCV данных (для небольших объёмов)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                quote_volume REAL,
                trades INTEGER,
                UNIQUE(symbol, interval, timestamp)
            )
        ''')
        
        # Индексы для быстрого поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval ON ohlcv(symbol, interval)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv(timestamp)')
        
        conn.commit()
        conn.close()
    
    def download_all_data(
        self,
        symbols: List[str],
        intervals: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        progress_callback=None
    ) -> Dict[str, Dict[str, int]]:
        """
        Загрузка данных для всех символов и интервалов
        
        Returns:
            Статистика загрузки: {symbol: {interval: candles_count}}
        """
        stats = {}
        total_tasks = len(symbols) * len(intervals)
        completed = 0
        
        for symbol in symbols:
            stats[symbol] = {}
            
            for interval in intervals:
                try:
                    logger.info(f"Downloading {symbol} {interval}...")
                    
                    # Проверяем, есть ли уже данные
                    existing = self._get_existing_data_range(symbol, interval)
                    
                    if existing and existing['end_date']:
                        # Дозагружаем только новые данные
                        actual_start = existing['end_date']
                        logger.info(f"Updating {symbol} {interval} from {actual_start}")
                    else:
                        actual_start = start_date
                    
                    # Загружаем данные
                    df = self.loader.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_date=actual_start,
                        end_date=end_date
                    )
                    
                    if not df.empty:
                        # Сохраняем в Parquet (эффективнее для больших данных)
                        self._save_to_parquet(symbol, interval, df)
                        
                        # Обновляем реестр
                        self._update_registry(symbol, interval, df)
                        
                        stats[symbol][interval] = len(df)
                        logger.info(f"Saved {len(df)} candles for {symbol} {interval}")
                    else:
                        stats[symbol][interval] = 0
                    
                    completed += 1
                    
                    if progress_callback:
                        progress_callback({
                            'symbol': symbol,
                            'interval': interval,
                            'progress': completed / total_tasks * 100,
                            'completed': completed,
                            'total': total_tasks
                        })
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol} {interval}: {e}")
                    stats[symbol][interval] = -1
        
        return stats
    
    def _save_to_parquet(self, symbol: str, interval: str, df: pd.DataFrame):
        """
        Сохранение данных в Parquet формате
        """
        # Создаём директорию для символа
        symbol_dir = self.data_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        file_path = symbol_dir / f"{interval}.parquet"
        
        # Если файл существует, объединяем данные
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df])
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
        
        df.to_parquet(file_path, compression='snappy')
    
    def _update_registry(self, symbol: str, interval: str, df: pd.DataFrame):
        """
        Обновление реестра данных
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        file_path = str(self.data_dir / symbol / f"{interval}.parquet")
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_registry 
            (symbol, interval, start_date, end_date, candles_count, file_path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            interval,
            df.index.min().isoformat(),
            df.index.max().isoformat(),
            len(df),
            file_path,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _get_existing_data_range(self, symbol: str, interval: str) -> Optional[Dict]:
        """
        Получить диапазон уже загруженных данных
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT start_date, end_date, candles_count 
            FROM data_registry 
            WHERE symbol = ? AND interval = ?
        ''', (symbol, interval))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'start_date': row[0],
                'end_date': row[1],
                'candles_count': row[2]
            }
        return None
    
    def load_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Загрузка данных из локального хранилища
        """
        file_path = self.data_dir / symbol / f"{interval}.parquet"
        
        if not file_path.exists():
            logger.warning(f"No data found for {symbol} {interval}")
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        
        # Фильтрация по датам
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def get_available_data(self) -> List[Dict]:
        """
        Получить список доступных данных
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, interval, start_date, end_date, candles_count, updated_at
            FROM data_registry
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
                'updated_at': row[5]
            }
            for row in rows
        ]
    
    def get_data_stats(self) -> Dict:
        """
        Статистика по загруженным данным
        """
        available = self.get_available_data()
        
        symbols = set(d['symbol'] for d in available)
        intervals = set(d['interval'] for d in available)
        total_candles = sum(d['candles_count'] for d in available)
        
        # Размер на диске
        total_size = sum(
            f.stat().st_size 
            for f in self.data_dir.rglob('*.parquet')
        )
        
        return {
            'symbols_count': len(symbols),
            'intervals_count': len(intervals),
            'total_candles': total_candles,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'symbols': list(symbols),
            'intervals': list(intervals)
        }
    
    def prepare_backtest_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Подготовка данных для бэктеста с feature engineering
        """
        df = self.load_data(symbol, interval, start_date, end_date)
        
        if df.empty:
            return df
        
        if add_features:
            df = self._add_technical_features(df)
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление технических индикаторов
        """
        from ta import trend, momentum, volatility, volume as ta_volume

        # Returns
        for period in [1, 5, 10, 20, 50]:
            df[f'return_{period}'] = df['close'].pct_change(period)

        # Moving Averages
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Volatility - ATR
        atr_indicator = volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr_14'] = atr_indicator.average_true_range()

        # Bollinger Bands
        bb_indicator = volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_mid'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # Momentum - RSI
        rsi_14 = momentum.RSIIndicator(df['close'], window=14)
        rsi_7 = momentum.RSIIndicator(df['close'], window=7)
        df['rsi_14'] = rsi_14.rsi()
        df['rsi_7'] = rsi_7.rsi()

        # MACD
        macd_indicator = trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()

        # Stochastic
        stoch_indicator = momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_indicator.stoch()
        df['stoch_d'] = stoch_indicator.stoch_signal()

        # Trend - ADX
        adx_indicator = trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()

        # Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        obv_indicator = ta_volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
        df['obv'] = obv_indicator.on_balance_volume()

        # Price position
        df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Z-score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)

        return df


class DataDownloaderCLI:
    """
    CLI для загрузки данных
    """
    
    def __init__(self):
        self.manager = DataManager()
    
    def download_default_dataset(self, start_date: str = '2022-01-01'):
        """
        Загрузка стандартного набора данных
        """
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
            'MATICUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'APTUSDT'
        ]
        
        intervals = ['5m', '15m', '1h', '4h', '1d']
        
        def progress(info):
            print(f"[{info['completed']}/{info['total']}] "
                  f"{info['symbol']} {info['interval']} - "
                  f"{info['progress']:.1f}%")
        
        print(f"Starting download of {len(symbols)} symbols, {len(intervals)} intervals")
        print(f"From: {start_date}")
        print("-" * 50)
        
        stats = self.manager.download_all_data(
            symbols=symbols,
            intervals=intervals,
            start_date=start_date,
            progress_callback=progress
        )
        
        print("-" * 50)
        print("Download complete!")
        
        # Статистика
        data_stats = self.manager.get_data_stats()
        print(f"Total symbols: {data_stats['symbols_count']}")
        print(f"Total candles: {data_stats['total_candles']:,}")
        print(f"Total size: {data_stats['total_size_mb']} MB")
        
        return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download historical data from Binance')
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to download')
    parser.add_argument('--intervals', type=str, nargs='+', default=['1h', '4h', '1d'], help='Intervals')
    
    args = parser.parse_args()
    
    cli = DataDownloaderCLI()
    cli.download_default_dataset(start_date=args.start)
