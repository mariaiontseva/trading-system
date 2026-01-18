"""
Binance API Client - загрузка исторических данных и live trading
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
from loguru import logger
import time


class BinanceDataLoader:
    """
    Загрузчик исторических данных с Binance
    """
    
    # Максимум свечей за запрос
    MAX_CANDLES_PER_REQUEST = 1000
    
    # Таймфреймы в миллисекундах
    TIMEFRAME_MS = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
    }
    
    def __init__(self, api_key: str = '', api_secret: str = ''):
        """
        Инициализация без ключей - только для публичных данных
        С ключами - для торговли
        """
        self.client = Client(api_key, api_secret) if api_key else Client()
        logger.info("BinanceDataLoader initialized")
    
    def get_all_symbols(self, quote_asset: str = 'USDT') -> List[str]:
        """
        Получить все торговые пары с определённой quote валютой
        """
        exchange_info = self.client.get_exchange_info()
        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['quoteAsset'] == quote_asset and s['status'] == 'TRADING'
        ]
        logger.info(f"Found {len(symbols)} trading pairs with {quote_asset}")
        return symbols
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Загрузка исторических свечей
        
        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            interval: Таймфрейм ('1m', '5m', '1h', '1d', etc.)
            start_date: Начальная дата ('2020-01-01' или '1 Jan, 2020')
            end_date: Конечная дата (по умолчанию - сейчас)
            progress_callback: Функция для отображения прогресса
        
        Returns:
            DataFrame с колонками: open, high, low, close, volume, etc.
        """
        logger.info(f"Loading {symbol} {interval} from {start_date} to {end_date or 'now'}")
        
        all_klines = []
        
        # Конвертация дат в timestamp
        start_ts = self._date_to_timestamp(start_date)
        end_ts = self._date_to_timestamp(end_date) if end_date else int(datetime.now().timestamp() * 1000)
        
        current_ts = start_ts
        total_expected = (end_ts - start_ts) // self.TIMEFRAME_MS.get(interval, 60000)
        loaded = 0
        
        while current_ts < end_ts:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_ts,
                    endTime=end_ts,
                    limit=self.MAX_CANDLES_PER_REQUEST
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                loaded += len(klines)
                
                # Следующий запрос начинается после последней свечи
                current_ts = klines[-1][0] + 1
                
                # Progress callback
                if progress_callback:
                    progress = min(loaded / max(total_expected, 1) * 100, 100)
                    progress_callback(symbol, interval, progress, loaded)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                time.sleep(1)
                continue
        
        if not all_klines:
            logger.warning(f"No data loaded for {symbol}")
            return pd.DataFrame()
        
        # Конвертация в DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Преобразование типов
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        df['trades'] = df['trades'].astype(int)
        
        # Удаляем ненужные колонки
        df.drop(['close_time', 'taker_buy_base', 'taker_buy_quote', 'ignore'], axis=1, inplace=True)
        
        # Удаляем дубликаты
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Loaded {len(df)} candles for {symbol} {interval}")
        return df
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        progress_callback=None
    ) -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных для нескольких символов
        """
        data = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Loading {symbol} ({i+1}/{total})")
            
            df = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if not df.empty:
                data[symbol] = df
            
            if progress_callback:
                progress_callback(symbol, (i + 1) / total * 100)
            
            # Rate limiting между символами
            time.sleep(0.5)
        
        return data
    
    def get_funding_rate_history(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Загрузка истории funding rate (только для фьючерсов)
        """
        from binance.client import Client as FuturesClient
        
        try:
            # Для фьючерсов нужен отдельный endpoint
            start_ts = self._date_to_timestamp(start_date)
            end_ts = self._date_to_timestamp(end_date) if end_date else int(datetime.now().timestamp() * 1000)
            
            funding_rates = self.client.futures_funding_rate(
                symbol=symbol,
                startTime=start_ts,
                endTime=end_ts,
                limit=1000
            )
            
            if not funding_rates:
                return pd.DataFrame()
            
            df = pd.DataFrame(funding_rates)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            df.set_index('fundingTime', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading funding rates for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Получить текущий order book
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                'bids': [(float(price), float(qty)) for price, qty in depth['bids']],
                'asks': [(float(price), float(qty)) for price, qty in depth['asks']],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {}
    
    def get_recent_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Получить последние сделки
        """
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            df = pd.DataFrame(trades)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['qty'] = df['qty'].astype(float)
            return df
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return pd.DataFrame()
    
    def _date_to_timestamp(self, date_str: str) -> int:
        """
        Конвертация строки даты в timestamp (ms)
        """
        if isinstance(date_str, int):
            return date_str
        
        try:
            # Попробуем разные форматы
            for fmt in ['%Y-%m-%d', '%d %b, %Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return int(dt.timestamp() * 1000)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse date: {date_str}")
        except Exception as e:
            logger.error(f"Date parsing error: {e}")
            raise


class BinanceTrader:
    """
    Клиент для торговли на Binance
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Args:
            api_key: API ключ
            api_secret: API секрет
            testnet: Использовать тестовую сеть
        """
        self.testnet = testnet
        
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binance.vision/api'
            logger.info("BinanceTrader initialized in TESTNET mode")
        else:
            self.client = Client(api_key, api_secret)
            logger.warning("BinanceTrader initialized in LIVE mode!")
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Получить баланс аккаунта
        """
        try:
            account = self.client.get_account()
            balances = {}
            for balance in account['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:
                    balances[balance['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            return balances
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """
        Получить текущую цену
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float
    ) -> Dict:
        """
        Рыночный ордер
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"Market order placed: {side} {quantity} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return {'error': str(e)}
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict:
        """
        Лимитный ордер
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=str(price)
            )
            logger.info(f"Limit order placed: {side} {quantity} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return {'error': str(e)}
    
    def place_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float,
        stop_limit_price: float
    ) -> Dict:
        """
        OCO ордер (One-Cancels-Other) - для stop-loss + take-profit
        """
        try:
            order = self.client.create_oco_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=str(price),  # Take profit
                stopPrice=str(stop_price),
                stopLimitPrice=str(stop_limit_price),
                stopLimitTimeInForce='GTC'
            )
            logger.info(f"OCO order placed: {side} {quantity} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Error placing OCO order: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """
        Отмена ордера
        """
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Order {order_id} cancelled")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'error': str(e)}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Получить открытые ордера
        """
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            return orders
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def get_order_status(self, symbol: str, order_id: int) -> Dict:
        """
        Получить статус ордера
        """
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return order
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'error': str(e)}


class BinanceWebSocket:
    """
    WebSocket клиент для real-time данных
    """
    
    def __init__(self, api_key: str = '', api_secret: str = ''):
        self.api_key = api_key
        self.api_secret = api_secret
        self.bm = None
        self.streams = {}
    
    async def start(self):
        """
        Запуск WebSocket менеджера
        """
        self.client = await AsyncClient.create(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.client)
        logger.info("WebSocket manager started")
    
    async def stop(self):
        """
        Остановка WebSocket
        """
        if self.client:
            await self.client.close_connection()
        logger.info("WebSocket manager stopped")
    
    async def subscribe_klines(self, symbol: str, interval: str, callback):
        """
        Подписка на обновления свечей
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        
        async with self.bm.kline_socket(symbol=symbol, interval=interval) as stream:
            while True:
                msg = await stream.recv()
                if msg:
                    kline = msg['k']
                    data = {
                        'symbol': kline['s'],
                        'interval': kline['i'],
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'is_closed': kline['x'],
                        'timestamp': datetime.fromtimestamp(kline['t'] / 1000)
                    }
                    await callback(data)
    
    async def subscribe_trades(self, symbol: str, callback):
        """
        Подписка на сделки
        """
        async with self.bm.trade_socket(symbol=symbol) as stream:
            while True:
                msg = await stream.recv()
                if msg:
                    data = {
                        'symbol': msg['s'],
                        'price': float(msg['p']),
                        'quantity': float(msg['q']),
                        'is_buyer_maker': msg['m'],
                        'timestamp': datetime.fromtimestamp(msg['T'] / 1000)
                    }
                    await callback(data)
    
    async def subscribe_depth(self, symbol: str, callback):
        """
        Подписка на order book
        """
        async with self.bm.depth_socket(symbol=symbol) as stream:
            while True:
                msg = await stream.recv()
                if msg:
                    data = {
                        'symbol': symbol,
                        'bids': [(float(p), float(q)) for p, q in msg['b']],
                        'asks': [(float(p), float(q)) for p, q in msg['a']],
                        'timestamp': datetime.now()
                    }
                    await callback(data)


# Тест загрузки данных
if __name__ == '__main__':
    loader = BinanceDataLoader()
    
    # Тест загрузки исторических данных
    df = loader.get_historical_klines(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    print(f"Loaded {len(df)} candles")
    print(df.head())
    print(df.tail())
