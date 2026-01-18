"""
Bot 9: Order Executor
Исполнение торговых ордеров через Binance API

Функционал:
- Market Order - немедленное исполнение
- Limit Order - по указанной цене
- Stop-Limit - стоп с лимитом
- Retry логика при ошибках
- Rate limiting
"""

import os
import sys
import hmac
import hashlib
import time
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OrderExecutor')

# Binance API endpoints
BINANCE_API_URL = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"


class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderSide(Enum):
    """Сторона ордера"""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Время действия ордера"""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


@dataclass
class OrderResult:
    """Результат исполнения ордера"""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: float = 0.0
    executed_qty: float = 0.0
    avg_price: float = 0.0
    status: str = ""
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OrderExecutor:
    """Исполнитель ордеров"""

    def __init__(self, use_testnet: bool = True):
        self.use_testnet = use_testnet
        self.base_url = BINANCE_TESTNET_URL if use_testnet else BINANCE_API_URL

        # API ключи
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.api_secret = os.getenv('BINANCE_API_SECRET', '')

        if not self.api_key or not self.api_secret:
            logger.warning("API keys not found in environment variables")

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms между запросами

        # Retry config
        self.max_retries = 3
        self.retry_delay = 1.0

        logger.info(f"OrderExecutor initialized (testnet={use_testnet})")

    def _sign_request(self, params: Dict) -> str:
        """Подписывание запроса"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        """Получение timestamp в мс"""
        return int(time.time() * 1000)

    def _rate_limit(self):
        """Rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Выполнение HTTP запроса"""
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key}

        if params is None:
            params = {}

        if signed:
            params['timestamp'] = self._get_timestamp()
            params['signature'] = self._sign_request(params)

        for attempt in range(self.max_retries):
            try:
                if method == 'GET':
                    response = requests.get(url, params=params, headers=headers, timeout=30)
                elif method == 'POST':
                    response = requests.post(url, params=params, headers=headers, timeout=30)
                elif method == 'DELETE':
                    response = requests.delete(url, params=params, headers=headers, timeout=30)
                else:
                    raise ValueError(f"Unknown method: {method}")

                data = response.json()

                if response.status_code != 200:
                    error_msg = data.get('msg', 'Unknown error')
                    logger.error(f"API error: {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    return {'error': error_msg}

                return data

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return {'error': str(e)}

        return {'error': 'Max retries exceeded'}

    def get_account_info(self) -> Optional[Dict]:
        """Получение информации об аккаунте"""
        result = self._make_request('GET', '/api/v3/account', signed=True)
        if 'error' in result:
            logger.error(f"Failed to get account info: {result['error']}")
            return None
        return result

    def get_balance(self, asset: str = 'USDT') -> float:
        """Получение баланса актива"""
        account = self.get_account_info()
        if not account:
            return 0.0

        for balance in account.get('balances', []):
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены"""
        result = self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
        if 'error' in result:
            return None
        return float(result.get('price', 0))

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Получение информации о торговой паре"""
        result = self._make_request('GET', '/api/v3/exchangeInfo', {'symbol': symbol})
        if 'error' in result:
            return None

        for s in result.get('symbols', []):
            if s['symbol'] == symbol:
                return s
        return None

    def _get_lot_size(self, symbol: str) -> Dict:
        """Получение параметров лота"""
        info = self.get_symbol_info(symbol)
        if not info:
            return {'minQty': 0, 'maxQty': 0, 'stepSize': 0}

        for f in info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                return {
                    'minQty': float(f['minQty']),
                    'maxQty': float(f['maxQty']),
                    'stepSize': float(f['stepSize'])
                }
        return {'minQty': 0, 'maxQty': 0, 'stepSize': 0}

    def _round_quantity(self, quantity: float, step_size: float) -> float:
        """Округление количества до размера шага"""
        if step_size == 0:
            return quantity
        precision = len(str(step_size).split('.')[-1].rstrip('0'))
        return round(quantity - (quantity % step_size), precision)

    def market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> OrderResult:
        """
        Создание Market ордера

        Args:
            symbol: Торговая пара (например, BTCUSDT)
            side: BUY или SELL
            quantity: Количество базового актива
        """
        # Получаем и округляем количество
        lot_size = self._get_lot_size(symbol)
        quantity = self._round_quantity(quantity, lot_size['stepSize'])

        if quantity < lot_size['minQty']:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                error_message=f"Quantity {quantity} below minimum {lot_size['minQty']}"
            )

        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.MARKET.value,
            'quantity': quantity
        }

        logger.info(f"Placing MARKET {side.value} order: {symbol} qty={quantity}")
        result = self._make_request('POST', '/api/v3/order', params, signed=True)

        if 'error' in result:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.MARKET.value,
                quantity=quantity,
                error_message=result['error']
            )

        # Вычисляем среднюю цену исполнения
        fills = result.get('fills', [])
        if fills:
            total_qty = sum(float(f['qty']) for f in fills)
            total_value = sum(float(f['qty']) * float(f['price']) for f in fills)
            avg_price = total_value / total_qty if total_qty > 0 else 0
        else:
            avg_price = float(result.get('price', 0))

        return OrderResult(
            success=True,
            order_id=str(result.get('orderId')),
            symbol=symbol,
            side=side.value,
            order_type=OrderType.MARKET.value,
            quantity=quantity,
            executed_qty=float(result.get('executedQty', 0)),
            avg_price=avg_price,
            status=result.get('status', 'UNKNOWN')
        )

    def limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        time_in_force: TimeInForce = TimeInForce.GTC
    ) -> OrderResult:
        """
        Создание Limit ордера

        Args:
            symbol: Торговая пара
            side: BUY или SELL
            quantity: Количество
            price: Цена
            time_in_force: Время действия
        """
        lot_size = self._get_lot_size(symbol)
        quantity = self._round_quantity(quantity, lot_size['stepSize'])

        if quantity < lot_size['minQty']:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                error_message=f"Quantity {quantity} below minimum {lot_size['minQty']}"
            )

        params = {
            'symbol': symbol,
            'side': side.value,
            'type': OrderType.LIMIT.value,
            'timeInForce': time_in_force.value,
            'quantity': quantity,
            'price': price
        }

        logger.info(f"Placing LIMIT {side.value} order: {symbol} qty={quantity} @ ${price}")
        result = self._make_request('POST', '/api/v3/order', params, signed=True)

        if 'error' in result:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                order_type=OrderType.LIMIT.value,
                quantity=quantity,
                price=price,
                error_message=result['error']
            )

        return OrderResult(
            success=True,
            order_id=str(result.get('orderId')),
            symbol=symbol,
            side=side.value,
            order_type=OrderType.LIMIT.value,
            quantity=quantity,
            price=price,
            executed_qty=float(result.get('executedQty', 0)),
            status=result.get('status', 'UNKNOWN')
        )

    def stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float = None
    ) -> OrderResult:
        """
        Создание Stop-Loss ордера

        Args:
            symbol: Торговая пара
            side: BUY или SELL
            quantity: Количество
            stop_price: Триггер цена
            limit_price: Лимитная цена (опционально)
        """
        lot_size = self._get_lot_size(symbol)
        quantity = self._round_quantity(quantity, lot_size['stepSize'])

        order_type = OrderType.STOP_LOSS_LIMIT if limit_price else OrderType.STOP_LOSS

        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': quantity,
            'stopPrice': stop_price
        }

        if limit_price:
            params['price'] = limit_price
            params['timeInForce'] = TimeInForce.GTC.value

        logger.info(f"Placing STOP_LOSS {side.value} order: {symbol} qty={quantity} stop=${stop_price}")
        result = self._make_request('POST', '/api/v3/order', params, signed=True)

        if 'error' in result:
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                quantity=quantity,
                price=stop_price,
                error_message=result['error']
            )

        return OrderResult(
            success=True,
            order_id=str(result.get('orderId')),
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            quantity=quantity,
            price=stop_price,
            status=result.get('status', 'UNKNOWN')
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Отмена ордера"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }

        logger.info(f"Canceling order {order_id} for {symbol}")
        result = self._make_request('DELETE', '/api/v3/order', params, signed=True)

        if 'error' in result:
            logger.error(f"Failed to cancel order: {result['error']}")
            return False

        return True

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Получение открытых ордеров"""
        params = {}
        if symbol:
            params['symbol'] = symbol

        result = self._make_request('GET', '/api/v3/openOrders', params, signed=True)

        if 'error' in result:
            logger.error(f"Failed to get open orders: {result['error']}")
            return []

        return result if isinstance(result, list) else []

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Получение статуса ордера"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }

        result = self._make_request('GET', '/api/v3/order', params, signed=True)

        if 'error' in result:
            return None

        return result


def main():
    """Демонстрация работы Order Executor"""
    print("=" * 60)
    print("BOT 9: ORDER EXECUTOR")
    print("Исполнение торговых ордеров")
    print("=" * 60)

    executor = OrderExecutor(use_testnet=True)

    # Получаем информацию об аккаунте
    print("\n--- Информация об аккаунте ---")
    account = executor.get_account_info()
    if account:
        print(f"Can Trade: {account.get('canTrade')}")
        print("\nБалансы (> 0):")
        for balance in account.get('balances', []):
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                print(f"  {balance['asset']}: {free:.4f} (locked: {locked:.4f})")
    else:
        print("Не удалось получить информацию об аккаунте")
        return

    # Получаем текущую цену BTC
    btc_price = executor.get_current_price('BTCUSDT')
    print(f"\nТекущая цена BTC: ${btc_price:,.2f}" if btc_price else "\nНе удалось получить цену BTC")

    # Тестовый Market Buy ордер (маленький объем)
    print("\n--- Тест: Market Buy Order ---")
    print("(Покупаем 0.001 BTC)")

    result = executor.market_order(
        symbol='BTCUSDT',
        side=OrderSide.BUY,
        quantity=0.001
    )

    if result.success:
        print(f"✅ Ордер исполнен!")
        print(f"   Order ID: {result.order_id}")
        print(f"   Executed: {result.executed_qty} BTC")
        print(f"   Avg Price: ${result.avg_price:,.2f}")
        print(f"   Status: {result.status}")

        # Закрываем позицию
        print("\n--- Тест: Market Sell Order ---")
        print("(Продаем купленные 0.001 BTC)")

        sell_result = executor.market_order(
            symbol='BTCUSDT',
            side=OrderSide.SELL,
            quantity=0.001
        )

        if sell_result.success:
            print(f"✅ Ордер исполнен!")
            print(f"   Order ID: {sell_result.order_id}")
            print(f"   Executed: {sell_result.executed_qty} BTC")
            print(f"   Avg Price: ${sell_result.avg_price:,.2f}")
            print(f"   Status: {sell_result.status}")

            # P&L расчет
            pnl = (sell_result.avg_price - result.avg_price) * result.executed_qty
            print(f"\n   P&L: ${pnl:+.2f}")
        else:
            print(f"❌ Ошибка: {sell_result.error_message}")
    else:
        print(f"❌ Ошибка: {result.error_message}")

    # Финальный баланс
    print("\n--- Финальный баланс ---")
    usdt = executor.get_balance('USDT')
    btc = executor.get_balance('BTC')
    print(f"  USDT: {usdt:,.2f}")
    print(f"  BTC: {btc:.6f}")


if __name__ == '__main__':
    main()
