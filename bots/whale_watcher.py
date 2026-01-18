"""
Bot 4: Whale Watcher
Отслеживание крупных игроков

Мониторинг:
- Крупные ордера в orderbook
- Большие транзакции
"""

import os
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WhaleWatcher')

BINANCE_API = "https://api.binance.com"
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


@dataclass
class WhaleOrder:
    """Крупный ордер"""
    price: float
    quantity: float
    value_usd: float
    side: str  # 'bid', 'ask'


@dataclass
class WhaleSignal:
    """Сигнал от китов"""
    symbol: str
    large_bids: List[WhaleOrder]
    large_asks: List[WhaleOrder]
    bid_wall_value: float
    ask_wall_value: float
    imbalance: float  # positive = more buy pressure
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float


class WhaleWatcher:
    """Наблюдатель за китами"""

    def __init__(self, min_order_value: float = 50000):
        self.min_order_value = min_order_value  # Минимум $50k для "кита"

    def get_orderbook(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Получение orderbook"""
        try:
            url = f"{BINANCE_API}/api/v3/depth"
            response = requests.get(url, params={'symbol': symbol, 'limit': limit}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены"""
        try:
            url = f"{BINANCE_API}/api/v3/ticker/price"
            response = requests.get(url, params={'symbol': symbol}, timeout=10)
            response.raise_for_status()
            return float(response.json().get('price', 0))
        except:
            return 0

    def analyze(self, symbol: str) -> Optional[WhaleSignal]:
        """Анализ активности китов"""
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return None

        current_price = self.get_current_price(symbol)
        if current_price == 0:
            return None

        large_bids = []
        large_asks = []

        # Анализ bids (покупки)
        for bid in orderbook.get('bids', []):
            price = float(bid[0])
            qty = float(bid[1])
            value = price * qty

            if value >= self.min_order_value:
                large_bids.append(WhaleOrder(
                    price=price,
                    quantity=qty,
                    value_usd=value,
                    side='bid'
                ))

        # Анализ asks (продажи)
        for ask in orderbook.get('asks', []):
            price = float(ask[0])
            qty = float(ask[1])
            value = price * qty

            if value >= self.min_order_value:
                large_asks.append(WhaleOrder(
                    price=price,
                    quantity=qty,
                    value_usd=value,
                    side='ask'
                ))

        bid_wall = sum(b.value_usd for b in large_bids)
        ask_wall = sum(a.value_usd for a in large_asks)

        total = bid_wall + ask_wall
        if total > 0:
            imbalance = (bid_wall - ask_wall) / total
        else:
            imbalance = 0

        # Определение сигнала
        if imbalance > 0.3:
            signal = 'bullish'
            strength = min(imbalance, 1.0)
        elif imbalance < -0.3:
            signal = 'bearish'
            strength = max(imbalance, -1.0)
        else:
            signal = 'neutral'
            strength = imbalance

        return WhaleSignal(
            symbol=symbol,
            large_bids=large_bids[:5],
            large_asks=large_asks[:5],
            bid_wall_value=bid_wall,
            ask_wall_value=ask_wall,
            imbalance=round(imbalance, 3),
            signal=signal,
            strength=round(strength, 3)
        )

    def analyze_all(self) -> List[WhaleSignal]:
        """Анализ всех символов"""
        signals = []
        for symbol in SYMBOLS:
            signal = self.analyze(symbol)
            if signal:
                signals.append(signal)
        return signals


def main():
    print("=" * 60)
    print("BOT 4: WHALE WATCHER")
    print("=" * 60)

    watcher = WhaleWatcher(min_order_value=50000)
    signals = watcher.analyze_all()

    for sig in signals:
        if sig.signal == 'bullish':
            icon = "🐋🟢"
        elif sig.signal == 'bearish':
            icon = "🐋🔴"
        else:
            icon = "🐋⚪"

        print(f"\n{icon} {sig.symbol}: {sig.signal.upper()}")
        print(f"   Bid Wall: ${sig.bid_wall_value:,.0f}")
        print(f"   Ask Wall: ${sig.ask_wall_value:,.0f}")
        print(f"   Imbalance: {sig.imbalance:+.1%}")

        if sig.large_bids:
            print(f"   Top Bids:")
            for b in sig.large_bids[:3]:
                print(f"     ${b.price:,.2f}: ${b.value_usd:,.0f}")

        if sig.large_asks:
            print(f"   Top Asks:")
            for a in sig.large_asks[:3]:
                print(f"     ${a.price:,.2f}: ${a.value_usd:,.0f}")


if __name__ == '__main__':
    main()
