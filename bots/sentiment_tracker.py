"""
Bot 3: Sentiment Tracker
Отслеживание рыночных настроений

Источники:
- Fear & Greed Index
- Funding Rate (заглушка)
- Long/Short Ratio (заглушка)
"""

import os
import logging
import requests
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SentimentTracker')


@dataclass
class SentimentSignal:
    """Сигнал настроений рынка"""
    fear_greed_index: int
    fear_greed_label: str
    signal: str  # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    trading_signal: str  # 'buy', 'sell', 'hold'
    strength: float  # -1 to 1
    timestamp: datetime


class SentimentTracker:
    """Трекер настроений"""

    FEAR_GREED_API = "https://api.alternative.me/fng/"

    def __init__(self):
        self.last_data = None

    def get_fear_greed_index(self) -> Optional[Dict]:
        """Получение Fear & Greed Index"""
        try:
            response = requests.get(self.FEAR_GREED_API, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('data'):
                return data['data'][0]
            return None
        except Exception as e:
            logger.error(f"Failed to get Fear & Greed Index: {e}")
            return None

    def analyze(self) -> Optional[SentimentSignal]:
        """Анализ настроений"""
        fg_data = self.get_fear_greed_index()

        if not fg_data:
            # Возвращаем нейтральный сигнал если API недоступен
            return SentimentSignal(
                fear_greed_index=50,
                fear_greed_label="Neutral",
                signal='neutral',
                trading_signal='hold',
                strength=0.0,
                timestamp=datetime.now()
            )

        index = int(fg_data.get('value', 50))
        label = fg_data.get('value_classification', 'Neutral')

        # Определение сигнала
        if index <= 20:
            signal = 'extreme_fear'
            trading_signal = 'buy'  # Покупаем когда все боятся
            strength = 0.8
        elif index <= 40:
            signal = 'fear'
            trading_signal = 'buy'
            strength = 0.4
        elif index >= 80:
            signal = 'extreme_greed'
            trading_signal = 'sell'  # Продаем когда все жадничают
            strength = -0.8
        elif index >= 60:
            signal = 'greed'
            trading_signal = 'sell'
            strength = -0.4
        else:
            signal = 'neutral'
            trading_signal = 'hold'
            strength = 0.0

        return SentimentSignal(
            fear_greed_index=index,
            fear_greed_label=label,
            signal=signal,
            trading_signal=trading_signal,
            strength=strength,
            timestamp=datetime.now()
        )


def main():
    print("=" * 60)
    print("BOT 3: SENTIMENT TRACKER")
    print("=" * 60)

    tracker = SentimentTracker()
    signal = tracker.analyze()

    if signal:
        if signal.signal in ['extreme_fear', 'fear']:
            icon = "😨"
            action = "🟢 BUY opportunity"
        elif signal.signal in ['extreme_greed', 'greed']:
            icon = "🤑"
            action = "🔴 SELL signal"
        else:
            icon = "😐"
            action = "⚪ HOLD"

        print(f"\nFear & Greed Index: {signal.fear_greed_index} ({signal.fear_greed_label})")
        print(f"Sentiment: {icon} {signal.signal.upper()}")
        print(f"Trading Signal: {action}")
        print(f"Strength: {signal.strength:+.2f}")

        # Визуализация индекса
        bar_len = 50
        pos = int(signal.fear_greed_index / 100 * bar_len)
        bar = "█" * pos + "░" * (bar_len - pos)
        print(f"\n[{bar}]")
        print(f" 0 Fear              50 Neutral             100 Greed")


if __name__ == '__main__':
    main()
