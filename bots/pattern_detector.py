"""
Bot 6: Pattern Detector
Распознавание графических паттернов

Свечные паттерны:
- Doji, Hammer, Shooting Star
- Engulfing (бычий/медвежий)
- Morning/Evening Star

Уровни:
- Support/Resistance
- Pivot Points
"""

import os
import sqlite3
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PatternDetector')

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'prices.db')
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


@dataclass
class Pattern:
    """Обнаруженный паттерн"""
    name: str
    type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0 to 1
    price: float


@dataclass
class PatternSignal:
    """Сигнал на основе паттернов"""
    symbol: str
    patterns: List[Pattern]
    support: float
    resistance: float
    pivot: float
    overall_signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float


class PatternDetector:
    """Детектор паттернов"""

    def __init__(self):
        self.db_path = DB_PATH

    def get_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[Dict]:
        """Получение данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv WHERE symbol = ? AND interval = ?
            ORDER BY open_time DESC LIMIT ?
        ''', (symbol, interval, limit))
        rows = list(reversed(cursor.fetchall()))
        conn.close()

        if not rows:
            return None

        return {
            'open': np.array([r[1] for r in rows]),
            'high': np.array([r[2] for r in rows]),
            'low': np.array([r[3] for r in rows]),
            'close': np.array([r[4] for r in rows]),
        }

    def detect_doji(self, o: float, h: float, l: float, c: float) -> Optional[Pattern]:
        """Детекция Doji"""
        body = abs(c - o)
        range_ = h - l
        if range_ > 0 and body / range_ < 0.1:
            return Pattern("Doji", "neutral", 0.5, c)
        return None

    def detect_hammer(self, o: float, h: float, l: float, c: float) -> Optional[Pattern]:
        """Детекция Hammer/Hanging Man"""
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        range_ = h - l

        if range_ > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.5:
            if c > o:
                return Pattern("Hammer", "bullish", 0.7, c)
            else:
                return Pattern("Hanging Man", "bearish", 0.6, c)
        return None

    def detect_shooting_star(self, o: float, h: float, l: float, c: float) -> Optional[Pattern]:
        """Детекция Shooting Star"""
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            return Pattern("Shooting Star", "bearish", 0.7, c)
        return None

    def detect_engulfing(self, opens: np.ndarray, closes: np.ndarray) -> Optional[Pattern]:
        """Детекция Engulfing"""
        if len(opens) < 2:
            return None

        prev_o, prev_c = opens[-2], closes[-2]
        curr_o, curr_c = opens[-1], closes[-1]

        # Bullish Engulfing
        if prev_c < prev_o and curr_c > curr_o:
            if curr_o < prev_c and curr_c > prev_o:
                return Pattern("Bullish Engulfing", "bullish", 0.8, curr_c)

        # Bearish Engulfing
        if prev_c > prev_o and curr_c < curr_o:
            if curr_o > prev_c and curr_c < prev_o:
                return Pattern("Bearish Engulfing", "bearish", 0.8, curr_c)

        return None

    def calculate_pivot_points(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, float, float]:
        """Расчет Pivot Points"""
        h = high[-1]
        l = low[-1]
        c = close[-1]

        pivot = (h + l + c) / 3
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h

        return round(s1, 2), round(pivot, 2), round(r1, 2)

    def find_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, float]:
        """Поиск уровней поддержки/сопротивления"""
        # Простой метод: минимумы и максимумы за последние N свечей
        recent_low = np.min(low[-20:])
        recent_high = np.max(high[-20:])
        return round(recent_low, 2), round(recent_high, 2)

    def analyze(self, symbol: str, interval: str = '1h') -> Optional[PatternSignal]:
        """Анализ паттернов"""
        data = self.get_data(symbol, interval)
        if data is None:
            return None

        opens = data['open']
        highs = data['high']
        lows = data['low']
        closes = data['close']

        patterns = []

        # Анализ последней свечи
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]

        if p := self.detect_doji(o, h, l, c):
            patterns.append(p)
        if p := self.detect_hammer(o, h, l, c):
            patterns.append(p)
        if p := self.detect_shooting_star(o, h, l, c):
            patterns.append(p)
        if p := self.detect_engulfing(opens, closes):
            patterns.append(p)

        # Уровни
        support, resistance = self.find_support_resistance(highs, lows, closes)
        s1, pivot, r1 = self.calculate_pivot_points(highs, lows, closes)

        # Общий сигнал
        bullish_count = sum(1 for p in patterns if p.type == 'bullish')
        bearish_count = sum(1 for p in patterns if p.type == 'bearish')

        if bullish_count > bearish_count:
            overall = 'bullish'
            strength = sum(p.strength for p in patterns if p.type == 'bullish') / max(bullish_count, 1)
        elif bearish_count > bullish_count:
            overall = 'bearish'
            strength = sum(p.strength for p in patterns if p.type == 'bearish') / max(bearish_count, 1)
        else:
            overall = 'neutral'
            strength = 0.0

        return PatternSignal(
            symbol=symbol,
            patterns=patterns,
            support=support,
            resistance=resistance,
            pivot=pivot,
            overall_signal=overall,
            strength=strength
        )

    def analyze_all(self, interval: str = '1h') -> List[PatternSignal]:
        """Анализ всех символов"""
        signals = []
        for symbol in SYMBOLS:
            signal = self.analyze(symbol, interval)
            if signal:
                signals.append(signal)
        return signals


def main():
    print("=" * 60)
    print("BOT 6: PATTERN DETECTOR")
    print("=" * 60)

    detector = PatternDetector()
    signals = detector.analyze_all()

    for sig in signals:
        icon = "🟢" if sig.overall_signal == 'bullish' else "🔴" if sig.overall_signal == 'bearish' else "⚪"
        print(f"\n{icon} {sig.symbol}: {sig.overall_signal.upper()}")
        print(f"   Support: ${sig.support:,.2f}")
        print(f"   Pivot: ${sig.pivot:,.2f}")
        print(f"   Resistance: ${sig.resistance:,.2f}")
        if sig.patterns:
            print(f"   Patterns:")
            for p in sig.patterns:
                print(f"   • {p.name} ({p.type}, strength: {p.strength:.1f})")
        else:
            print(f"   No patterns detected")


if __name__ == '__main__':
    main()
