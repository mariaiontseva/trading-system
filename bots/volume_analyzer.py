"""
Bot 2: Volume Analyzer
Анализ объемов торгов

Индикаторы:
- VWAP (Volume Weighted Average Price)
- OBV (On Balance Volume)
- Volume Profile
- Аномальные всплески объема
"""

import os
import sqlite3
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VolumeAnalyzer')

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'prices.db')
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


@dataclass
class VolumeSignal:
    """Сигнал на основе объема"""
    symbol: str
    signal_type: str  # 'spike', 'accumulation', 'distribution', 'neutral'
    strength: float   # -1 to 1
    volume_ratio: float  # Отношение к среднему
    vwap: float
    obv_trend: str  # 'bullish', 'bearish', 'neutral'
    reasons: List[str]


class VolumeAnalyzer:
    """Анализатор объемов"""

    def __init__(self):
        self.db_path = DB_PATH

    def get_data(self, symbol: str, interval: str = '1h', limit: int = 200) -> Optional[Dict]:
        """Получение данных из базы"""
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
            'timestamps': np.array([r[0] for r in rows]),
            'open': np.array([r[1] for r in rows]),
            'high': np.array([r[2] for r in rows]),
            'low': np.array([r[3] for r in rows]),
            'close': np.array([r[4] for r in rows]),
            'volume': np.array([r[5] for r in rows])
        }

    def calculate_vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """VWAP - Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = np.cumsum(typical_price * volume)
        cumulative_vol = np.cumsum(volume)
        return cumulative_tp_vol / cumulative_vol

    def calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """OBV - On Balance Volume"""
        obv = np.zeros(len(close))
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv

    def detect_volume_spike(self, volume: np.ndarray, threshold: float = 2.0) -> bool:
        """Детекция всплеска объема"""
        if len(volume) < 20:
            return False
        avg_volume = np.mean(volume[-20:-1])
        current_volume = volume[-1]
        return current_volume > avg_volume * threshold

    def analyze(self, symbol: str, interval: str = '1h') -> Optional[VolumeSignal]:
        """Анализ объемов для символа"""
        data = self.get_data(symbol, interval)
        if data is None:
            return None

        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # Расчет индикаторов
        vwap = self.calculate_vwap(high, low, close, volume)
        obv = self.calculate_obv(close, volume)

        # Анализ
        reasons = []
        score = 0.0

        # Объем относительно среднего
        avg_vol = np.mean(volume[-20:-1]) if len(volume) > 20 else np.mean(volume)
        volume_ratio = volume[-1] / avg_vol if avg_vol > 0 else 1

        # Всплеск объема
        if volume_ratio > 2.0:
            if close[-1] > close[-2]:
                score += 0.3
                reasons.append(f"Bullish volume spike ({volume_ratio:.1f}x)")
            else:
                score -= 0.3
                reasons.append(f"Bearish volume spike ({volume_ratio:.1f}x)")

        # VWAP
        current_price = close[-1]
        current_vwap = vwap[-1]
        if current_price > current_vwap * 1.01:
            score += 0.2
            reasons.append("Price above VWAP (bullish)")
        elif current_price < current_vwap * 0.99:
            score -= 0.2
            reasons.append("Price below VWAP (bearish)")

        # OBV тренд
        obv_sma = np.mean(obv[-20:]) if len(obv) >= 20 else np.mean(obv)
        obv_trend = 'neutral'
        if obv[-1] > obv_sma * 1.05:
            obv_trend = 'bullish'
            score += 0.2
            reasons.append("OBV rising (accumulation)")
        elif obv[-1] < obv_sma * 0.95:
            obv_trend = 'bearish'
            score -= 0.2
            reasons.append("OBV falling (distribution)")

        # Определение типа сигнала
        if score > 0.3:
            signal_type = 'accumulation'
        elif score < -0.3:
            signal_type = 'distribution'
        elif volume_ratio > 2.0:
            signal_type = 'spike'
        else:
            signal_type = 'neutral'

        return VolumeSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=max(-1, min(1, score)),
            volume_ratio=round(volume_ratio, 2),
            vwap=round(current_vwap, 2),
            obv_trend=obv_trend,
            reasons=reasons
        )

    def analyze_all(self, interval: str = '1h') -> List[VolumeSignal]:
        """Анализ всех символов"""
        signals = []
        for symbol in SYMBOLS:
            signal = self.analyze(symbol, interval)
            if signal:
                signals.append(signal)
        return signals


def main():
    print("=" * 60)
    print("BOT 2: VOLUME ANALYZER")
    print("=" * 60)

    analyzer = VolumeAnalyzer()
    signals = analyzer.analyze_all()

    for sig in signals:
        icon = "🟢" if sig.strength > 0 else "🔴" if sig.strength < 0 else "⚪"
        print(f"\n{icon} {sig.symbol}: {sig.signal_type.upper()}")
        print(f"   Strength: {sig.strength:+.2f}")
        print(f"   Volume Ratio: {sig.volume_ratio}x")
        print(f"   VWAP: ${sig.vwap:,.2f}")
        print(f"   OBV Trend: {sig.obv_trend}")
        for r in sig.reasons:
            print(f"   • {r}")


if __name__ == '__main__':
    main()
