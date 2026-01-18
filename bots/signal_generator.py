"""
Bot 5: Signal Generator
Генерация торговых сигналов на основе технических индикаторов

Индикаторы:
- RSI (14) - перекупленность/перепроданность
- MACD (12, 26, 9) - тренд и моментум
- Bollinger Bands (20, 2) - волатильность
- EMA (9, 21, 50, 200) - тренд
- Stochastic RSI - точки входа
- ATR (14) - волатильность для стоп-лоссов
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SignalGenerator')

# Путь к базе данных
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'prices.db')

# Символы для анализа
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


class SignalType(Enum):
    """Типы сигналов"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Signal:
    """Структура торгового сигнала"""
    symbol: str
    signal_type: SignalType
    strength: float  # от -1 до 1
    price: float
    timestamp: datetime
    indicators: Dict[str, float]
    reasons: List[str]
    suggested_stop_loss: float
    suggested_take_profit: float


class TechnicalIndicators:
    """Расчет технических индикаторов"""

    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        return np.convolve(prices, np.ones(period) / period, mode='valid')

    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = np.zeros(len(prices))
        multiplier = 2 / (period + 1)

        # Первое значение - SMA
        ema[period - 1] = np.mean(prices[:period])

        # Остальные значения
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

        ema[:period - 1] = np.nan
        return ema

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))

        # Первые средние
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # Сглаженные средние
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = np.nan
        return rsi

    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)

        macd_line = ema_fast - ema_slow

        # Рассчитываем сигнальную линию
        signal_full = np.full(len(prices), np.nan)

        # Находим валидные значения MACD (начиная с slow-1)
        valid_start = slow - 1
        valid_macd = macd_line[valid_start:]

        if len(valid_macd) >= signal:
            # EMA сигнальной линии
            signal_ema = np.zeros(len(valid_macd))
            multiplier = 2 / (signal + 1)
            signal_ema[signal - 1] = np.mean(valid_macd[:signal])
            for i in range(signal, len(valid_macd)):
                signal_ema[i] = (valid_macd[i] - signal_ema[i - 1]) * multiplier + signal_ema[i - 1]
            signal_ema[:signal - 1] = np.nan

            # Помещаем в полный массив
            signal_full[valid_start:valid_start + len(signal_ema)] = signal_ema

        histogram = macd_line - signal_full
        return macd_line, signal_full, histogram

    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(prices, period)
        sma_full = np.full(len(prices), np.nan)
        sma_full[period - 1:] = sma

        std = np.array([np.std(prices[max(0, i - period + 1):i + 1]) for i in range(len(prices))])
        std[:period - 1] = np.nan

        upper = sma_full + std_dev * std
        lower = sma_full - std_dev * std
        return upper, sma_full, lower

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        if len(high) < 2:
            return np.full(len(high), np.nan)

        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        atr = np.zeros(len(high))
        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, len(high)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        atr[:period - 1] = np.nan
        return atr

    @staticmethod
    def stochastic_rsi(rsi: np.ndarray, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic RSI"""
        if len(rsi) < period:
            return np.full(len(rsi), np.nan), np.full(len(rsi), np.nan)

        stoch_rsi = np.zeros(len(rsi))
        for i in range(period - 1, len(rsi)):
            rsi_window = rsi[i - period + 1:i + 1]
            rsi_min = np.nanmin(rsi_window)
            rsi_max = np.nanmax(rsi_window)
            if rsi_max - rsi_min != 0:
                stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) * 100
            else:
                stoch_rsi[i] = 50

        stoch_rsi[:period - 1] = np.nan

        # Сглаживание K
        k = np.convolve(stoch_rsi[~np.isnan(stoch_rsi)], np.ones(smooth_k) / smooth_k, mode='valid')
        k_full = np.full(len(rsi), np.nan)
        k_full[period - 1 + smooth_k - 1:period - 1 + smooth_k - 1 + len(k)] = k

        # Сглаживание D
        d = np.convolve(k, np.ones(smooth_d) / smooth_d, mode='valid')
        d_full = np.full(len(rsi), np.nan)
        d_full[period - 1 + smooth_k - 1 + smooth_d - 1:period - 1 + smooth_k - 1 + smooth_d - 1 + len(d)] = d

        return k_full, d_full


class SignalGenerator:
    """Генератор торговых сигналов"""

    def __init__(self):
        self.db_path = DB_PATH
        self.indicators = TechnicalIndicators()
        self.signals_db = os.path.join(os.path.dirname(__file__), '..', 'data', 'signals.db')
        self._init_signals_db()

    def _init_signals_db(self):
        """Инициализация базы данных сигналов"""
        conn = sqlite3.connect(self.signals_db)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                ema_9 REAL,
                ema_21 REAL,
                ema_50 REAL,
                ema_200 REAL,
                atr REAL,
                stoch_k REAL,
                stoch_d REAL,
                stop_loss REAL,
                take_profit REAL,
                reasons TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Signals database initialized at {self.signals_db}")

    def get_ohlcv_data(self, symbol: str, interval: str = '1h', limit: int = 500) -> Optional[Dict]:
        """Получение OHLCV данных из базы"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time DESC
            LIMIT ?
        ''', (symbol, interval, limit))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Разворачиваем от старых к новым
        rows = list(reversed(rows))

        return {
            'timestamps': np.array([row[0] for row in rows]),
            'open': np.array([row[1] for row in rows]),
            'high': np.array([row[2] for row in rows]),
            'low': np.array([row[3] for row in rows]),
            'close': np.array([row[4] for row in rows]),
            'volume': np.array([row[5] for row in rows])
        }

    def calculate_indicators(self, data: Dict) -> Dict:
        """Расчет всех индикаторов"""
        close = data['close']
        high = data['high']
        low = data['low']

        # EMAs
        ema_9 = self.indicators.ema(close, 9)
        ema_21 = self.indicators.ema(close, 21)
        ema_50 = self.indicators.ema(close, 50)
        ema_200 = self.indicators.ema(close, 200)

        # RSI
        rsi = self.indicators.rsi(close, 14)

        # MACD
        macd_line, macd_signal, macd_histogram = self.indicators.macd(close)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(close)

        # ATR
        atr = self.indicators.atr(high, low, close, 14)

        # Stochastic RSI
        stoch_k, stoch_d = self.indicators.stochastic_rsi(rsi)

        return {
            'ema_9': ema_9[-1] if not np.isnan(ema_9[-1]) else None,
            'ema_21': ema_21[-1] if not np.isnan(ema_21[-1]) else None,
            'ema_50': ema_50[-1] if not np.isnan(ema_50[-1]) else None,
            'ema_200': ema_200[-1] if not np.isnan(ema_200[-1]) else None,
            'rsi': rsi[-1] if not np.isnan(rsi[-1]) else None,
            'macd': macd_line[-1] if not np.isnan(macd_line[-1]) else None,
            'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else None,
            'macd_histogram': macd_histogram[-1] if not np.isnan(macd_histogram[-1]) else None,
            'bb_upper': bb_upper[-1] if not np.isnan(bb_upper[-1]) else None,
            'bb_middle': bb_middle[-1] if not np.isnan(bb_middle[-1]) else None,
            'bb_lower': bb_lower[-1] if not np.isnan(bb_lower[-1]) else None,
            'atr': atr[-1] if not np.isnan(atr[-1]) else None,
            'stoch_k': stoch_k[-1] if not np.isnan(stoch_k[-1]) else None,
            'stoch_d': stoch_d[-1] if not np.isnan(stoch_d[-1]) else None,
            'current_price': close[-1],
            'prev_close': close[-2] if len(close) > 1 else close[-1],
            # Для анализа тренда
            'ema_9_prev': ema_9[-2] if len(ema_9) > 1 and not np.isnan(ema_9[-2]) else None,
            'macd_histogram_prev': macd_histogram[-2] if len(macd_histogram) > 1 and not np.isnan(macd_histogram[-2]) else None,
        }

    def generate_signal(self, symbol: str, interval: str = '1h') -> Optional[Signal]:
        """Генерация торгового сигнала для символа"""
        data = self.get_ohlcv_data(symbol, interval)
        if data is None:
            logger.warning(f"No data for {symbol}")
            return None

        indicators = self.calculate_indicators(data)
        if indicators['rsi'] is None:
            logger.warning(f"Not enough data to calculate indicators for {symbol}")
            return None

        # Анализ сигналов
        score = 0.0
        reasons = []

        current_price = indicators['current_price']
        rsi = indicators['rsi']
        macd_hist = indicators['macd_histogram']
        ema_21 = indicators['ema_21']
        ema_50 = indicators['ema_50']
        ema_200 = indicators['ema_200']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        stoch_k = indicators['stoch_k']
        atr = indicators['atr']

        # RSI сигналы
        if rsi < 30:
            score += 0.3
            reasons.append(f"RSI перепродан ({rsi:.1f})")
        elif rsi < 40:
            score += 0.1
            reasons.append(f"RSI низкий ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.3
            reasons.append(f"RSI перекуплен ({rsi:.1f})")
        elif rsi > 60:
            score -= 0.1
            reasons.append(f"RSI высокий ({rsi:.1f})")

        # MACD сигналы
        if macd_hist is not None:
            macd_hist_prev = indicators['macd_histogram_prev']
            if macd_hist > 0:
                score += 0.2
                reasons.append("MACD положительный")
                if macd_hist_prev is not None and macd_hist > macd_hist_prev:
                    score += 0.1
                    reasons.append("MACD растет")
            else:
                score -= 0.2
                reasons.append("MACD отрицательный")
                if macd_hist_prev is not None and macd_hist < macd_hist_prev:
                    score -= 0.1
                    reasons.append("MACD падает")

        # EMA тренд
        if ema_21 and ema_50:
            if current_price > ema_21 > ema_50:
                score += 0.2
                reasons.append("Бычий тренд (цена > EMA21 > EMA50)")
            elif current_price < ema_21 < ema_50:
                score -= 0.2
                reasons.append("Медвежий тренд (цена < EMA21 < EMA50)")

        # EMA 200 (долгосрочный тренд)
        if ema_200:
            if current_price > ema_200:
                score += 0.1
                reasons.append("Выше EMA200 (бычий)")
            else:
                score -= 0.1
                reasons.append("Ниже EMA200 (медвежий)")

        # Bollinger Bands
        if bb_lower and bb_upper:
            if current_price <= bb_lower:
                score += 0.2
                reasons.append("Цена на нижней BB (возможен отскок)")
            elif current_price >= bb_upper:
                score -= 0.2
                reasons.append("Цена на верхней BB (возможна коррекция)")

        # Stochastic RSI
        if stoch_k is not None:
            if stoch_k < 20:
                score += 0.15
                reasons.append(f"Stoch RSI перепродан ({stoch_k:.1f})")
            elif stoch_k > 80:
                score -= 0.15
                reasons.append(f"Stoch RSI перекуплен ({stoch_k:.1f})")

        # Нормализуем score в диапазон [-1, 1]
        score = max(-1, min(1, score))

        # Определяем тип сигнала
        if score >= 0.5:
            signal_type = SignalType.STRONG_BUY
        elif score >= 0.2:
            signal_type = SignalType.BUY
        elif score <= -0.5:
            signal_type = SignalType.STRONG_SELL
        elif score <= -0.2:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL

        # Расчет Stop Loss и Take Profit
        if atr:
            if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                stop_loss = current_price - 2 * atr
                take_profit = current_price + 3 * atr
            elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                stop_loss = current_price + 2 * atr
                take_profit = current_price - 3 * atr
            else:
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.03
        else:
            # Fallback: 2% SL, 3% TP
            stop_loss = current_price * (0.98 if score >= 0 else 1.02)
            take_profit = current_price * (1.03 if score >= 0 else 0.97)

        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=score,
            price=current_price,
            timestamp=datetime.now(),
            indicators=indicators,
            reasons=reasons,
            suggested_stop_loss=round(stop_loss, 2),
            suggested_take_profit=round(take_profit, 2)
        )

        # Сохраняем в базу
        self._save_signal(signal)

        return signal

    def _save_signal(self, signal: Signal):
        """Сохранение сигнала в базу"""
        conn = sqlite3.connect(self.signals_db)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO signals (
                symbol, signal_type, strength, price, timestamp,
                rsi, macd, macd_signal, macd_histogram,
                bb_upper, bb_middle, bb_lower,
                ema_9, ema_21, ema_50, ema_200,
                atr, stoch_k, stoch_d,
                stop_loss, take_profit, reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol,
            signal.signal_type.name,
            signal.strength,
            signal.price,
            signal.timestamp.isoformat(),
            signal.indicators.get('rsi'),
            signal.indicators.get('macd'),
            signal.indicators.get('macd_signal'),
            signal.indicators.get('macd_histogram'),
            signal.indicators.get('bb_upper'),
            signal.indicators.get('bb_middle'),
            signal.indicators.get('bb_lower'),
            signal.indicators.get('ema_9'),
            signal.indicators.get('ema_21'),
            signal.indicators.get('ema_50'),
            signal.indicators.get('ema_200'),
            signal.indicators.get('atr'),
            signal.indicators.get('stoch_k'),
            signal.indicators.get('stoch_d'),
            signal.suggested_stop_loss,
            signal.suggested_take_profit,
            '|'.join(signal.reasons)
        ))

        conn.commit()
        conn.close()

    def generate_all_signals(self, interval: str = '1h') -> List[Signal]:
        """Генерация сигналов для всех символов"""
        signals = []
        for symbol in SYMBOLS:
            signal = self.generate_signal(symbol, interval)
            if signal:
                signals.append(signal)
        return signals

    def get_latest_signals(self, limit: int = 50) -> List[Dict]:
        """Получение последних сигналов из базы"""
        conn = sqlite3.connect(self.signals_db)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT symbol, signal_type, strength, price, timestamp,
                   rsi, macd_histogram, stop_loss, take_profit, reasons
            FROM signals
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'symbol': row[0],
                'signal_type': row[1],
                'strength': row[2],
                'price': row[3],
                'timestamp': row[4],
                'rsi': row[5],
                'macd_histogram': row[6],
                'stop_loss': row[7],
                'take_profit': row[8],
                'reasons': row[9].split('|') if row[9] else []
            }
            for row in rows
        ]


def format_signal(signal: Signal) -> str:
    """Форматирование сигнала для вывода"""
    type_colors = {
        SignalType.STRONG_BUY: '🟢🟢',
        SignalType.BUY: '🟢',
        SignalType.NEUTRAL: '⚪',
        SignalType.SELL: '🔴',
        SignalType.STRONG_SELL: '🔴🔴'
    }

    output = []
    output.append(f"\n{'='*60}")
    output.append(f"{type_colors[signal.signal_type]} {signal.symbol} - {signal.signal_type.name}")
    output.append(f"{'='*60}")
    output.append(f"Цена: ${signal.price:,.2f}")
    output.append(f"Сила сигнала: {signal.strength:+.2f}")
    output.append(f"Stop Loss: ${signal.suggested_stop_loss:,.2f}")
    output.append(f"Take Profit: ${signal.suggested_take_profit:,.2f}")
    output.append(f"\nИндикаторы:")
    output.append(f"  RSI: {signal.indicators.get('rsi', 'N/A'):.1f}" if signal.indicators.get('rsi') else "  RSI: N/A")
    output.append(f"  MACD Histogram: {signal.indicators.get('macd_histogram', 0):+.4f}")
    output.append(f"  EMA21: ${signal.indicators.get('ema_21', 0):,.2f}")
    output.append(f"  ATR: ${signal.indicators.get('atr', 0):,.2f}")
    output.append(f"\nПричины:")
    for reason in signal.reasons:
        output.append(f"  • {reason}")

    return '\n'.join(output)


def main():
    """Основная функция"""
    print("=" * 60)
    print("BOT 5: SIGNAL GENERATOR")
    print("Генерация торговых сигналов")
    print("=" * 60)

    generator = SignalGenerator()

    print(f"\nГенерация сигналов для {len(SYMBOLS)} активов...")
    print(f"Активы: {', '.join(SYMBOLS)}")
    print()

    signals = generator.generate_all_signals('1h')

    # Сортируем по силе сигнала
    signals.sort(key=lambda s: abs(s.strength), reverse=True)

    # Выводим сигналы
    for signal in signals:
        print(format_signal(signal))

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("СВОДКА СИГНАЛОВ")
    print("=" * 60)

    buy_signals = [s for s in signals if s.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]]
    sell_signals = [s for s in signals if s.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]]
    neutral_signals = [s for s in signals if s.signal_type == SignalType.NEUTRAL]

    print(f"🟢 BUY: {len(buy_signals)}")
    print(f"🔴 SELL: {len(sell_signals)}")
    print(f"⚪ NEUTRAL: {len(neutral_signals)}")

    if buy_signals:
        print(f"\nЛучший для покупки: {buy_signals[0].symbol} ({buy_signals[0].strength:+.2f})")
    if sell_signals:
        print(f"Лучший для продажи: {sell_signals[0].symbol} ({sell_signals[0].strength:+.2f})")

    return signals


if __name__ == '__main__':
    main()
