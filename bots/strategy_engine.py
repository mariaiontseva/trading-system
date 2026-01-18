"""
Bot 7: Strategy Engine
Объединение всех сигналов в торговые решения

Скоринг:
- Signal Generator: 30%
- Pattern Detector: 20%
- Volume Analyzer: 20%
- Sentiment: 15%
- Whale Activity: 15%
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Импорт других ботов
from signal_generator import SignalGenerator, SignalType
from volume_analyzer import VolumeAnalyzer
from pattern_detector import PatternDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StrategyEngine')

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


class TradeDecision(Enum):
    """Торговое решение"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class StrategySignal:
    """Финальный сигнал стратегии"""
    symbol: str
    decision: TradeDecision
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    components: Dict[str, float]
    entry_price: float
    stop_loss: float
    take_profit: float
    reasons: List[str]
    timestamp: datetime


class StrategyEngine:
    """Стратегический движок"""

    # Веса компонентов
    WEIGHTS = {
        'signal_generator': 0.35,
        'volume_analyzer': 0.25,
        'pattern_detector': 0.25,
        'sentiment': 0.15,  # Заглушка пока
    }

    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.volume_analyzer = VolumeAnalyzer()
        self.pattern_detector = PatternDetector()

        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'strategy.db')
        self._init_database()

    def _init_database(self):
        """Инициализация базы"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                decision TEXT NOT NULL,
                score REAL,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                reasons TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def analyze(self, symbol: str, interval: str = '1h') -> Optional[StrategySignal]:
        """Анализ символа и генерация сигнала"""

        # Получаем сигналы от всех ботов
        tech_signal = self.signal_generator.generate_signal(symbol, interval)
        volume_signal = self.volume_analyzer.analyze(symbol, interval)
        pattern_signal = self.pattern_detector.analyze(symbol, interval)

        if not tech_signal:
            return None

        components = {}
        reasons = []

        # 1. Signal Generator (технический анализ)
        tech_score = tech_signal.strength
        components['signal_generator'] = tech_score
        if tech_signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            reasons.append(f"Tech: BUY ({tech_score:+.2f})")
        elif tech_signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            reasons.append(f"Tech: SELL ({tech_score:+.2f})")

        # 2. Volume Analyzer
        vol_score = 0.0
        if volume_signal:
            vol_score = volume_signal.strength
            components['volume_analyzer'] = vol_score
            if volume_signal.signal_type == 'accumulation':
                reasons.append(f"Volume: Accumulation ({vol_score:+.2f})")
            elif volume_signal.signal_type == 'distribution':
                reasons.append(f"Volume: Distribution ({vol_score:+.2f})")

        # 3. Pattern Detector
        pattern_score = 0.0
        if pattern_signal:
            if pattern_signal.overall_signal == 'bullish':
                pattern_score = pattern_signal.strength
            elif pattern_signal.overall_signal == 'bearish':
                pattern_score = -pattern_signal.strength
            components['pattern_detector'] = pattern_score
            if pattern_signal.patterns:
                patterns = ', '.join([p.name for p in pattern_signal.patterns[:2]])
                reasons.append(f"Patterns: {patterns}")

        # 4. Sentiment (заглушка - нейтральный)
        sentiment_score = 0.0
        components['sentiment'] = sentiment_score

        # Расчет взвешенного score
        total_score = (
            tech_score * self.WEIGHTS['signal_generator'] +
            vol_score * self.WEIGHTS['volume_analyzer'] +
            pattern_score * self.WEIGHTS['pattern_detector'] +
            sentiment_score * self.WEIGHTS['sentiment']
        )

        # Нормализация
        total_score = max(-1, min(1, total_score))

        # Расчет confidence (чем больше компонентов согласны, тем выше)
        signs = [1 if s > 0 else -1 if s < 0 else 0 for s in components.values()]
        agreement = abs(sum(signs)) / len(signs) if signs else 0
        confidence = agreement * abs(total_score)

        # Определение решения
        if total_score >= 0.5:
            decision = TradeDecision.STRONG_BUY
        elif total_score >= 0.2:
            decision = TradeDecision.BUY
        elif total_score <= -0.5:
            decision = TradeDecision.STRONG_SELL
        elif total_score <= -0.2:
            decision = TradeDecision.SELL
        else:
            decision = TradeDecision.HOLD

        signal = StrategySignal(
            symbol=symbol,
            decision=decision,
            score=round(total_score, 3),
            confidence=round(confidence, 3),
            components=components,
            entry_price=tech_signal.price,
            stop_loss=tech_signal.suggested_stop_loss,
            take_profit=tech_signal.suggested_take_profit,
            reasons=reasons,
            timestamp=datetime.now()
        )

        # Сохраняем в базу
        self._save_signal(signal)

        return signal

    def _save_signal(self, signal: StrategySignal):
        """Сохранение сигнала"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO strategy_signals (symbol, decision, score, confidence,
                                         entry_price, stop_loss, take_profit, reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol,
            signal.decision.name,
            signal.score,
            signal.confidence,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            '|'.join(signal.reasons)
        ))
        conn.commit()
        conn.close()

    def analyze_all(self, interval: str = '1h') -> List[StrategySignal]:
        """Анализ всех символов"""
        signals = []
        for symbol in SYMBOLS:
            signal = self.analyze(symbol, interval)
            if signal:
                signals.append(signal)
        return signals

    def get_top_opportunities(self, interval: str = '1h', top_n: int = 3) -> List[StrategySignal]:
        """Получение лучших возможностей"""
        signals = self.analyze_all(interval)
        # Сортируем по абсолютному score и confidence
        signals.sort(key=lambda s: abs(s.score) * s.confidence, reverse=True)
        return signals[:top_n]


def main():
    print("=" * 60)
    print("BOT 7: STRATEGY ENGINE")
    print("Объединение сигналов")
    print("=" * 60)

    engine = StrategyEngine()
    signals = engine.analyze_all()

    # Сортируем по score
    signals.sort(key=lambda s: s.score, reverse=True)

    for sig in signals:
        if sig.decision == TradeDecision.STRONG_BUY:
            icon = "🟢🟢"
        elif sig.decision == TradeDecision.BUY:
            icon = "🟢"
        elif sig.decision == TradeDecision.STRONG_SELL:
            icon = "🔴🔴"
        elif sig.decision == TradeDecision.SELL:
            icon = "🔴"
        else:
            icon = "⚪"

        print(f"\n{icon} {sig.symbol}: {sig.decision.name}")
        print(f"   Score: {sig.score:+.3f} (confidence: {sig.confidence:.1%})")
        print(f"   Entry: ${sig.entry_price:,.2f}")
        print(f"   SL: ${sig.stop_loss:,.2f} | TP: ${sig.take_profit:,.2f}")
        print(f"   Components:")
        for comp, val in sig.components.items():
            print(f"     • {comp}: {val:+.2f}")
        print(f"   Reasons: {', '.join(sig.reasons)}")

    # Лучшие возможности
    print("\n" + "=" * 60)
    print("TOP OPPORTUNITIES")
    print("=" * 60)
    top = engine.get_top_opportunities(top_n=2)
    for sig in top:
        action = "BUY" if sig.score > 0 else "SELL"
        print(f"  {sig.symbol}: {action} (score: {sig.score:+.3f})")


if __name__ == '__main__':
    main()
