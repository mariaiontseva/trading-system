"""
Trading Strategies
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional
from loguru import logger


class BaseStrategy(ABC):
    """
    Базовый класс для всех стратегий
    """
    
    name: str = "BaseStrategy"
    warmup_period: int = 100
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Генерация торгового сигнала
        
        Returns:
            {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0.0-1.0,
                'stop_loss': price or None,
                'take_profit': price or None,
                'position_size': 0.0-1.0 (fraction of capital)
            }
        """
        pass
    
    def train(self, data: pd.DataFrame):
        """
        Обучение стратегии (опционально)
        """
        pass


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    
    Покупаем когда цена сильно ниже средней (oversold)
    Продаём когда цена сильно выше средней (overbought)
    """
    
    name = "MeanReversion"
    warmup_period = 50
    
    def __init__(
        self,
        lookback: int = 20,
        z_threshold: float = 2.0,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        if len(data) < self.warmup_period:
            return {'action': 'HOLD'}
        
        close = data['close']
        current_price = close.iloc[-1]
        
        # Z-score
        mean = close.rolling(self.lookback).mean().iloc[-1]
        std = close.rolling(self.lookback).std().iloc[-1]
        z_score = (current_price - mean) / (std + 1e-10)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Сигнал
        signal = {'action': 'HOLD', 'confidence': 0.0}
        
        # Oversold - покупаем
        if z_score < -self.z_threshold and current_rsi < self.rsi_oversold:
            signal = {
                'action': 'BUY',
                'confidence': min(abs(z_score) / 3, 1.0),
                'stop_loss': current_price * (1 - self.stop_loss_pct),
                'take_profit': current_price * (1 + self.take_profit_pct),
                'position_size': 0.02
            }
        
        # Overbought - продаём
        elif z_score > self.z_threshold and current_rsi > self.rsi_overbought:
            signal = {
                'action': 'SELL',
                'confidence': min(abs(z_score) / 3, 1.0),
                'stop_loss': current_price * (1 + self.stop_loss_pct),
                'take_profit': current_price * (1 - self.take_profit_pct),
                'position_size': 0.02
            }
        
        return signal


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy
    
    Следуем за трендом используя EMA crossover и ADX
    """
    
    name = "TrendFollowing"
    warmup_period = 100
    
    def __init__(
        self,
        fast_ema: int = 12,
        slow_ema: int = 26,
        signal_ema: int = 9,
        adx_threshold: float = 25,
        stop_loss_pct: float = 0.025,
        take_profit_pct: float = 0.05
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_ema = signal_ema
        self.adx_threshold = adx_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        if len(data) < self.warmup_period:
            return {'action': 'HOLD'}
        
        close = data['close']
        high = data['high']
        low = data['low']
        current_price = close.iloc[-1]
        
        # EMAs
        ema_fast = close.ewm(span=self.fast_ema).mean()
        ema_slow = close.ewm(span=self.slow_ema).mean()
        
        # MACD
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal_ema).mean()
        
        # ADX
        adx = self._calculate_adx(high, low, close, 14)
        
        # Previous values
        prev_macd = macd.iloc[-2]
        curr_macd = macd.iloc[-1]
        prev_signal = macd_signal.iloc[-2]
        curr_signal = macd_signal.iloc[-1]
        curr_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        
        signal = {'action': 'HOLD', 'confidence': 0.0}
        
        # Bullish crossover
        if prev_macd < prev_signal and curr_macd > curr_signal:
            if curr_adx > self.adx_threshold:  # Strong trend
                signal = {
                    'action': 'BUY',
                    'confidence': min(curr_adx / 50, 1.0),
                    'stop_loss': current_price * (1 - self.stop_loss_pct),
                    'take_profit': current_price * (1 + self.take_profit_pct),
                    'position_size': 0.02
                }
        
        # Bearish crossover
        elif prev_macd > prev_signal and curr_macd < curr_signal:
            if curr_adx > self.adx_threshold:
                signal = {
                    'action': 'SELL',
                    'confidence': min(curr_adx / 50, 1.0),
                    'stop_loss': current_price * (1 + self.stop_loss_pct),
                    'take_profit': current_price * (1 - self.take_profit_pct),
                    'position_size': 0.02
                }
        
        return signal
    
    def _calculate_adx(self, high, low, close, period=14):
        """
        Calculate ADX indicator
        """
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx


class CombinedStrategy(BaseStrategy):
    """
    Combined Strategy
    
    Комбинирует несколько стратегий для более надёжных сигналов
    """
    
    name = "Combined"
    warmup_period = 100
    
    def __init__(self, min_agreement: int = 2):
        self.strategies = [
            MeanReversionStrategy(),
            TrendFollowingStrategy()
        ]
        self.min_agreement = min_agreement
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        if len(data) < self.warmup_period:
            return {'action': 'HOLD'}
        
        signals = []
        for strategy in self.strategies:
            sig = strategy.generate_signal(data)
            signals.append(sig)
        
        # Count votes
        buy_votes = sum(1 for s in signals if s['action'] == 'BUY')
        sell_votes = sum(1 for s in signals if s['action'] == 'SELL')
        
        current_price = data['close'].iloc[-1]
        
        if buy_votes >= self.min_agreement:
            avg_confidence = np.mean([s['confidence'] for s in signals if s['action'] == 'BUY'])
            return {
                'action': 'BUY',
                'confidence': avg_confidence,
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.04,
                'position_size': 0.02
            }
        
        elif sell_votes >= self.min_agreement:
            avg_confidence = np.mean([s['confidence'] for s in signals if s['action'] == 'SELL'])
            return {
                'action': 'SELL',
                'confidence': avg_confidence,
                'stop_loss': current_price * 1.02,
                'take_profit': current_price * 0.96,
                'position_size': 0.02
            }
        
        return {'action': 'HOLD', 'confidence': 0.0}


# Registry
STRATEGIES = {
    'mean_reversion': MeanReversionStrategy,
    'trend_following': TrendFollowingStrategy,
    'combined': CombinedStrategy
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    Получить стратегию по имени
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)
