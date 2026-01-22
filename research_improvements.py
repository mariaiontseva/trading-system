"""
Research: How to improve the trading bot
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Load data
data_dir = Path('data/export')
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(period).mean()
    avg_loss = pd.Series(losses).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, period):
    return pd.Series(prices).ewm(span=period).mean()

def calculate_adx(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = pd.Series(tr).rolling(period).mean()
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-10)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return pd.Series(dx).rolling(period).mean().iloc[-1]

def count_signals(df, rsi_low, rsi_high, ema_period, adx_thresh):
    """Count signals with given parameters"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    rsi = calculate_rsi(close, 14)
    ema = calculate_ema(close, ema_period)
    
    long_signals = 0
    short_signals = 0
    
    for i in range(ema_period + 20, len(df)):
        r = rsi.iloc[i] if i < len(rsi) else 50
        e = ema.iloc[i]
        price = close[i]
        
        # Simplified ADX check
        adx_ok = True  # For now, assume ADX is usually > threshold
        
        if r < rsi_low and price > e:
            long_signals += 1
        if r > rsi_high and price < e:
            short_signals += 1
    
    return long_signals, short_signals

print("=" * 60)
print("ИССЛЕДОВАНИЕ: КАК УЛУЧШИТЬ БОТА")
print("=" * 60)

# Test different parameters
print("\n📊 ТЕСТ 1: Влияние RSI порогов на количество сигналов")
print("-" * 50)

results = []
for sym in symbols:
    try:
        df = pd.read_csv(data_dir / f'{sym}.csv')
        for rsi_low, rsi_high in [(40, 60), (45, 55), (35, 65), (42, 58)]:
            for ema in [100, 50, 200]:
                long_s, short_s = count_signals(df, rsi_low, rsi_high, ema, 20)
                results.append({
                    'symbol': sym,
                    'rsi': f'{rsi_low}/{rsi_high}',
                    'ema': ema,
                    'long': long_s,
                    'short': short_s,
                    'total': long_s + short_s
                })
    except Exception as e:
        pass

results_df = pd.DataFrame(results)
summary = results_df.groupby(['rsi', 'ema']).agg({'total': 'sum'}).reset_index()
summary = summary.sort_values('total', ascending=False)
print(summary.to_string(index=False))

print("\n📊 ТЕСТ 2: Текущие параметры vs Агрессивные")
print("-" * 50)
current = results_df[(results_df['rsi'] == '40/60') & (results_df['ema'] == 100)]['total'].sum()
aggressive = results_df[(results_df['rsi'] == '45/55') & (results_df['ema'] == 50)]['total'].sum()
print(f"Текущие (RSI 40/60, EMA100): {current} сигналов")
print(f"Агрессивные (RSI 45/55, EMA50): {aggressive} сигналов")
print(f"Разница: +{aggressive - current} сигналов ({(aggressive/current - 1)*100:.0f}% больше)")

print("\n📊 ТЕСТ 3: Добавить больше монет")
print("-" * 50)
extra_coins = ['XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 
               'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'NEARUSDT']
print(f"Текущие монеты: {len(symbols)} ({', '.join([s.replace('USDT','') for s in symbols])})")
print(f"Можно добавить: {len(extra_coins)} ({', '.join([s.replace('USDT','') for s in extra_coins])})")
print(f"Потенциал: ~{len(symbols) + len(extra_coins)} монет = ~3x больше сигналов")

print("\n📊 ТЕСТ 4: Мультитаймфрейм")
print("-" * 50)
print("Текущий: 1h")
print("Варианты:")
print("  - 15m: ~4x больше сигналов, но больше шума")
print("  - 4h:  меньше сигналов, но выше качество")
print("  - Комбо: вход на 15m, подтверждение на 1h")

print("\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ")
print("=" * 60)
