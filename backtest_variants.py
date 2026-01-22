"""
Backtest different parameter variants
"""
import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path('data/export')
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

def calculate_indicators(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # RSI
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).ewm(span=14).mean()
    avg_loss = pd.Series(losses).ewm(span=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = pd.Series(tr).rolling(14).mean()
    
    # ADX (simplified)
    plus_dm = np.maximum(high[1:] - high[:-1], 0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0)
    plus_di = pd.Series(plus_dm).rolling(14).mean() / (pd.Series(tr).rolling(14).mean() + 1e-10) * 100
    minus_di = pd.Series(minus_dm).rolling(14).mean() / (pd.Series(tr).rolling(14).mean() + 1e-10) * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).rolling(14).mean()
    
    return rsi, atr, adx

def backtest(df, rsi_low, rsi_high, ema_period, adx_thresh, sl_mult, tp_mult):
    close = df['close'].values
    rsi, atr, adx = calculate_indicators(df)
    ema = pd.Series(close).ewm(span=ema_period).mean().values
    
    trades = []
    position = None
    
    for i in range(max(ema_period, 20) + 1, len(df) - 1):
        r = rsi.iloc[i-1] if i-1 < len(rsi) else 50
        a = adx.iloc[i-1] if i-1 < len(adx) else 0
        at = atr.iloc[i-1] if i-1 < len(atr) else 0
        e = ema[i]
        price = close[i]
        
        # Check exits
        if position:
            if position['side'] == 'LONG':
                if price >= position['tp']:
                    trades.append({'pnl': (position['tp'] - position['entry']) / position['entry'] * 100, 'reason': 'TP'})
                    position = None
                elif price <= position['sl']:
                    trades.append({'pnl': (position['sl'] - position['entry']) / position['entry'] * 100, 'reason': 'SL'})
                    position = None
            else:
                if price <= position['tp']:
                    trades.append({'pnl': (position['entry'] - position['tp']) / position['entry'] * 100, 'reason': 'TP'})
                    position = None
                elif price >= position['sl']:
                    trades.append({'pnl': (position['entry'] - position['sl']) / position['entry'] * 100, 'reason': 'SL'})
                    position = None
        
        # Check entries
        if not position and a > adx_thresh and at > 0:
            if r < rsi_low and price > e:
                position = {
                    'side': 'LONG',
                    'entry': price,
                    'sl': price - sl_mult * at,
                    'tp': price + tp_mult * at
                }
            elif r > rsi_high and price < e:
                position = {
                    'side': 'SHORT',
                    'entry': price,
                    'sl': price + sl_mult * at,
                    'tp': price - tp_mult * at
                }
    
    return trades

# Test variants
variants = [
    {'name': 'Текущий', 'rsi': (40, 60), 'ema': 100, 'adx': 20, 'sl': 2.0, 'tp': 4.0},
    {'name': 'RSI 45/55', 'rsi': (45, 55), 'ema': 100, 'adx': 20, 'sl': 2.0, 'tp': 4.0},
    {'name': 'EMA 50', 'rsi': (40, 60), 'ema': 50, 'adx': 20, 'sl': 2.0, 'tp': 4.0},
    {'name': 'ADX 15', 'rsi': (40, 60), 'ema': 100, 'adx': 15, 'sl': 2.0, 'tp': 4.0},
    {'name': 'Агрессивный', 'rsi': (45, 55), 'ema': 50, 'adx': 15, 'sl': 1.5, 'tp': 3.0},
    {'name': 'Консервативный', 'rsi': (35, 65), 'ema': 100, 'adx': 25, 'sl': 2.5, 'tp': 5.0},
]

print("\n" + "=" * 80)
print("БЭКТЕСТ РАЗНЫХ ВАРИАНТОВ (2 года данных)")
print("=" * 80)

results = []
for var in variants:
    all_trades = []
    for sym in symbols:
        try:
            df = pd.read_csv(data_dir / f'{sym}.csv')
            trades = backtest(df, var['rsi'][0], var['rsi'][1], var['ema'], var['adx'], var['sl'], var['tp'])
            all_trades.extend(trades)
        except:
            pass
    
    if all_trades:
        wins = [t for t in all_trades if t['pnl'] > 0]
        losses = [t for t in all_trades if t['pnl'] < 0]
        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))
        pf = total_profit / total_loss if total_loss > 0 else 0
        wr = len(wins) / len(all_trades) * 100 if all_trades else 0
        avg_pnl = np.mean([t['pnl'] for t in all_trades])
        
        results.append({
            'variant': var['name'],
            'trades': len(all_trades),
            'win_rate': f'{wr:.1f}%',
            'profit_factor': f'{pf:.2f}',
            'avg_trade': f'{avg_pnl:.2f}%',
            'score': len(all_trades) * pf  # Combined score
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('score', ascending=False)
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("🏆 ЛУЧШИЙ ВАРИАНТ: " + results_df.iloc[0]['variant'])
print("=" * 80)
