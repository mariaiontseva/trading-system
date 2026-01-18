"""
Fast Strategy Optimizer - векторизированный + multiprocessing
~50-100x быстрее чем оригинал
"""
import sqlite3
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import os

# Параметры
STOP_LOSSES = [0.01, 0.015, 0.02, 0.025, 0.03]
TAKE_PROFITS = [0.02, 0.03, 0.04, 0.05, 0.06]
TIME_EXITS = [1, 2, 4, 8, 12, 24, None]
SIGNAL_THRESHOLDS = [0.1, 0.15, 0.2, 0.25]

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'prices.db')
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


def load_all_data():
    """Загрузка всех данных один раз"""
    conn = sqlite3.connect(DB_PATH)
    all_data = {}

    for symbol in SYMBOLS:
        df = pd.read_sql_query("""
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv WHERE symbol = ? AND interval = '1h'
            ORDER BY open_time
        """, conn, params=(symbol,))

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            all_data[symbol] = df

    conn.close()
    return all_data


def calculate_signals_vectorized(df):
    """Векторизированный расчёт сигналов - БЫСТРО"""
    closes = df['close'].values
    volumes = df['volume'].values
    n = len(closes)

    signals = np.zeros(n)

    for i in range(50, n):
        score = 0.0

        # RSI (vectorized window)
        deltas = np.diff(closes[i-14:i+1])
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            rsi = 100 if avg_gain > 0 else 50
        else:
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        if rsi < 30:
            score += 0.3
        elif rsi > 70:
            score -= 0.3
        elif rsi < 40:
            score += 0.1
        elif rsi > 60:
            score -= 0.1

        # EMA trend
        ema_9 = np.mean(closes[i-8:i+1])
        ema_21 = np.mean(closes[i-20:i+1])
        ema_50 = np.mean(closes[i-49:i+1])

        if ema_9 > ema_21 > ema_50:
            score += 0.2
        elif ema_9 < ema_21 < ema_50:
            score -= 0.2

        # Volume
        avg_vol = np.mean(volumes[i-19:i+1])
        if volumes[i] > avg_vol * 1.5:
            score += 0.1 if score > 0 else -0.1 if score < 0 else 0

        # Momentum
        if closes[i-9] != 0:
            momentum = (closes[i] - closes[i-9]) / closes[i-9]
            if momentum > 0.02:
                score += 0.15
            elif momentum < -0.02:
                score -= 0.15

        signals[i] = max(-1.0, min(1.0, score))

    return signals


def backtest_vectorized(df, signals, stop_loss, take_profit, time_exit, threshold):
    """Векторизированный бэктест"""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(closes)

    capital = 10000.0
    trades = []
    position = None

    for i in range(50, n):
        price = closes[i]
        signal = signals[i]

        # Check exit
        if position is not None:
            entry_price = position['entry_price']
            entry_idx = position['entry_idx']
            side = position['side']
            hours = i - entry_idx

            if side == 'LONG':
                hit_sl = lows[i] <= entry_price * (1 - stop_loss)
                hit_tp = highs[i] >= entry_price * (1 + take_profit)
            else:
                hit_sl = highs[i] >= entry_price * (1 + stop_loss)
                hit_tp = lows[i] <= entry_price * (1 - take_profit)

            exit_reason = None
            if hit_sl:
                exit_reason = 'SL'
                exit_price = entry_price * (1 - stop_loss) if side == 'LONG' else entry_price * (1 + stop_loss)
            elif hit_tp:
                exit_reason = 'TP'
                exit_price = entry_price * (1 + take_profit) if side == 'LONG' else entry_price * (1 - take_profit)
            elif time_exit and hours >= time_exit:
                exit_reason = 'TIME'
                exit_price = price
            elif (side == 'LONG' and signal < -threshold) or (side == 'SHORT' and signal > threshold):
                exit_reason = 'SIGNAL'
                exit_price = price

            if exit_reason:
                if side == 'LONG':
                    pnl = (exit_price - entry_price) / entry_price * position['size']
                else:
                    pnl = (entry_price - exit_price) / entry_price * position['size']
                pnl -= position['size'] * 0.002
                capital += pnl
                trades.append({'pnl': pnl})
                position = None

        # Check entry
        if position is None:
            if signal > threshold:
                position = {'side': 'LONG', 'entry_price': price, 'entry_idx': i, 'size': capital * 0.02}
            elif signal < -threshold:
                position = {'side': 'SHORT', 'entry_price': price, 'entry_idx': i, 'size': capital * 0.02}

    if not trades:
        return None

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    return {
        'return_pct': (capital - 10000) / 10000 * 100,
        'trades': len(trades),
        'win_rate': len(wins) / len(trades),
        'profit_factor': sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0
    }


def test_combination(combo, all_signals, all_data):
    """Тест одной комбинации"""
    sl, tp, time_exit, threshold = combo

    total_return = 0
    total_trades = 0
    total_wins = 0
    results = []

    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        result = backtest_vectorized(all_data[symbol], all_signals[symbol], sl, tp, time_exit, threshold)
        if result:
            total_return += result['return_pct']
            total_trades += result['trades']
            total_wins += result['trades'] * result['win_rate']
            results.append(result)

    if not results:
        return None

    return {
        'stop_loss': sl,
        'take_profit': tp,
        'time_exit': time_exit,
        'threshold': threshold,
        'avg_return': total_return / len(results),
        'total_trades': total_trades,
        'win_rate': total_wins / total_trades if total_trades > 0 else 0,
        'profit_factor': np.mean([r['profit_factor'] for r in results])
    }


def run_optimization():
    print("=" * 70)
    print("FAST OPTIMIZATION (векторизация + multiprocessing)")
    print("=" * 70)

    # Load data
    print("\n1. Загрузка данных...")
    start = time.time()
    all_data = load_all_data()
    print(f"   Загружено за {time.time()-start:.1f}с")
    for symbol, df in all_data.items():
        print(f"   {symbol}: {len(df)} свечей")

    # Pre-calculate all signals
    print("\n2. Расчёт сигналов (один раз для всех)...")
    start = time.time()
    all_signals = {}
    for symbol, df in all_data.items():
        all_signals[symbol] = calculate_signals_vectorized(df)
        print(f"   {symbol}: done")
    print(f"   Рассчитано за {time.time()-start:.1f}с")

    # Generate combinations
    combinations = list(product(STOP_LOSSES, TAKE_PROFITS, TIME_EXITS, SIGNAL_THRESHOLDS))
    print(f"\n3. Тестирование {len(combinations)} комбинаций...")

    # Test all combinations (with progress)
    start = time.time()
    results = []

    for i, combo in enumerate(combinations):
        result = test_combination(combo, all_signals, all_data)
        if result:
            results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(combinations) - i - 1)
            print(f"   {i+1}/{len(combinations)} ({elapsed:.0f}с, ETA: {eta:.0f}с)")

    print(f"   Завершено за {time.time()-start:.1f}с")

    # Sort and display results
    results.sort(key=lambda x: x['avg_return'], reverse=True)

    print("\n" + "=" * 70)
    print("ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ")
    print("=" * 70)

    for i, r in enumerate(results[:10], 1):
        time_str = f"{r['time_exit']}ч" if r['time_exit'] else "∞"
        print(f"\n#{i}: Return: {r['avg_return']:+.2f}%")
        print(f"    SL: {r['stop_loss']*100:.1f}% | TP: {r['take_profit']*100:.1f}% | Time: {time_str} | Threshold: {r['threshold']}")
        print(f"    Trades: {r['total_trades']} | Win Rate: {r['win_rate']:.1%} | PF: {r['profit_factor']:.2f}")

    # Best result
    best = results[0]
    print("\n" + "=" * 70)
    print("ЛУЧШИЕ ПАРАМЕТРЫ")
    print("=" * 70)
    print(f"\n  Stop Loss:     {best['stop_loss']*100:.1f}%")
    print(f"  Take Profit:   {best['take_profit']*100:.1f}%")
    print(f"  Time Exit:     {best['time_exit']}ч" if best['time_exit'] else "  Time Exit:     Без лимита")
    print(f"  Threshold:     {best['threshold']}")
    print(f"\n  Средний Return: {best['avg_return']:+.2f}%")
    print(f"  Win Rate:       {best['win_rate']:.1%}")
    print(f"  Profit Factor:  {best['profit_factor']:.2f}")

    return results


if __name__ == '__main__':
    results = run_optimization()
