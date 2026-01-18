"""
TURBO OPTIMIZER - Numba JIT compiled (50-100x faster)
"""
import sqlite3
import numpy as np
import pandas as pd
from numba import jit, prange
from multiprocessing import Pool, cpu_count
import time
import os
import warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'prices.db')
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


# ============================================================
# NUMBA JIT COMPILED FUNCTIONS (C-speed)
# ============================================================

@jit(nopython=True, cache=True)
def calc_ema_fast(prices, period):
    n = len(prices)
    ema = np.empty(n)
    ema[0] = prices[0]
    mult = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = prices[i] * mult + ema[i-1] * (1 - mult)
    return ema


@jit(nopython=True, cache=True)
def calc_rsi_fast(prices, period=14):
    n = len(prices)
    rsi = np.full(n, 50.0)

    if n < period + 1:
        return rsi

    gains = np.zeros(n-1)
    losses = np.zeros(n-1)

    for i in range(n-1):
        diff = prices[i+1] - prices[i]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n-1):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i+1] = 100.0
        else:
            rsi[i+1] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    return rsi


@jit(nopython=True, cache=True)
def calc_atr_fast(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

    return atr


@jit(nopython=True, cache=True)
def donchian_signals(high, low, close, entry_period, exit_period):
    n = len(close)
    signals = np.zeros(n)

    for i in range(entry_period, n):
        upper = np.max(high[i-entry_period:i])
        lower = np.min(low[max(0, i-exit_period):i])

        if close[i] > upper:
            signals[i] = 1.0
        elif close[i] < lower:
            signals[i] = -1.0

    return signals


@jit(nopython=True, cache=True)
def ema_cross_signals(close, fast_period, slow_period):
    n = len(close)
    signals = np.zeros(n)

    ema_fast = calc_ema_fast(close, fast_period)
    ema_slow = calc_ema_fast(close, slow_period)

    for i in range(slow_period + 1, n):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            signals[i] = 1.0
        elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
            signals[i] = -1.0

    return signals


@jit(nopython=True, cache=True)
def rsi_trend_signals(close, rsi_low, rsi_high, trend_period):
    n = len(close)
    signals = np.zeros(n)

    rsi = calc_rsi_fast(close, 14)
    ema_trend = calc_ema_fast(close, trend_period)

    for i in range(trend_period, n):
        if close[i] > ema_trend[i]:  # Uptrend
            if rsi[i] < rsi_low:
                signals[i] = 1.0
        else:  # Downtrend
            if rsi[i] > rsi_high:
                signals[i] = -1.0

    return signals


@jit(nopython=True, cache=True)
def backtest_core(close, high, low, signals, atr, stop_mult, target_mult, max_hours):
    n = len(close)
    capital = 10000.0
    position_side = 0  # 0=none, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    stop_price = 0.0
    target_price = 0.0
    position_size = 0.0

    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0

    for i in range(50, n):
        price = close[i]

        # Check exit
        if position_side != 0:
            hours = i - entry_idx
            should_exit = False
            exit_price = price

            if position_side == 1:  # LONG
                if low[i] <= stop_price:
                    exit_price = stop_price
                    should_exit = True
                elif high[i] >= target_price:
                    exit_price = target_price
                    should_exit = True
                elif max_hours > 0 and hours >= max_hours:
                    should_exit = True
                elif signals[i] == -1:
                    should_exit = True
            else:  # SHORT
                if high[i] >= stop_price:
                    exit_price = stop_price
                    should_exit = True
                elif low[i] <= target_price:
                    exit_price = target_price
                    should_exit = True
                elif max_hours > 0 and hours >= max_hours:
                    should_exit = True
                elif signals[i] == 1:
                    should_exit = True

            if should_exit:
                if position_side == 1:
                    pnl = (exit_price - entry_price) / entry_price * position_size
                else:
                    pnl = (entry_price - exit_price) / entry_price * position_size

                pnl -= position_size * 0.002  # Commission
                capital += pnl

                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)

                position_side = 0

        # Check entry
        if position_side == 0 and signals[i] != 0:
            current_atr = atr[i] if atr[i] > 0 else price * 0.02

            if signals[i] == 1:  # LONG
                position_side = 1
                entry_price = price
                entry_idx = i
                stop_price = price - current_atr * stop_mult
                target_price = price + current_atr * target_mult
                position_size = capital * 0.02
            elif signals[i] == -1:  # SHORT
                position_side = -1
                entry_price = price
                entry_idx = i
                stop_price = price + current_atr * stop_mult
                target_price = price - current_atr * target_mult
                position_size = capital * 0.02

    total_trades = wins + losses
    if total_trades == 0:
        return 0.0, 0, 0.0, 0.0

    win_rate = wins / total_trades
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    return_pct = (capital - 10000.0) / 10000.0 * 100.0

    return return_pct, total_trades, win_rate, profit_factor


# ============================================================
# MAIN OPTIMIZER
# ============================================================

def load_data():
    conn = sqlite3.connect(DB_PATH)
    data = {}
    for symbol in SYMBOLS:
        df = pd.read_sql_query("""
            SELECT open, high, low, close, volume FROM ohlcv
            WHERE symbol = ? AND interval = '1h' ORDER BY open_time
        """, conn, params=(symbol,))
        if not df.empty:
            data[symbol] = {
                'close': df['close'].values.astype(np.float64),
                'high': df['high'].values.astype(np.float64),
                'low': df['low'].values.astype(np.float64)
            }
    conn.close()
    return data


def test_donchian(data, entry, exit_p, stop, target, hours):
    results = []
    for symbol, d in data.items():
        signals = donchian_signals(d['high'], d['low'], d['close'], entry, exit_p)
        atr = calc_atr_fast(d['high'], d['low'], d['close'], 14)
        ret, trades, wr, pf = backtest_core(d['close'], d['high'], d['low'], signals, atr, stop, target, hours)
        if trades > 0:
            results.append((ret, trades, wr, pf))

    if not results:
        return None

    return {
        'strategy': 'donchian',
        'params': f"entry={entry}, exit={exit_p}, stop={stop}, target={target}, hours={hours}",
        'return': np.mean([r[0] for r in results]),
        'trades': sum(r[1] for r in results),
        'win_rate': np.mean([r[2] for r in results]),
        'pf': np.mean([r[3] for r in results])
    }


def test_ema(data, fast, slow, stop, target, hours):
    results = []
    for symbol, d in data.items():
        signals = ema_cross_signals(d['close'], fast, slow)
        atr = calc_atr_fast(d['high'], d['low'], d['close'], 14)
        ret, trades, wr, pf = backtest_core(d['close'], d['high'], d['low'], signals, atr, stop, target, hours)
        if trades > 0:
            results.append((ret, trades, wr, pf))

    if not results:
        return None

    return {
        'strategy': 'ema_cross',
        'params': f"fast={fast}, slow={slow}, stop={stop}, target={target}, hours={hours}",
        'return': np.mean([r[0] for r in results]),
        'trades': sum(r[1] for r in results),
        'win_rate': np.mean([r[2] for r in results]),
        'pf': np.mean([r[3] for r in results])
    }


def test_rsi(data, rsi_low, rsi_high, trend, stop, target, hours):
    results = []
    for symbol, d in data.items():
        signals = rsi_trend_signals(d['close'], rsi_low, rsi_high, trend)
        atr = calc_atr_fast(d['high'], d['low'], d['close'], 14)
        ret, trades, wr, pf = backtest_core(d['close'], d['high'], d['low'], signals, atr, stop, target, hours)
        if trades > 0:
            results.append((ret, trades, wr, pf))

    if not results:
        return None

    return {
        'strategy': 'rsi_trend',
        'params': f"low={rsi_low}, high={rsi_high}, trend={trend}, stop={stop}, target={target}, hours={hours}",
        'return': np.mean([r[0] for r in results]),
        'trades': sum(r[1] for r in results),
        'win_rate': np.mean([r[2] for r in results]),
        'pf': np.mean([r[3] for r in results])
    }


def run():
    print("=" * 70)
    print("TURBO OPTIMIZER (Numba JIT - C-speed)")
    print("=" * 70)

    print("\n1. Loading data...")
    data = load_data()
    for s, d in data.items():
        print(f"   {s}: {len(d['close']):,} candles")

    print("\n2. Warming up JIT (first run slower)...")
    start = time.time()
    # Warm up JIT compilation
    test_donchian(data, 20, 10, 2.0, 3.0, 24)
    test_ema(data, 9, 21, 2.0, 3.0, 24)
    test_rsi(data, 30, 70, 50, 2.0, 3.0, 24)
    print(f"   JIT compiled in {time.time()-start:.1f}s")

    print("\n3. Running optimization...")
    start = time.time()

    results = []
    tested = 0

    # Donchian
    for entry in [10, 20, 30, 55]:
        for exit_p in [5, 10, 20]:
            for stop in [1.5, 2.0, 3.0]:
                for target in [2.0, 3.0, 4.0, 6.0]:
                    for hours in [12, 24, 48, 0]:  # 0 = no limit
                        r = test_donchian(data, entry, exit_p, stop, target, hours)
                        if r:
                            results.append(r)
                        tested += 1

    # EMA crossover
    for fast in [5, 9, 12, 20]:
        for slow in [21, 50, 100, 200]:
            if fast >= slow:
                continue
            for stop in [1.5, 2.0, 3.0]:
                for target in [2.0, 3.0, 4.0, 6.0]:
                    for hours in [12, 24, 48, 0]:
                        r = test_ema(data, fast, slow, stop, target, hours)
                        if r:
                            results.append(r)
                        tested += 1

    # RSI + Trend
    for rsi_low in [25, 30, 35]:
        for rsi_high in [65, 70, 75]:
            for trend in [50, 100, 200]:
                for stop in [1.5, 2.0, 3.0]:
                    for target in [2.0, 3.0, 4.0]:
                        for hours in [12, 24, 48, 0]:
                            r = test_rsi(data, rsi_low, rsi_high, trend, stop, target, hours)
                            if r:
                                results.append(r)
                            tested += 1

    elapsed = time.time() - start
    print(f"   Tested {tested} combinations in {elapsed:.1f}s ({tested/elapsed:.0f}/sec)")

    # Sort by profit factor
    results.sort(key=lambda x: x['pf'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 15 BY PROFIT FACTOR")
    print("=" * 70)

    for i, r in enumerate(results[:15], 1):
        print(f"\n#{i}: {r['strategy'].upper()}")
        print(f"    PF: {r['pf']:.2f} | Win: {r['win_rate']:.1%} | Return: {r['return']:+.1f}%")
        print(f"    Trades: {r['trades']} | {r['params']}")

    # Sort by win rate
    results.sort(key=lambda x: x['win_rate'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 15 BY WIN RATE")
    print("=" * 70)

    for i, r in enumerate(results[:15], 1):
        print(f"\n#{i}: {r['strategy'].upper()}")
        print(f"    Win: {r['win_rate']:.1%} | PF: {r['pf']:.2f} | Return: {r['return']:+.1f}%")
        print(f"    Trades: {r['trades']} | {r['params']}")

    # Best balanced
    good = [r for r in results if r['pf'] > 1.2 and r['win_rate'] > 0.45 and r['return'] > 0]
    if good:
        good.sort(key=lambda x: x['return'], reverse=True)
        print("\n" + "=" * 70)
        print("🏆 PROFITABLE STRATEGIES (PF>1.2, WinRate>45%, Return>0)")
        print("=" * 70)

        for i, r in enumerate(good[:10], 1):
            print(f"\n#{i}: {r['strategy'].upper()}")
            print(f"    Return: {r['return']:+.1f}% | Win: {r['win_rate']:.1%} | PF: {r['pf']:.2f}")
            print(f"    Trades: {r['trades']} | {r['params']}")
    else:
        print("\n⚠️  No profitable strategies found with PF>1.2 and WinRate>45%")

    return results


if __name__ == '__main__':
    run()
