"""
Test variants to increase trade count while maintaining quality
"""
import sqlite3
import numpy as np
import pandas as pd
from numba import jit
import time
import os
import warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'prices.db')

# Base 5 symbols
SYMBOLS_BASE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

# Extended 10 symbols (Variant D)
SYMBOLS_EXTENDED = SYMBOLS_BASE + ['XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT']


# ============================================================
# NUMBA JIT COMPILED INDICATORS (same as before)
# ============================================================

@jit(nopython=True, cache=True)
def calc_ema(prices, period):
    n = len(prices)
    ema = np.empty(n)
    ema[0] = prices[0]
    mult = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = prices[i] * mult + ema[i-1] * (1 - mult)
    return ema


@jit(nopython=True, cache=True)
def calc_rsi(prices, period=14):
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
def calc_atr(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

    return atr


@jit(nopython=True, cache=True)
def calc_adx(high, low, close, period=14):
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr = calc_atr(high, low, close, period)
    atr_safe = np.where(atr == 0, 0.0001, atr)

    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)
    smooth_plus[period] = np.sum(plus_dm[1:period+1])
    smooth_minus[period] = np.sum(minus_dm[1:period+1])

    for i in range(period+1, n):
        smooth_plus[i] = smooth_plus[i-1] - smooth_plus[i-1]/period + plus_dm[i]
        smooth_minus[i] = smooth_minus[i-1] - smooth_minus[i-1]/period + minus_dm[i]

    plus_di = 100 * smooth_plus / (atr_safe * period)
    minus_di = 100 * smooth_minus / (atr_safe * period)

    dx = np.zeros(n)
    for i in range(period, n):
        sum_di = plus_di[i] + minus_di[i]
        if sum_di > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / sum_di

    adx = np.zeros(n)
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period+1, n):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

    return adx


# ============================================================
# SIGNAL GENERATION
# ============================================================

@jit(nopython=True, cache=True)
def generate_signals(close, high, low, rsi_oversold, rsi_overbought,
                     trend_period, adx_threshold):
    """Generate signals with configurable parameters"""
    n = len(close)
    signals = np.zeros(n)

    rsi = calc_rsi(close, 14)
    ema_trend = calc_ema(close, trend_period)
    atr = calc_atr(high, low, close, 14)
    adx = calc_adx(high, low, close, 14)

    for i in range(max(trend_period, 30), n):
        # ADX filter (skip if threshold is 0 = disabled)
        if adx_threshold > 0 and adx[i] < adx_threshold:
            continue

        price = close[i]
        trend_ema = ema_trend[i]

        # LONG: RSI oversold + uptrend
        if rsi[i] < rsi_oversold and price > trend_ema:
            signals[i] = 1.0

        # SHORT: RSI overbought + downtrend
        elif rsi[i] > rsi_overbought and price < trend_ema:
            signals[i] = -1.0

    return signals, atr


# ============================================================
# BACKTEST ENGINE
# ============================================================

@jit(nopython=True, cache=True)
def backtest(close, high, low, signals, atr,
             stop_mult=2.0, target_mult=4.0,
             trail_activate=1.5, trail_distance=1.0,
             commission=0.001):
    """Backtest with trailing stop"""
    n = len(close)
    capital = 10000.0
    position_side = 0
    entry_price = 0.0
    entry_idx = 0
    entry_atr = 0.0
    stop_price = 0.0
    target_price = 0.0
    trailing_active = False
    trailing_stop = 0.0

    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0

    for i in range(50, n):
        price = close[i]

        if position_side != 0:
            if trailing_active:
                if position_side == 1:
                    new_trail = price - entry_atr * trail_distance
                    if new_trail > trailing_stop:
                        trailing_stop = new_trail
                else:
                    new_trail = price + entry_atr * trail_distance
                    if new_trail < trailing_stop:
                        trailing_stop = new_trail

            should_exit = False
            exit_price = price

            if position_side == 1:
                if trailing_active and price <= trailing_stop:
                    should_exit = True
                    exit_price = trailing_stop
                elif price <= stop_price:
                    should_exit = True
                    exit_price = stop_price
                elif price >= target_price:
                    should_exit = True
                    exit_price = target_price
                elif not trailing_active and price >= entry_price + entry_atr * trail_activate:
                    trailing_active = True
                    trailing_stop = price - entry_atr * trail_distance
            else:
                if trailing_active and price >= trailing_stop:
                    should_exit = True
                    exit_price = trailing_stop
                elif price >= stop_price:
                    should_exit = True
                    exit_price = stop_price
                elif price <= target_price:
                    should_exit = True
                    exit_price = target_price
                elif not trailing_active and price <= entry_price - entry_atr * trail_activate:
                    trailing_active = True
                    trailing_stop = price + entry_atr * trail_distance

            if not should_exit:
                if position_side == 1 and signals[i] == -1:
                    should_exit = True
                elif position_side == -1 and signals[i] == 1:
                    should_exit = True

            if should_exit:
                position_size = capital * 0.02

                if position_side == 1:
                    pnl = (exit_price - entry_price) / entry_price * position_size
                else:
                    pnl = (entry_price - exit_price) / entry_price * position_size

                pnl -= position_size * commission * 2
                capital += pnl

                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)

                position_side = 0
                trailing_active = False

        if position_side == 0 and signals[i] != 0:
            entry_atr = atr[i] if atr[i] > 0 else price * 0.02

            if signals[i] == 1:
                position_side = 1
                entry_price = price
                entry_idx = i
                stop_price = price - entry_atr * stop_mult
                target_price = price + entry_atr * target_mult
                trailing_active = False
            elif signals[i] == -1:
                position_side = -1
                entry_price = price
                entry_idx = i
                stop_price = price + entry_atr * stop_mult
                target_price = price - entry_atr * target_mult
                trailing_active = False

    total_trades = wins + losses
    if total_trades == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    win_rate = wins / total_trades
    pf = total_profit / total_loss if total_loss > 0 else 0.0

    return total_trades, win_rate, pf, total_profit, total_loss


# ============================================================
# DATA LOADING
# ============================================================

def load_data(symbols, interval='1h', train_end_year=2023):
    """Load data with train/test split"""
    conn = sqlite3.connect(DB_PATH)
    data = {'train': {}, 'test': {}}

    for symbol in symbols:
        try:
            df = pd.read_sql_query("""
                SELECT open_time, open, high, low, close, volume FROM ohlcv
                WHERE symbol = ? AND interval = ? ORDER BY open_time
            """, conn, params=(symbol, interval))

            if df.empty or len(df) < 500:
                continue

            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

            train_mask = df['timestamp'].dt.year <= train_end_year
            test_mask = df['timestamp'].dt.year > train_end_year

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) > 200:
                data['train'][symbol] = {
                    'close': train_df['close'].values.astype(np.float64),
                    'high': train_df['high'].values.astype(np.float64),
                    'low': train_df['low'].values.astype(np.float64),
                }

            if len(test_df) > 100:
                data['test'][symbol] = {
                    'close': test_df['close'].values.astype(np.float64),
                    'high': test_df['high'].values.astype(np.float64),
                    'low': test_df['low'].values.astype(np.float64),
                }
        except Exception as e:
            pass

    conn.close()
    return data


def test_variant(data_dict, rsi_oversold, rsi_overbought, adx_threshold, trend_period=100):
    """Test a variant on data"""
    total_trades = 0
    total_profit = 0.0
    total_loss = 0.0
    total_wins = 0

    for symbol, d in data_dict.items():
        signals, atr = generate_signals(
            d['close'], d['high'], d['low'],
            rsi_oversold, rsi_overbought, trend_period, adx_threshold
        )

        trades, wr, pf, profit, loss = backtest(
            d['close'], d['high'], d['low'], signals, atr,
            stop_mult=2.0, target_mult=4.0,
            trail_activate=1.5, trail_distance=1.0
        )

        if trades > 0:
            total_trades += trades
            total_profit += profit
            total_loss += loss
            total_wins += int(round(wr * trades))

    if total_trades == 0:
        return {'trades': 0, 'pf': 0, 'win_rate': 0}

    return {
        'trades': total_trades,
        'pf': total_profit / total_loss if total_loss > 0 else 0,
        'win_rate': total_wins / total_trades
    }


def run_tests():
    print("=" * 70)
    print("VARIANT TESTING: Increase trades while maintaining quality")
    print("=" * 70)

    # Baseline params
    BASE_RSI_OS = 35
    BASE_RSI_OB = 70
    BASE_ADX = 20

    # ============================================================
    # BASELINE
    # ============================================================
    print("\n[BASELINE] Loading 5 symbols, 1h timeframe...")
    data_base = load_data(SYMBOLS_BASE, '1h')
    print(f"  Train: {len(data_base['train'])} symbols")
    print(f"  Test: {len(data_base['test'])} symbols")

    # Warm up JIT
    print("  Warming up JIT...")
    if data_base['train']:
        symbol = list(data_base['train'].keys())[0]
        d = data_base['train'][symbol]
        signals, atr = generate_signals(d['close'], d['high'], d['low'], 35, 70, 100, 20)
        backtest(d['close'], d['high'], d['low'], signals, atr)

    baseline_train = test_variant(data_base['train'], BASE_RSI_OS, BASE_RSI_OB, BASE_ADX)
    baseline_test = test_variant(data_base['test'], BASE_RSI_OS, BASE_RSI_OB, BASE_ADX)
    print(f"  Baseline Train: {baseline_train['trades']} trades, PF={baseline_train['pf']:.2f}, WR={baseline_train['win_rate']:.1%}")
    print(f"  Baseline Test:  {baseline_test['trades']} trades, PF={baseline_test['pf']:.2f}, WR={baseline_test['win_rate']:.1%}")

    results = {
        'Baseline': {
            'train': baseline_train,
            'test': baseline_test
        }
    }

    # ============================================================
    # VARIANT A: Softer RSI (40/60)
    # ============================================================
    print("\n[VARIANT A] Softer RSI: 40/60 (was 35/70)...")
    a_train = test_variant(data_base['train'], 40, 60, BASE_ADX)
    a_test = test_variant(data_base['test'], 40, 60, BASE_ADX)
    print(f"  Train: {a_train['trades']} trades, PF={a_train['pf']:.2f}, WR={a_train['win_rate']:.1%}")
    print(f"  Test:  {a_test['trades']} trades, PF={a_test['pf']:.2f}, WR={a_test['win_rate']:.1%}")
    results['A: RSI 40/60'] = {'train': a_train, 'test': a_test}

    # ============================================================
    # VARIANT B: No ADX filter
    # ============================================================
    print("\n[VARIANT B] No ADX filter (ADX=0)...")
    b_train = test_variant(data_base['train'], BASE_RSI_OS, BASE_RSI_OB, 0)
    b_test = test_variant(data_base['test'], BASE_RSI_OS, BASE_RSI_OB, 0)
    print(f"  Train: {b_train['trades']} trades, PF={b_train['pf']:.2f}, WR={b_train['win_rate']:.1%}")
    print(f"  Test:  {b_test['trades']} trades, PF={b_test['pf']:.2f}, WR={b_test['win_rate']:.1%}")
    results['B: No ADX'] = {'train': b_train, 'test': b_test}

    # ============================================================
    # VARIANT C: 15m timeframe
    # ============================================================
    print("\n[VARIANT C] 15m timeframe (was 1h)...")
    data_15m = load_data(SYMBOLS_BASE, '15m')
    if data_15m['train']:
        c_train = test_variant(data_15m['train'], BASE_RSI_OS, BASE_RSI_OB, BASE_ADX)
        c_test = test_variant(data_15m['test'], BASE_RSI_OS, BASE_RSI_OB, BASE_ADX)
        print(f"  Train: {c_train['trades']} trades, PF={c_train['pf']:.2f}, WR={c_train['win_rate']:.1%}")
        print(f"  Test:  {c_test['trades']} trades, PF={c_test['pf']:.2f}, WR={c_test['win_rate']:.1%}")
        results['C: 15m TF'] = {'train': c_train, 'test': c_test}
    else:
        print("  No 15m data available!")
        results['C: 15m TF'] = {'train': {'trades': 0, 'pf': 0, 'win_rate': 0},
                                'test': {'trades': 0, 'pf': 0, 'win_rate': 0}}

    # ============================================================
    # VARIANT D: 10 symbols
    # ============================================================
    print("\n[VARIANT D] 10 symbols (was 5)...")
    data_10 = load_data(SYMBOLS_EXTENDED, '1h')
    print(f"  Loaded: {len(data_10['train'])} train, {len(data_10['test'])} test symbols")
    d_train = test_variant(data_10['train'], BASE_RSI_OS, BASE_RSI_OB, BASE_ADX)
    d_test = test_variant(data_10['test'], BASE_RSI_OS, BASE_RSI_OB, BASE_ADX)
    print(f"  Train: {d_train['trades']} trades, PF={d_train['pf']:.2f}, WR={d_train['win_rate']:.1%}")
    print(f"  Test:  {d_test['trades']} trades, PF={d_test['pf']:.2f}, WR={d_test['win_rate']:.1%}")
    results['D: 10 symbols'] = {'train': d_train, 'test': d_test}

    # ============================================================
    # VARIANT A+D: Soft RSI + 10 symbols
    # ============================================================
    print("\n[VARIANT A+D] RSI 40/60 + 10 symbols...")
    ad_train = test_variant(data_10['train'], 40, 60, BASE_ADX)
    ad_test = test_variant(data_10['test'], 40, 60, BASE_ADX)
    print(f"  Train: {ad_train['trades']} trades, PF={ad_train['pf']:.2f}, WR={ad_train['win_rate']:.1%}")
    print(f"  Test:  {ad_test['trades']} trades, PF={ad_test['pf']:.2f}, WR={ad_test['win_rate']:.1%}")
    results['A+D: RSI+Symbols'] = {'train': ad_train, 'test': ad_test}

    # ============================================================
    # VARIANT B+D: No ADX + 10 symbols
    # ============================================================
    print("\n[VARIANT B+D] No ADX + 10 symbols...")
    bd_train = test_variant(data_10['train'], BASE_RSI_OS, BASE_RSI_OB, 0)
    bd_test = test_variant(data_10['test'], BASE_RSI_OS, BASE_RSI_OB, 0)
    print(f"  Train: {bd_train['trades']} trades, PF={bd_train['pf']:.2f}, WR={bd_train['win_rate']:.1%}")
    print(f"  Test:  {bd_test['trades']} trades, PF={bd_test['pf']:.2f}, WR={bd_test['win_rate']:.1%}")
    results['B+D: NoADX+Symbols'] = {'train': bd_train, 'test': bd_test}

    # ============================================================
    # RESULTS TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON (TEST DATA 2024-2025)")
    print("=" * 70)
    print("\n| Вариант            | Сделок | PF   | Win Rate | Цель 200+ | PF>1.3 |")
    print("|" + "-" * 20 + "|" + "-" * 8 + "|" + "-" * 6 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 8 + "|")

    for name, res in results.items():
        test = res['test']
        trades_ok = "✅" if test['trades'] >= 200 else "❌"
        pf_ok = "✅" if test['pf'] >= 1.3 else "❌"
        print(f"| {name:<18} | {test['trades']:>6} | {test['pf']:.2f} | {test['win_rate']:>8.1%} | {trades_ok:>9} | {pf_ok:>6} |")

    # Find best variant
    print("\n" + "=" * 70)
    print("BEST VARIANTS (sorted by trades meeting criteria)")
    print("=" * 70)

    valid = [(name, res) for name, res in results.items() if res['test']['pf'] >= 1.3]
    valid.sort(key=lambda x: x[1]['test']['trades'], reverse=True)

    if valid:
        for i, (name, res) in enumerate(valid[:3], 1):
            test = res['test']
            train = res['train']
            print(f"\n#{i}: {name}")
            print(f"    TEST:  {test['trades']} trades | PF={test['pf']:.2f} | WR={test['win_rate']:.1%}")
            print(f"    TRAIN: {train['trades']} trades | PF={train['pf']:.2f} | WR={train['win_rate']:.1%}")
    else:
        print("\nNo variants meet PF >= 1.3 criteria!")

    return results


if __name__ == '__main__':
    run_tests()
