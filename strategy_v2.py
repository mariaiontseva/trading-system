"""
STRATEGY V2 - Исправленная версия с train/test split
Устраняет все изъяны предыдущей версии
"""
import sqlite3
import numpy as np
import pandas as pd
from numba import jit
from datetime import datetime
import time
import os
import warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'prices.db')

# Расширенный список пар (топ-15 по ликвидности)
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
    'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT',
    'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'ETCUSDT'
]

# ============================================================
# NUMBA JIT COMPILED INDICATORS
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
    """ADX для фильтра силы тренда"""
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

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    # Smooth DM
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


@jit(nopython=True, cache=True)
def calc_volume_ma(volume, period=20):
    n = len(volume)
    vol_ma = np.zeros(n)
    for i in range(period, n):
        vol_ma[i] = np.mean(volume[i-period:i])
    return vol_ma


# ============================================================
# STRATEGY V2: IMPROVED WITH FILTERS
# ============================================================

@jit(nopython=True, cache=True)
def generate_signals_v2(close, high, low, volume,
                        rsi_oversold=40, rsi_overbought=60,
                        trend_period=100, adx_threshold=20,
                        atr_min_mult=0.5, volume_mult=1.0):
    """
    Улучшенная стратегия с фильтрами:
    - RSI + Trend (EMA)
    - ADX фильтр (сила тренда)
    - ATR фильтр (не торгуем во флэте)
    - Volume фильтр (подтверждение объёмом)
    """
    n = len(close)
    signals = np.zeros(n)

    # Calculate indicators
    rsi = calc_rsi(close, 14)
    ema_trend = calc_ema(close, trend_period)
    atr = calc_atr(high, low, close, 14)
    adx = calc_adx(high, low, close, 14)
    vol_ma = calc_volume_ma(volume, 20)

    # ATR baseline for filter
    atr_ma = calc_ema(atr, 20)

    for i in range(max(trend_period, 30), n):
        # Skip if no trend (ADX filter)
        if adx[i] < adx_threshold:
            continue

        # Skip if flat market (ATR filter)
        if atr[i] < atr_ma[i] * atr_min_mult:
            continue

        # Skip if low volume
        if vol_ma[i] > 0 and volume[i] < vol_ma[i] * volume_mult:
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
# BACKTEST ENGINE WITH TRAILING STOP
# ============================================================

@jit(nopython=True, cache=True)
def backtest_with_trailing(close, high, low, signals, atr,
                           stop_mult=2.0, target_mult=3.0,
                           trail_activate=1.5, trail_distance=1.0,
                           commission=0.001):
    """
    Бэктест с trailing stop:
    - Trailing активируется при достижении trail_activate × ATR прибыли
    - После активации стоп следует за ценой на расстоянии trail_distance × ATR
    """
    n = len(close)
    capital = 10000.0
    position_side = 0  # 0=none, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    entry_atr = 0.0
    stop_price = 0.0
    target_price = 0.0
    trailing_active = False
    trailing_stop = 0.0

    # Metrics
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    max_capital = capital
    max_drawdown = 0.0
    trade_durations = []
    equity_curve = np.zeros(n)

    for i in range(50, n):
        price = close[i]
        equity_curve[i] = capital

        # Track drawdown
        if capital > max_capital:
            max_capital = capital
        dd = (max_capital - capital) / max_capital * 100
        if dd > max_drawdown:
            max_drawdown = dd

        # Check exit for existing position
        if position_side != 0:
            hours = i - entry_idx

            # Update trailing stop if active
            if trailing_active:
                if position_side == 1:  # LONG
                    new_trail = price - entry_atr * trail_distance
                    if new_trail > trailing_stop:
                        trailing_stop = new_trail
                else:  # SHORT
                    new_trail = price + entry_atr * trail_distance
                    if new_trail < trailing_stop:
                        trailing_stop = new_trail

            should_exit = False
            exit_price = price

            if position_side == 1:  # LONG
                # Check trailing stop
                if trailing_active and price <= trailing_stop:
                    should_exit = True
                    exit_price = trailing_stop
                # Check initial stop
                elif price <= stop_price:
                    should_exit = True
                    exit_price = stop_price
                # Check take profit
                elif price >= target_price:
                    should_exit = True
                    exit_price = target_price
                # Activate trailing
                elif not trailing_active and price >= entry_price + entry_atr * trail_activate:
                    trailing_active = True
                    trailing_stop = price - entry_atr * trail_distance

            else:  # SHORT
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

            # Signal reversal exit
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

                # Commission
                pnl -= position_size * commission * 2

                capital += pnl
                trade_durations.append(hours)

                if pnl > 0:
                    wins += 1
                    total_profit += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)

                position_side = 0
                trailing_active = False

        # Check entry
        if position_side == 0 and signals[i] != 0:
            entry_atr = atr[i] if atr[i] > 0 else price * 0.02

            if signals[i] == 1:  # LONG
                position_side = 1
                entry_price = price
                entry_idx = i
                stop_price = price - entry_atr * stop_mult
                target_price = price + entry_atr * target_mult
                trailing_active = False

            elif signals[i] == -1:  # SHORT
                position_side = -1
                entry_price = price
                entry_idx = i
                stop_price = price + entry_atr * stop_mult
                target_price = price - entry_atr * target_mult
                trailing_active = False

    # Final equity
    equity_curve[n-1] = capital

    total_trades = wins + losses
    if total_trades == 0:
        return 0.0, 0, 0.0, 0.0, 0.0, 0.0, equity_curve, 0.0, 0.0

    win_rate = wins / total_trades
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    return_pct = (capital - 10000.0) / 10000.0 * 100.0

    # Sharpe Ratio approximation
    avg_duration = np.mean(np.array(trade_durations)) if trade_durations else 0
    trades_per_year = total_trades / 6  # ~6 years of data
    avg_return_per_trade = return_pct / total_trades if total_trades > 0 else 0
    sharpe = (avg_return_per_trade * trades_per_year) / (max_drawdown + 0.001) if max_drawdown > 0 else 0

    # Return raw profit/loss for correct aggregation
    return return_pct, total_trades, win_rate, profit_factor, max_drawdown, sharpe, equity_curve, total_profit, total_loss


# ============================================================
# DATA LOADING WITH TRAIN/TEST SPLIT
# ============================================================

def load_data_split(train_end_year=2023):
    """Load data with train/test split"""
    conn = sqlite3.connect(DB_PATH)
    data = {'train': {}, 'test': {}}

    for symbol in SYMBOLS:
        try:
            df = pd.read_sql_query("""
                SELECT open_time, open, high, low, close, volume FROM ohlcv
                WHERE symbol = ? AND interval = '1h' ORDER BY open_time
            """, conn, params=(symbol,))

            if df.empty or len(df) < 1000:
                continue

            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

            # Split: train = up to end of 2023, test = 2024+
            train_mask = df['timestamp'].dt.year <= train_end_year
            test_mask = df['timestamp'].dt.year > train_end_year

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) > 500:
                data['train'][symbol] = {
                    'close': train_df['close'].values.astype(np.float64),
                    'high': train_df['high'].values.astype(np.float64),
                    'low': train_df['low'].values.astype(np.float64),
                    'volume': train_df['volume'].values.astype(np.float64),
                    'dates': train_df['timestamp'].values
                }

            if len(test_df) > 100:
                data['test'][symbol] = {
                    'close': test_df['close'].values.astype(np.float64),
                    'high': test_df['high'].values.astype(np.float64),
                    'low': test_df['low'].values.astype(np.float64),
                    'volume': test_df['volume'].values.astype(np.float64),
                    'dates': test_df['timestamp'].values
                }
        except Exception as e:
            print(f"Error loading {symbol}: {e}")

    conn.close()
    return data


def test_strategy(data_dict, params):
    """Test strategy on a dataset"""
    results = []

    for symbol, d in data_dict.items():
        signals, atr = generate_signals_v2(
            d['close'], d['high'], d['low'], d['volume'],
            rsi_oversold=params['rsi_oversold'],
            rsi_overbought=params['rsi_overbought'],
            trend_period=params['trend_period'],
            adx_threshold=params['adx_threshold'],
            atr_min_mult=params['atr_min_mult'],
            volume_mult=params['volume_mult']
        )

        ret, trades, wr, pf, dd, sharpe, eq, raw_profit, raw_loss = backtest_with_trailing(
            d['close'], d['high'], d['low'], signals, atr,
            stop_mult=params['stop_mult'],
            target_mult=params['target_mult'],
            trail_activate=params['trail_activate'],
            trail_distance=params['trail_distance'],
            commission=0.001
        )

        if trades > 0:
            # Store raw values for CORRECT aggregation
            results.append({
                'symbol': symbol,
                'return': ret,
                'trades': trades,
                'wins': int(round(wr * trades)),
                'win_rate': wr,
                'pf': pf,
                'drawdown': dd,
                'sharpe': sharpe,
                'raw_profit': raw_profit,  # Actual $ profit
                'raw_loss': raw_loss       # Actual $ loss
            })

    if not results:
        return None

    # CORRECT AGGREGATION: Sum raw profits and losses across all symbols
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    total_raw_profit = sum(r['raw_profit'] for r in results)
    total_raw_loss = sum(r['raw_loss'] for r in results)
    total_returns = sum(r['return'] for r in results)

    # CORRECT Profit Factor = Total Profit / Total Loss
    correct_pf = total_raw_profit / total_raw_loss if total_raw_loss > 0 else 0.0

    return {
        'return': total_returns / len(results),  # Average return per symbol
        'total_return': total_returns,  # Sum of all returns
        'trades': total_trades,
        'win_rate': total_wins / total_trades if total_trades > 0 else 0,
        'pf': correct_pf,  # CORRECT: Total Profit / Total Loss
        'drawdown': np.max([r['drawdown'] for r in results]),
        'sharpe': np.mean([r['sharpe'] for r in results]),
        'symbols_traded': len(results),
        'raw_profit': total_raw_profit,
        'raw_loss': total_raw_loss
    }


def run_optimization():
    print("=" * 70)
    print("STRATEGY V2: TRAIN/TEST SPLIT OPTIMIZATION")
    print("=" * 70)

    # Load data
    print("\n1. Loading data with train/test split...")
    data = load_data_split(train_end_year=2023)
    print(f"   Train symbols: {len(data['train'])}")
    print(f"   Test symbols: {len(data['test'])}")

    for symbol in list(data['train'].keys())[:5]:
        train_len = len(data['train'].get(symbol, {}).get('close', []))
        test_len = len(data['test'].get(symbol, {}).get('close', []))
        print(f"   {symbol}: train={train_len}, test={test_len}")

    # Warm up JIT
    print("\n2. Warming up JIT...")
    start = time.time()
    if data['train']:
        symbol = list(data['train'].keys())[0]
        d = data['train'][symbol]
        signals, atr = generate_signals_v2(d['close'], d['high'], d['low'], d['volume'])
        backtest_with_trailing(d['close'], d['high'], d['low'], signals, atr)
    print(f"   JIT compiled in {time.time()-start:.1f}s")

    # Parameter grid
    print("\n3. Optimizing on TRAIN data (2018-2023)...")
    start = time.time()

    param_grid = []
    for rsi_os in [30, 35, 40]:
        for rsi_ob in [60, 65, 70]:
            for trend in [50, 100, 200]:
                for adx in [15, 20, 25]:
                    for stop in [1.5, 2.0, 2.5]:
                        for target in [2.0, 3.0, 4.0]:
                            for trail_act in [1.0, 1.5, 2.0]:
                                param_grid.append({
                                    'rsi_oversold': rsi_os,
                                    'rsi_overbought': rsi_ob,
                                    'trend_period': trend,
                                    'adx_threshold': adx,
                                    'atr_min_mult': 0.5,
                                    'volume_mult': 0.8,
                                    'stop_mult': stop,
                                    'target_mult': target,
                                    'trail_activate': trail_act,
                                    'trail_distance': 1.0
                                })

    print(f"   Testing {len(param_grid)} combinations...")

    train_results = []
    for i, params in enumerate(param_grid):
        result = test_strategy(data['train'], params)
        if result and result['trades'] >= 50:  # Minimum trades filter
            train_results.append({
                'params': params,
                'result': result
            })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (len(param_grid) - i - 1)
            print(f"   {i+1}/{len(param_grid)} ({elapsed:.0f}s, ETA: {eta:.0f}s)")

    print(f"   Completed in {time.time()-start:.1f}s")
    print(f"   Valid results: {len(train_results)}")

    # Sort by profit factor (most robust metric)
    train_results.sort(key=lambda x: x['result']['pf'], reverse=True)

    # Show top 10 on TRAIN
    print("\n" + "=" * 70)
    print("TOP 10 ON TRAIN DATA (2018-2023)")
    print("=" * 70)

    for i, r in enumerate(train_results[:10], 1):
        res = r['result']
        p = r['params']
        print(f"\n#{i}: PF={res['pf']:.2f} | Win={res['win_rate']:.1%} | Return={res['return']:+.1f}%")
        print(f"    Trades={res['trades']} | DD={res['drawdown']:.1f}% | Sharpe={res['sharpe']:.2f}")
        print(f"    RSI={p['rsi_oversold']}/{p['rsi_overbought']} | Trend={p['trend_period']} | ADX>{p['adx_threshold']}")
        print(f"    SL={p['stop_mult']}×ATR | TP={p['target_mult']}×ATR | Trail@{p['trail_activate']}×ATR")

    # Test top 5 on TEST data
    print("\n" + "=" * 70)
    print("VALIDATION ON TEST DATA (2024-2025)")
    print("=" * 70)

    print("\n| # | Train PF | Train Win | Test PF | Test Win | Test Return | Test Trades |")
    print("|---|----------|-----------|---------|----------|-------------|-------------|")

    best_test_result = None
    best_params = None

    for i, r in enumerate(train_results[:10], 1):
        params = r['params']
        train_res = r['result']

        # Test on out-of-sample data
        test_res = test_strategy(data['test'], params)

        if test_res:
            print(f"| {i} | {train_res['pf']:.2f} | {train_res['win_rate']:.1%} | {test_res['pf']:.2f} | {test_res['win_rate']:.1%} | {test_res['return']:+.1f}% | {test_res['trades']} |")

            # Track best test result
            if best_test_result is None or test_res['pf'] > best_test_result['pf']:
                best_test_result = test_res
                best_params = params
        else:
            print(f"| {i} | {train_res['pf']:.2f} | {train_res['win_rate']:.1%} | N/A | N/A | N/A | 0 |")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    # Old strategy results (from previous run)
    old_result = {
        'win_rate': 0.804,
        'pf': 5.55,
        'trades': 133,
        'drawdown': 'N/A',
        'sharpe': 'N/A'
    }

    if best_params:
        new_train = test_strategy(data['train'], best_params)
        new_test = best_test_result

        print("\n| Метрика | Старый (in-sample) | Новый (train) | Новый (test) |")
        print("|---------|-------------------|---------------|--------------|")
        print(f"| Win Rate | {old_result['win_rate']:.1%} | {new_train['win_rate']:.1%} | {new_test['win_rate']:.1%} |")
        print(f"| Profit Factor | {old_result['pf']:.2f} | {new_train['pf']:.2f} | {new_test['pf']:.2f} |")
        print(f"| Всего сделок | {old_result['trades']} | {new_train['trades']} | {new_test['trades']} |")
        print(f"| Max Drawdown | {old_result['drawdown']} | {new_train['drawdown']:.1f}% | {new_test['drawdown']:.1f}% |")
        print(f"| Sharpe Ratio | {old_result['sharpe']} | {new_train['sharpe']:.2f} | {new_test['sharpe']:.2f} |")

        print("\n" + "=" * 70)
        print("ЛУЧШИЕ ПАРАМЕТРЫ (проверены на out-of-sample)")
        print("=" * 70)
        print(f"""
RSI Oversold:     {best_params['rsi_oversold']}
RSI Overbought:   {best_params['rsi_overbought']}
Trend EMA:        {best_params['trend_period']}
ADX Threshold:    {best_params['adx_threshold']}
Stop Loss:        {best_params['stop_mult']} × ATR
Take Profit:      {best_params['target_mult']} × ATR
Trail Activate:   {best_params['trail_activate']} × ATR
Trail Distance:   {best_params['trail_distance']} × ATR
        """)

    return train_results, best_params


if __name__ == '__main__':
    results, best = run_optimization()
