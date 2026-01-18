"""
DEBUG BACKTEST - Проверка на ошибки
"""
import sqlite3
import numpy as np
import pandas as pd
from numba import jit
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'prices.db')


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


def debug_backtest(symbol='BTCUSDT'):
    """Детальный бэктест с выводом каждой сделки"""

    print("=" * 80)
    print(f"DEBUG BACKTEST: {symbol}")
    print("=" * 80)

    # Load data
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT open_time, open, high, low, close, volume FROM ohlcv
        WHERE symbol = ? AND interval = '1h' ORDER BY open_time
    """, conn, params=(symbol,))
    conn.close()

    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

    # Filter train period
    df = df[df['timestamp'].dt.year <= 2023]

    print(f"\nДанные: {len(df)} свечей")
    print(f"Период: {df['timestamp'].iloc[0]} — {df['timestamp'].iloc[-1]}")

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    open_price = df['open'].values.astype(np.float64)
    timestamps = df['timestamp'].values

    # Calculate indicators
    rsi = calc_rsi(close, 14)
    ema_100 = calc_ema(close, 100)
    atr = calc_atr(high, low, close, 14)
    adx = calc_adx(high, low, close, 14)

    # Strategy parameters
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 70
    ADX_MIN = 20
    STOP_MULT = 2.0
    TARGET_MULT = 4.0
    TRAIL_ACTIVATE = 1.0
    TRAIL_DISTANCE = 1.0
    COMMISSION = 0.001  # 0.1%

    print(f"\nПараметры:")
    print(f"  RSI: <{RSI_OVERSOLD} для LONG, >{RSI_OVERBOUGHT} для SHORT")
    print(f"  Trend: EMA100")
    print(f"  ADX: >{ADX_MIN}")
    print(f"  Stop: {STOP_MULT}×ATR, Target: {TARGET_MULT}×ATR")
    print(f"  Trailing: активация @{TRAIL_ACTIVATE}×ATR, дистанция {TRAIL_DISTANCE}×ATR")
    print(f"  Комиссия: {COMMISSION*100}% × 2 = {COMMISSION*2*100}%")

    # Backtest
    capital = 10000.0
    initial_capital = capital
    position = None
    trades = []

    print("\n" + "=" * 80)
    print("ПРОВЕРКА LOOK-AHEAD BIAS:")
    print("=" * 80)
    print("Сигнал генерируется на основе закрытия свечи [i]")
    print("Вход происходит по цене ОТКРЫТИЯ следующей свечи [i+1]")
    print("Это правильно — нет look-ahead bias")

    n = len(close)
    for i in range(101, n-1):  # -1 чтобы был следующий бар для входа
        # ========================================
        # ПРОВЕРКА 1: Look-ahead bias
        # Все индикаторы рассчитаны на данных [0:i] включительно
        # Сигнал на свече [i], вход на открытии свечи [i+1]
        # ========================================

        current_close = close[i]
        current_rsi = rsi[i]
        current_ema = ema_100[i]
        current_atr = atr[i]
        current_adx = adx[i]

        # Цена входа = открытие СЛЕДУЮЩЕЙ свечи (избегаем look-ahead)
        entry_price_for_signal = open_price[i+1]

        # Check exit for existing position
        if position is not None:
            pos_side = position['side']
            pos_entry = position['entry_price']
            pos_stop = position['stop']
            pos_target = position['target']
            pos_atr = position['atr']
            pos_trailing_active = position['trailing_active']
            pos_trailing_stop = position['trailing_stop']

            # Update trailing
            if pos_trailing_active:
                if pos_side == 'LONG':
                    new_trail = current_close - pos_atr * TRAIL_DISTANCE
                    if new_trail > pos_trailing_stop:
                        position['trailing_stop'] = new_trail
                        pos_trailing_stop = new_trail
                else:
                    new_trail = current_close + pos_atr * TRAIL_DISTANCE
                    if new_trail < pos_trailing_stop:
                        position['trailing_stop'] = new_trail
                        pos_trailing_stop = new_trail

            should_exit = False
            exit_price = current_close
            exit_reason = None

            if pos_side == 'LONG':
                # Trailing stop hit
                if pos_trailing_active and low[i] <= pos_trailing_stop:
                    should_exit = True
                    exit_price = pos_trailing_stop
                    exit_reason = 'TRAILING'
                # Stop loss hit
                elif low[i] <= pos_stop:
                    should_exit = True
                    exit_price = pos_stop
                    exit_reason = 'STOP_LOSS'
                # Take profit hit
                elif high[i] >= pos_target:
                    should_exit = True
                    exit_price = pos_target
                    exit_reason = 'TAKE_PROFIT'
                # Activate trailing
                elif not pos_trailing_active and high[i] >= pos_entry + pos_atr * TRAIL_ACTIVATE:
                    position['trailing_active'] = True
                    position['trailing_stop'] = current_close - pos_atr * TRAIL_DISTANCE

            else:  # SHORT
                if pos_trailing_active and high[i] >= pos_trailing_stop:
                    should_exit = True
                    exit_price = pos_trailing_stop
                    exit_reason = 'TRAILING'
                elif high[i] >= pos_stop:
                    should_exit = True
                    exit_price = pos_stop
                    exit_reason = 'STOP_LOSS'
                elif low[i] <= pos_target:
                    should_exit = True
                    exit_price = pos_target
                    exit_reason = 'TAKE_PROFIT'
                elif not pos_trailing_active and low[i] <= pos_entry - pos_atr * TRAIL_ACTIVATE:
                    position['trailing_active'] = True
                    position['trailing_stop'] = current_close + pos_atr * TRAIL_DISTANCE

            if should_exit:
                # Calculate P&L
                position_size = position['size']  # Fixed at entry!

                if pos_side == 'LONG':
                    gross_pnl = (exit_price - pos_entry) / pos_entry * position_size
                else:
                    gross_pnl = (pos_entry - exit_price) / pos_entry * position_size

                # Commission: 0.1% on entry + 0.1% on exit
                commission_cost = position_size * COMMISSION * 2
                net_pnl = gross_pnl - commission_cost

                capital += net_pnl

                pnl_pct = (exit_price - pos_entry) / pos_entry * 100 if pos_side == 'LONG' else (pos_entry - exit_price) / pos_entry * 100

                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': pos_entry,
                    'exit_date': pd.Timestamp(timestamps[i]),
                    'exit_price': exit_price,
                    'side': pos_side,
                    'reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': net_pnl,
                    'gross_pnl': gross_pnl,
                    'commission': commission_cost,
                    'position_size': position_size
                })

                position = None

        # Check entry signal (only if no position)
        if position is None:
            signal = 0

            # Filters
            if current_adx < ADX_MIN:
                continue

            # LONG: RSI oversold + uptrend
            if current_rsi < RSI_OVERSOLD and current_close > current_ema:
                signal = 1
            # SHORT: RSI overbought + downtrend
            elif current_rsi > RSI_OVERBOUGHT and current_close < current_ema:
                signal = -1

            if signal != 0:
                # Entry price = OPEN of NEXT candle (no look-ahead!)
                actual_entry_price = open_price[i+1]
                entry_atr = current_atr

                # Position size = 2% of current capital (FIXED at entry)
                pos_size = capital * 0.02

                if signal == 1:  # LONG
                    stop = actual_entry_price - entry_atr * STOP_MULT
                    target = actual_entry_price + entry_atr * TARGET_MULT
                    position = {
                        'side': 'LONG',
                        'entry_price': actual_entry_price,
                        'entry_date': pd.Timestamp(timestamps[i+1]),
                        'stop': stop,
                        'target': target,
                        'atr': entry_atr,
                        'size': pos_size,
                        'trailing_active': False,
                        'trailing_stop': 0
                    }
                else:  # SHORT
                    stop = actual_entry_price + entry_atr * STOP_MULT
                    target = actual_entry_price - entry_atr * TARGET_MULT
                    position = {
                        'side': 'SHORT',
                        'entry_price': actual_entry_price,
                        'entry_date': pd.Timestamp(timestamps[i+1]),
                        'stop': stop,
                        'target': target,
                        'atr': entry_atr,
                        'size': pos_size,
                        'trailing_active': False,
                        'trailing_stop': 0
                    }

    # Results
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 80)

    if not trades:
        print("Нет сделок!")
        return

    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]

    total_gross_profit = sum(t['gross_pnl'] for t in wins)
    total_gross_loss = abs(sum(t['gross_pnl'] for t in losses))
    total_commission = sum(t['commission'] for t in trades)
    total_net_profit = sum(t['pnl_usd'] for t in wins)
    total_net_loss = abs(sum(t['pnl_usd'] for t in losses))

    print(f"\nВсего сделок: {len(trades)}")
    print(f"  Выигрышных: {len(wins)}")
    print(f"  Убыточных: {len(losses)}")
    print(f"  Win Rate: {len(wins)/len(trades)*100:.1f}%")

    print(f"\n--- РАСЧЁТ PROFIT FACTOR ---")
    print(f"Gross прибыль (до комиссий): ${total_gross_profit:.2f}")
    print(f"Gross убыток (до комиссий): ${total_gross_loss:.2f}")
    print(f"Всего комиссий: ${total_commission:.2f}")
    print(f"Net прибыль: ${total_net_profit:.2f}")
    print(f"Net убыток: ${total_net_loss:.2f}")

    if total_net_loss > 0:
        pf = total_net_profit / total_net_loss
        print(f"\nProfit Factor = {total_net_profit:.2f} / {total_net_loss:.2f} = {pf:.2f}")
    else:
        print(f"\nProfit Factor = ∞ (нет убыточных сделок с учётом net!)")
        if total_gross_loss > 0:
            pf_gross = total_gross_profit / total_gross_loss
            print(f"Gross PF = {total_gross_profit:.2f} / {total_gross_loss:.2f} = {pf_gross:.2f}")

    print(f"\nНачальный капитал: ${initial_capital:.2f}")
    print(f"Конечный капитал: ${capital:.2f}")
    print(f"Доходность: {(capital-initial_capital)/initial_capital*100:.2f}%")

    # Show 5 example trades
    print("\n" + "=" * 80)
    print("5 ПРИМЕРОВ СДЕЛОК")
    print("=" * 80)

    for i, t in enumerate(trades[:5], 1):
        print(f"\n--- Сделка #{i} ---")
        print(f"  Сторона: {t['side']}")
        print(f"  Вход: {t['entry_date']} @ ${t['entry_price']:.2f}")
        print(f"  Выход: {t['exit_date']} @ ${t['exit_price']:.2f}")
        print(f"  Причина: {t['reason']}")
        print(f"  P&L: {t['pnl_pct']:+.2f}% (${t['pnl_usd']:+.2f})")
        print(f"  Размер позиции: ${t['position_size']:.2f}")
        print(f"  Комиссия: ${t['commission']:.2f}")

    # Check for issues
    print("\n" + "=" * 80)
    print("ДИАГНОСТИКА ПРОБЛЕМ")
    print("=" * 80)

    # Check if all trades are winners
    if len(losses) == 0:
        print("⚠️  ПРОБЛЕМА: Все сделки выигрышные! Возможно:")
        print("   - Stop Loss никогда не срабатывает")
        print("   - Take Profit слишком близко")
        print("   - Trailing Stop слишком агрессивный")

    # Check PF
    if total_net_loss > 0:
        pf = total_net_profit / total_net_loss
        if pf > 10:
            print(f"⚠️  ПРОБЛЕМА: PF={pf:.2f} слишком высокий!")
            print("   Проверь что stop loss реально срабатывает")

    # Check trade frequency
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    trades_per_year = len(trades) / (days / 365)
    print(f"\nСделок в год: {trades_per_year:.1f}")
    if trades_per_year < 10:
        print("⚠️  ПРОБЛЕМА: Мало сделок для статистической значимости")

    # Check exit reasons
    exit_reasons = {}
    for t in trades:
        r = t['reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    print(f"\nПричины выхода:")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")

    if 'STOP_LOSS' not in exit_reasons or exit_reasons.get('STOP_LOSS', 0) == 0:
        print("⚠️  ПРОБЛЕМА: Stop Loss ни разу не сработал!")

    return trades


if __name__ == '__main__':
    trades = debug_backtest('BTCUSDT')
