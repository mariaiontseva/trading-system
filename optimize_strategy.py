"""
Strategy Optimizer - поиск оптимальных параметров на исторических данных
"""
import sqlite3
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime, timedelta
import os

# Параметры для оптимизации
STOP_LOSSES = [0.01, 0.015, 0.02, 0.025, 0.03]  # 1%, 1.5%, 2%, 2.5%, 3%
TAKE_PROFITS = [0.02, 0.03, 0.04, 0.05, 0.06]   # 2%, 3%, 4%, 5%, 6%
TIME_EXITS = [1, 2, 4, 8, 12, 24, None]          # часы или None (без лимита)
SIGNAL_THRESHOLDS = [0.1, 0.15, 0.2, 0.25]       # порог входа

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'prices.db')
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


def load_candles(symbol: str, interval: str = '1h') -> pd.DataFrame:
    """Загрузка свечей из базы"""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND interval = ?
        ORDER BY open_time ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol, interval))
    conn.close()

    if df.empty:
        return df

    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def calculate_signal(closes: np.ndarray, volumes: np.ndarray) -> float:
    """Расчёт сигнала (тот же что в auto_trading)"""
    if len(closes) < 50:
        return 0.0

    score = 0.0

    # RSI
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
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
    ema_9 = np.mean(closes[-9:])
    ema_21 = np.mean(closes[-21:])
    ema_50 = np.mean(closes[-50:])

    if ema_9 > ema_21 > ema_50:
        score += 0.2
    elif ema_9 < ema_21 < ema_50:
        score -= 0.2

    # Volume
    avg_vol = np.mean(volumes[-20:])
    if volumes[-1] > avg_vol * 1.5:
        if score > 0:
            score += 0.1
        elif score < 0:
            score -= 0.1

    # Momentum
    momentum = (closes[-1] - closes[-10]) / closes[-10]
    if momentum > 0.02:
        score += 0.15
    elif momentum < -0.02:
        score -= 0.15

    return max(-1.0, min(1.0, score))


def backtest_params(df: pd.DataFrame, stop_loss: float, take_profit: float,
                    time_exit: int, signal_threshold: float) -> dict:
    """Бэктест одной комбинации параметров"""

    capital = 10000.0
    position = None
    trades = []

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    timestamps = df.index

    for i in range(50, len(df)):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        current_time = timestamps[i]

        # Расчёт сигнала
        score = calculate_signal(closes[:i+1], volumes[:i+1])

        # Если есть позиция - проверяем выход
        if position:
            entry_price = position['entry_price']
            side = position['side']
            entry_time = position['entry_time']

            # P&L
            if side == 'LONG':
                pnl_pct = (price - entry_price) / entry_price
                hit_sl = low <= entry_price * (1 - stop_loss)
                hit_tp = high >= entry_price * (1 + take_profit)
            else:  # SHORT
                pnl_pct = (entry_price - price) / entry_price
                hit_sl = high >= entry_price * (1 + stop_loss)
                hit_tp = low <= entry_price * (1 - take_profit)

            # Время в позиции
            hours_in_position = (current_time - entry_time).total_seconds() / 3600

            # Условия выхода
            exit_reason = None
            exit_price = price

            if hit_sl:
                exit_reason = 'SL'
                exit_price = entry_price * (1 - stop_loss) if side == 'LONG' else entry_price * (1 + stop_loss)
            elif hit_tp:
                exit_reason = 'TP'
                exit_price = entry_price * (1 + take_profit) if side == 'LONG' else entry_price * (1 - take_profit)
            elif time_exit and hours_in_position >= time_exit:
                exit_reason = 'TIME'
            elif side == 'LONG' and score < -signal_threshold:
                exit_reason = 'SIGNAL'
            elif side == 'SHORT' and score > signal_threshold:
                exit_reason = 'SIGNAL'

            if exit_reason:
                # Закрываем позицию
                if side == 'LONG':
                    pnl = (exit_price - entry_price) / entry_price * position['size']
                else:
                    pnl = (entry_price - exit_price) / entry_price * position['size']

                pnl -= position['size'] * 0.002  # комиссия 0.1% * 2
                capital += pnl

                trades.append({
                    'side': side,
                    'entry': entry_price,
                    'exit': exit_price,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'hours': hours_in_position
                })
                position = None

        # Если нет позиции - проверяем вход
        if not position:
            if score > signal_threshold:
                position = {
                    'side': 'LONG',
                    'entry_price': price,
                    'entry_time': current_time,
                    'size': capital * 0.02  # 2% капитала
                }
            elif score < -signal_threshold:
                position = {
                    'side': 'SHORT',
                    'entry_price': price,
                    'entry_time': current_time,
                    'size': capital * 0.02
                }

    # Статистика
    if not trades:
        return None

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0
    profit_factor = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0

    return {
        'total_pnl': total_pnl,
        'return_pct': (capital - 10000) / 10000 * 100,
        'trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_capital': capital
    }


def run_optimization():
    """Запуск оптимизации"""
    print("=" * 70)
    print("ОПТИМИЗАЦИЯ СТРАТЕГИИ")
    print("=" * 70)

    # Загружаем данные для всех символов
    all_data = {}
    for symbol in SYMBOLS:
        df = load_candles(symbol, '1h')
        if not df.empty:
            all_data[symbol] = df
            print(f"Загружено {symbol}: {len(df)} свечей")

    print(f"\nТестируем комбинации:")
    print(f"  Stop Loss: {STOP_LOSSES}")
    print(f"  Take Profit: {TAKE_PROFITS}")
    print(f"  Time Exit: {TIME_EXITS}")
    print(f"  Signal Threshold: {SIGNAL_THRESHOLDS}")

    total_combinations = len(STOP_LOSSES) * len(TAKE_PROFITS) * len(TIME_EXITS) * len(SIGNAL_THRESHOLDS)
    print(f"\nВсего комбинаций: {total_combinations}")
    print("\nОптимизация...\n")

    results = []
    tested = 0

    for sl, tp, time_exit, threshold in product(STOP_LOSSES, TAKE_PROFITS, TIME_EXITS, SIGNAL_THRESHOLDS):
        tested += 1
        if tested % 50 == 0:
            print(f"  Протестировано: {tested}/{total_combinations}")

        # Тестируем на всех символах
        total_return = 0
        total_trades = 0
        total_wins = 0
        symbol_results = []

        for symbol, df in all_data.items():
            result = backtest_params(df, sl, tp, time_exit, threshold)
            if result:
                total_return += result['return_pct']
                total_trades += result['trades']
                total_wins += result['trades'] * result['win_rate']
                symbol_results.append(result)

        if symbol_results:
            avg_return = total_return / len(symbol_results)
            avg_win_rate = total_wins / total_trades if total_trades > 0 else 0

            results.append({
                'stop_loss': sl,
                'take_profit': tp,
                'time_exit': time_exit,
                'threshold': threshold,
                'avg_return': avg_return,
                'total_trades': total_trades,
                'win_rate': avg_win_rate,
                'profit_factor': np.mean([r['profit_factor'] for r in symbol_results])
            })

    # Сортируем по доходности
    results.sort(key=lambda x: x['avg_return'], reverse=True)

    print("\n" + "=" * 70)
    print("ТОП-10 ЛУЧШИХ КОМБИНАЦИЙ")
    print("=" * 70)

    for i, r in enumerate(results[:10], 1):
        time_str = f"{r['time_exit']}ч" if r['time_exit'] else "∞"
        print(f"\n#{i}: Return: {r['avg_return']:+.1f}%")
        print(f"    SL: {r['stop_loss']*100:.1f}% | TP: {r['take_profit']*100:.1f}% | Time: {time_str} | Threshold: {r['threshold']}")
        print(f"    Trades: {r['total_trades']} | Win Rate: {r['win_rate']:.1%} | PF: {r['profit_factor']:.2f}")

    # Лучший результат
    best = results[0]
    print("\n" + "=" * 70)
    print("ЛУЧШИЕ ПАРАМЕТРЫ")
    print("=" * 70)
    print(f"\n  Stop Loss:     {best['stop_loss']*100:.1f}%")
    print(f"  Take Profit:   {best['take_profit']*100:.1f}%")
    print(f"  Time Exit:     {best['time_exit']}ч" if best['time_exit'] else "  Time Exit:     Без лимита")
    print(f"  Threshold:     {best['threshold']}")
    print(f"\n  Средний Return: {best['avg_return']:+.1f}%")
    print(f"  Win Rate:       {best['win_rate']:.1%}")
    print(f"  Profit Factor:  {best['profit_factor']:.2f}")

    return best


if __name__ == '__main__':
    best = run_optimization()
