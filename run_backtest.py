"""
Backtest Runner - тестирование Strategy Engine на 4 годах данных
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bots'))
sys.path.insert(0, os.path.dirname(__file__))

from backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult


class StrategyEngineAdapter:
    """
    Адаптер Strategy Engine для backtesting
    """
    warmup_period = 200  # Need 200 bars for indicators

    def __init__(self):
        self.last_signal = None

    def generate_signal(self, data: pd.DataFrame) -> dict:
        """
        Generate signal based on trend-following + mean reversion
        """
        if len(data) < self.warmup_period:
            return {'action': 'HOLD'}

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values

        current_price = close[-1]

        # Calculate indicators
        score = 0.0

        # 1. Trend Filter - EMA 200
        ema_200 = self._calculate_ema(close, 200)
        is_uptrend = current_price > ema_200
        is_downtrend = current_price < ema_200

        # Only trade in trend direction
        if not is_uptrend and not is_downtrend:
            return {'action': 'HOLD'}

        # 2. RSI - mean reversion in trend
        rsi = self._calculate_rsi(close, 14)
        if is_uptrend:
            if rsi < 40:  # Oversold in uptrend = BUY
                score += 0.4
            elif rsi > 75:  # Overbought = exit
                score -= 0.2
        else:  # downtrend
            if rsi > 60:  # Overbought in downtrend = SELL
                score -= 0.4
            elif rsi < 25:  # Oversold = exit
                score += 0.2

        # 3. MACD confirmation
        macd, signal_line = self._calculate_macd(close)
        macd_hist = macd - signal_line

        if is_uptrend and macd_hist > 0:
            score += 0.2
        elif is_downtrend and macd_hist < 0:
            score -= 0.2

        # 4. EMA Crossover
        ema_9 = self._calculate_ema(close, 9)
        ema_21 = self._calculate_ema(close, 21)

        # Check for recent crossover
        prev_ema_9 = self._calculate_ema(close[:-1], 9)
        prev_ema_21 = self._calculate_ema(close[:-1], 21)

        # Golden cross
        if prev_ema_9 < prev_ema_21 and ema_9 > ema_21 and is_uptrend:
            score += 0.3
        # Death cross
        elif prev_ema_9 > prev_ema_21 and ema_9 < ema_21 and is_downtrend:
            score -= 0.3

        # 5. Volume surge
        avg_volume = np.mean(volume[-20:])
        volume_ratio = volume[-1] / avg_volume if avg_volume > 0 else 1

        if volume_ratio > 2.0:
            # Strong volume confirms move
            if score > 0:
                score += 0.15
            elif score < 0:
                score -= 0.15

        # 6. Support/Resistance via recent swing points
        recent_high = np.max(high[-50:])
        recent_low = np.min(low[-50:])
        range_size = recent_high - recent_low

        if range_size > 0:
            position_in_range = (current_price - recent_low) / range_size

            # Near support in uptrend
            if is_uptrend and position_in_range < 0.3:
                score += 0.15
            # Near resistance in downtrend
            elif is_downtrend and position_in_range > 0.7:
                score -= 0.15

        # 7. ATR for dynamic stop-loss
        atr = self._calculate_atr(high, low, close, 14)

        # Generate action - trend following
        threshold = 0.35  # Slightly higher threshold

        if score > threshold and is_uptrend:
            return {
                'action': 'BUY',
                'score': score,
                'stop_loss': current_price - 2.5 * atr,
                'take_profit': current_price + 4 * atr,  # 1.6:1 RR
                'position_size': 0.015  # 1.5% of capital
            }
        elif score < -threshold and is_downtrend:
            return {
                'action': 'SELL',
                'score': score,
                'stop_loss': current_price + 2.5 * atr,
                'take_profit': current_price - 4 * atr,
                'position_size': 0.015
            }
        else:
            return {'action': 'HOLD', 'score': score}

    def _calculate_rsi(self, prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow

        # Simple signal line
        signal_line = np.mean(prices[-signal:]) - np.mean(prices[-signal-10:-10]) if len(prices) > signal + 10 else macd_line
        return macd_line, signal_line * 0.5

    def _calculate_ema(self, prices, period):
        if len(prices) < period:
            return prices[-1]
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema

    def _calculate_bollinger(self, prices, period=20, std_dev=2):
        if len(prices) < period:
            return prices[-1] * 1.02, prices[-1], prices[-1] * 0.98

        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def _calculate_atr(self, high, low, close, period=14):
        if len(high) < period + 1:
            return (high[-1] - low[-1])

        tr = []
        for i in range(-period, 0):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr.append(max(tr1, tr2, tr3))
        return np.mean(tr)


def load_data_from_db(symbol: str, interval: str = '1h') -> pd.DataFrame:
    """Load candle data from prices.db"""
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'prices.db')

    conn = sqlite3.connect(db_path)

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

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop('open_time', axis=1, inplace=True)

    return df


def run_full_backtest():
    """Run backtest on all symbols"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

    print("=" * 80)
    print("BACKTEST: Strategy Engine на 4+ годах данных")
    print("=" * 80)

    # Config - optimized settings
    config = BacktestConfig(
        initial_capital=10000.0,
        fee_rate=0.001,  # 0.1% Binance fee
        slippage_pct=0.0003,  # 0.03% slippage
        max_position_pct=0.02,  # 2% per trade
        max_positions=5,
        default_stop_loss_pct=0.03,  # 3% stop loss
        default_take_profit_pct=0.06,  # 6% take profit (2:1 RR)
        use_trailing_stop=True,
        trailing_stop_pct=0.02,
        max_daily_loss_pct=1.0,  # 100% - disabled
        max_consecutive_losses=100  # Effectively disabled
    )

    all_results = {}
    total_pnl = 0
    total_trades = 0

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"{'='*60}")

        # Load data
        df = load_data_from_db(symbol, '1h')

        if df.empty:
            print(f"  No data for {symbol}")
            continue

        print(f"  Data: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Candles: {len(df):,}")
        print(f"  Years: {(df.index[-1] - df.index[0]).days / 365:.1f}")

        # Create strategy
        strategy = StrategyEngineAdapter()

        # Run backtest
        engine = BacktestEngine(config)
        result = engine.run(strategy, df, symbol)

        all_results[symbol] = result
        total_pnl += result.total_pnl
        total_trades += result.total_trades

        # Print results
        print(f"\n  RESULTS:")
        print(f"  ├─ Trades: {result.total_trades}")
        print(f"  ├─ Win Rate: {result.win_rate:.1%}")
        print(f"  ├─ Total P&L: ${result.total_pnl:,.2f}")
        print(f"  ├─ Return: {result.total_return_pct:.2%}")
        print(f"  ├─ Profit Factor: {result.profit_factor:.2f}")
        print(f"  ├─ Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  ├─ Max Drawdown: {result.max_drawdown_pct:.2%}")
        print(f"  ├─ Avg Win: ${result.avg_win:.2f}")
        print(f"  ├─ Avg Loss: ${result.avg_loss:.2f}")
        print(f"  └─ Final Equity: ${result.final_equity:,.2f}")

    # Portfolio summary
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)

    avg_return = np.mean([r.total_return_pct for r in all_results.values()])
    avg_sharpe = np.mean([r.sharpe_ratio for r in all_results.values()])
    avg_winrate = np.mean([r.win_rate for r in all_results.values()])
    avg_drawdown = np.mean([r.max_drawdown_pct for r in all_results.values()])

    print(f"\n  Initial Capital: $10,000 x 5 = $50,000")
    print(f"  Total Trades: {total_trades}")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Avg Return per Symbol: {avg_return:.2%}")
    print(f"  Avg Win Rate: {avg_winrate:.1%}")
    print(f"  Avg Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"  Avg Max Drawdown: {avg_drawdown:.2%}")

    # Save results
    save_results(all_results)

    return all_results


def save_results(results: dict):
    """Save backtest results to file"""
    output_dir = os.path.join(os.path.dirname(__file__), 'backtest_results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Summary CSV
    summary_data = []
    for symbol, result in results.items():
        summary_data.append({
            'symbol': symbol,
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'total_pnl': result.total_pnl,
            'return_pct': result.total_return_pct,
            'profit_factor': result.profit_factor,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown_pct,
            'final_equity': result.final_equity
        })

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'backtest_summary_{timestamp}.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Results saved to: {summary_path}")

    # Equity curves
    for symbol, result in results.items():
        if result.equity_curve:
            eq_path = os.path.join(output_dir, f'equity_{symbol}_{timestamp}.csv')
            pd.DataFrame({'equity': result.equity_curve}).to_csv(eq_path, index=False)


if __name__ == '__main__':
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO", format="{message}")

    run_full_backtest()
