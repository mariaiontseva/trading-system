#!/usr/bin/env python3
"""
Run backtests across all strategies and symbols to find best parameters
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_manager import DataManager
from backtesting.engine import BacktestEngine, BacktestConfig
from bots.strategies import get_strategy, STRATEGIES
import json

def run_comprehensive_backtests():
    dm = DataManager()

    # Top symbols by market cap
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
               'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']

    intervals = ['1h', '4h']
    strategies = ['mean_reversion', 'trend_following', 'combined']

    results = []

    print("=" * 80)
    print("COMPREHENSIVE BACKTEST ANALYSIS")
    print("=" * 80)

    for strategy_name in strategies:
        strategy = get_strategy(strategy_name)
        print(f"\n{'='*40}")
        print(f"STRATEGY: {strategy_name.upper()}")
        print(f"{'='*40}")

        strategy_results = []

        for symbol in symbols:
            for interval in intervals:
                # Load data
                df = dm.prepare_backtest_data(
                    symbol=symbol,
                    interval=interval,
                    start_date='2022-01-01',
                    end_date='2026-01-17',
                    add_features=True
                )

                if df.empty or len(df) < 100:
                    continue

                # Run backtest
                config = BacktestConfig(
                    initial_capital=10000,
                    fee_rate=0.001,
                    max_position_pct=0.02
                )

                engine = BacktestEngine(config)
                result = engine.run(strategy, df, symbol)

                result_data = {
                    'symbol': symbol,
                    'interval': interval,
                    'strategy': strategy_name,
                    'total_return': result.total_return_pct,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown_pct,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'profit_factor': result.profit_factor
                }

                strategy_results.append(result_data)
                results.append(result_data)

                # Print summary
                print(f"\n{symbol} ({interval}):")
                print(f"  Return: {result.total_return_pct:.2f}%")
                print(f"  Sharpe: {result.sharpe_ratio:.2f}")
                print(f"  Max DD: {result.max_drawdown_pct:.2f}%")
                print(f"  Win Rate: {result.win_rate*100:.1f}%")
                print(f"  Trades: {result.total_trades}")

        # Strategy summary
        if strategy_results:
            avg_return = sum(r['total_return'] for r in strategy_results) / len(strategy_results)
            avg_sharpe = sum(r['sharpe_ratio'] for r in strategy_results) / len(strategy_results)
            avg_winrate = sum(r['win_rate'] for r in strategy_results) / len(strategy_results)

            print(f"\n--- {strategy_name.upper()} SUMMARY ---")
            print(f"Avg Return: {avg_return:.2f}%")
            print(f"Avg Sharpe: {avg_sharpe:.2f}")
            print(f"Avg Win Rate: {avg_winrate*100:.1f}%")

    # Find best combinations
    print("\n" + "=" * 80)
    print("TOP 10 BEST PERFORMING COMBINATIONS")
    print("=" * 80)

    # Sort by Sharpe ratio (risk-adjusted return)
    sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)

    for i, r in enumerate(sorted_results[:10], 1):
        print(f"\n{i}. {r['symbol']} | {r['interval']} | {r['strategy']}")
        print(f"   Return: {r['total_return']:.2f}% | Sharpe: {r['sharpe_ratio']:.2f} | "
              f"Win: {r['win_rate']*100:.1f}% | DD: {r['max_drawdown']:.2f}%")

    # Save results
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to backtest_results.json")

    return sorted_results[:10]

if __name__ == '__main__':
    best = run_comprehensive_backtests()
