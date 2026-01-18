#!/usr/bin/env python3
"""
Trading Bot System - Main Entry Point
"""
import os
import sys
import argparse
import asyncio
from datetime import datetime
from loguru import logger

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import config
from core.binance_client import BinanceDataLoader
from data.data_manager import DataManager
from core.paper_trading import PaperTradingEngine
from bots.strategies import get_strategy, STRATEGIES


def setup_logging():
    """Configure logging"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    logger.add(os.path.join(log_dir, "trading_{time}.log"), rotation="1 day", retention="7 days", level="DEBUG")


def download_data(symbols: list = None, intervals: list = None, start_date: str = '2022-01-01'):
    """Download historical data from Binance"""
    setup_logging()
    
    if not symbols:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
            'MATICUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'APTUSDT',
            'ARBUSDT', 'OPUSDT', 'INJUSDT', 'NEARUSDT', 'FILUSDT'
        ]
    
    if not intervals:
        intervals = ['5m', '15m', '1h', '4h', '1d']
    
    logger.info("=" * 60)
    logger.info("STARTING DATA DOWNLOAD")
    logger.info("=" * 60)
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Start date: {start_date}")
    logger.info("=" * 60)
    
    manager = DataManager()
    
    def progress_callback(info):
        logger.info(f"[{info['completed']}/{info['total']}] {info['symbol']} {info['interval']} - {info['progress']:.1f}%")
    
    stats = manager.download_all_data(
        symbols=symbols,
        intervals=intervals,
        start_date=start_date,
        progress_callback=progress_callback
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    
    data_stats = manager.get_data_stats()
    logger.info(f"Total symbols: {data_stats['symbols_count']}")
    logger.info(f"Total candles: {data_stats['total_candles']:,}")
    logger.info(f"Total size: {data_stats['total_size_mb']} MB")
    
    return stats


def run_backtest(symbol: str = 'BTCUSDT', interval: str = '1h', 
                 strategy_name: str = 'mean_reversion',
                 start_date: str = '2023-01-01', end_date: str = None):
    """Run a backtest"""
    setup_logging()
    
    from backtesting.engine import BacktestEngine, BacktestConfig
    
    logger.info("=" * 60)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Period: {start_date} to {end_date or 'now'}")
    logger.info("=" * 60)
    
    # Load data
    manager = DataManager()
    df = manager.prepare_backtest_data(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        add_features=True
    )
    
    if df.empty:
        logger.error("No data available. Run download first.")
        return None
    
    logger.info(f"Loaded {len(df)} candles")
    
    # Create strategy
    strategy = get_strategy(strategy_name)
    
    # Run backtest
    bt_config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.001,
        max_position_pct=0.02
    )
    
    engine = BacktestEngine(bt_config)
    result = engine.run(strategy, df, symbol)
    
    # Print results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total trades: {result.total_trades}")
    logger.info(f"Win rate: {result.win_rate:.1%}")
    logger.info(f"Total return: {result.total_return_pct:.2%}")
    logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"Max drawdown: {result.max_drawdown_pct:.2%}")
    logger.info(f"Profit factor: {result.profit_factor:.2f}")
    logger.info(f"Final equity: ${result.final_equity:,.2f}")
    
    return result


def run_dashboard(host: str = '0.0.0.0', port: int = 5000):
    """Run web dashboard"""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("STARTING WEB DASHBOARD")
    logger.info("=" * 60)
    logger.info(f"URL: http://{host}:{port}")
    logger.info("=" * 60)
    
    from dashboard.app import run_server
    run_server(host=host, port=port, debug=True)


def run_paper_trading(symbols: list = None):
    """Run paper trading bot"""
    setup_logging()
    
    if not symbols:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    
    logger.info("=" * 60)
    logger.info("STARTING PAPER TRADING BOT")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info("=" * 60)
    
    import asyncio
    from core.binance_client import BinanceDataLoader
    
    engine = PaperTradingEngine(initial_capital=10000)
    loader = BinanceDataLoader()
    
    # Add strategies
    for symbol in symbols:
        strategy = get_strategy('combined')
        engine.add_strategy(symbol, strategy)
    
    # Main loop
    async def trading_loop():
        while True:
            for symbol in symbols:
                try:
                    # Get current price
                    ticker = loader.client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    
                    # Update engine
                    engine.update_price(symbol, price)
                    
                    # Get historical data for signal
                    df = loader.get_historical_klines(
                        symbol=symbol,
                        interval='1h',
                        start_date='2024-01-01'
                    )
                    
                    if not df.empty and symbol in engine.strategies:
                        signal = engine.strategies[symbol].generate_signal(df)
                        engine.process_signal(symbol, signal)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Status update
            status = engine.get_status()
            logger.info(f"Equity: ${status['equity']:,.2f} | Positions: {status['positions_count']} | Trades: {status['trades_count']}")
            
            # Wait before next iteration
            await asyncio.sleep(60)  # Check every minute
    
    try:
        asyncio.run(trading_loop())
    except KeyboardInterrupt:
        logger.info("Paper trading stopped")


def main():
    parser = argparse.ArgumentParser(description='Trading Bot System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download historical data')
    download_parser.add_argument('--symbols', nargs='+', help='Symbols to download')
    download_parser.add_argument('--intervals', nargs='+', default=['1h', '4h', '1d'], help='Intervals')
    download_parser.add_argument('--start', default='2022-01-01', help='Start date (YYYY-MM-DD)')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--symbol', default='BTCUSDT', help='Symbol')
    backtest_parser.add_argument('--interval', default='1h', help='Interval')
    backtest_parser.add_argument('--strategy', default='mean_reversion', choices=list(STRATEGIES.keys()))
    backtest_parser.add_argument('--start', default='2023-01-01', help='Start date')
    backtest_parser.add_argument('--end', help='End date')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run web dashboard')
    dashboard_parser.add_argument('--host', default='0.0.0.0', help='Host')
    dashboard_parser.add_argument('--port', type=int, default=5000, help='Port')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        download_data(args.symbols, args.intervals, args.start)
    elif args.command == 'backtest':
        run_backtest(args.symbol, args.interval, args.strategy, args.start, args.end)
    elif args.command == 'dashboard':
        run_dashboard(args.host, args.port)
    elif args.command == 'paper':
        run_paper_trading(args.symbols)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
