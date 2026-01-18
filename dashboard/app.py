"""
Web Dashboard - Flask application
"""
import os
import sys
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from bots.strategies import get_strategy, STRATEGIES


app = Flask(__name__)
app.config['SECRET_KEY'] = config.dashboard.secret_key
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Lazy initialization of components to avoid async conflicts
_paper_engine = None
_data_manager = None
_binance_loader = None


def get_paper_engine():
    global _paper_engine
    if _paper_engine is None:
        from core.paper_trading import PaperTradingEngine
        _paper_engine = PaperTradingEngine(initial_capital=10000.0)
        _paper_engine.add_callback(on_paper_event)
    return _paper_engine


def get_data_manager():
    global _data_manager
    if _data_manager is None:
        from data.data_manager import DataManager
        _data_manager = DataManager()
    return _data_manager


def get_binance_loader():
    global _binance_loader
    if _binance_loader is None:
        from core.binance_client import BinanceDataLoader
        _binance_loader = BinanceDataLoader()
    return _binance_loader


# WebSocket event handler for paper trading
def on_paper_event(event_type: str, data: dict):
    """Broadcast paper trading events to clients"""
    socketio.emit('paper_event', {'type': event_type, 'data': data})


# ============== Routes ==============

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'paper_trading': get_paper_engine().get_status(),
        'data': get_data_manager().get_data_stats(),
        'strategies': list(STRATEGIES.keys()),
        'symbols': config.trading.symbols
    })


# ============== Paper Trading API ==============

@app.route('/api/paper/status')
def paper_status():
    """Get paper trading status"""
    return jsonify(get_paper_engine().get_status())


@app.route('/api/paper/positions')
def paper_positions():
    """Get open positions"""
    return jsonify([p.to_dict() for p in get_paper_engine().positions.values()])


@app.route('/api/paper/trades')
def paper_trades():
    """Get trade history"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify([t.to_dict() for t in get_paper_engine().trades[:limit]])


@app.route('/api/paper/equity')
def paper_equity():
    """Get equity history"""
    days = request.args.get('days', 30, type=int)
    return jsonify(get_paper_engine().get_equity_history(days))


@app.route('/api/paper/open', methods=['POST'])
def paper_open():
    """Open a position"""
    data = request.json
    symbol = data['symbol']

    # Auto-fetch current price before opening position
    try:
        loader = get_binance_loader()
        ticker = loader.client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        get_paper_engine().update_price(symbol, price)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to fetch price: {e}'}), 400

    result = get_paper_engine().open_position(
        symbol=symbol,
        side=data['side'],
        size_usd=data.get('size_usd'),
        stop_loss=data.get('stop_loss'),
        take_profit=data.get('take_profit')
    )
    if result:
        return jsonify({'success': True, 'position': result.to_dict()})
    return jsonify({'success': False, 'error': 'Failed to open position'}), 400


@app.route('/api/paper/close', methods=['POST'])
def paper_close():
    """Close a position"""
    data = request.json
    symbol = data['symbol']

    # Auto-fetch current price before closing position
    try:
        loader = get_binance_loader()
        ticker = loader.client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        get_paper_engine().update_price(symbol, price)
    except Exception as e:
        pass  # Use last known price if fetch fails

    result = get_paper_engine().close_position(
        symbol=symbol,
        reason=data.get('reason', 'MANUAL')
    )
    if result:
        return jsonify({'success': True, 'trade': result.to_dict()})
    return jsonify({'success': False, 'error': 'Failed to close position'}), 400


@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    """Reset paper trading"""
    get_paper_engine().reset()
    return jsonify({'success': True})


@app.route('/api/paper/deposit', methods=['POST'])
def paper_deposit():
    """Deposit funds"""
    data = request.json
    engine = get_paper_engine()
    engine.deposit(data['amount'])
    return jsonify({'success': True, 'capital': engine.capital})


# ============== Bot Signals API ==============

@app.route('/api/bot/signals')
def bot_signals():
    """Get signals from all bots"""
    try:
        import sys
        import os
        bots_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bots')
        if bots_path not in sys.path:
            sys.path.insert(0, bots_path)

        from strategy_engine import StrategyEngine

        engine = StrategyEngine()
        signals = engine.analyze_all('1h')

        result = {}
        for sig in signals:
            result[sig.symbol] = {
                'decision': sig.decision.name,
                'score': sig.score,
                'confidence': sig.confidence,
                'entry_price': sig.entry_price,
                'stop_loss': sig.stop_loss,
                'take_profit': sig.take_profit,
                'reasons': sig.reasons
            }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting bot signals: {e}")
        # Return mock data if bots not available
        return jsonify({
            'BTCUSDT': {'score': -0.14, 'decision': 'HOLD', 'reasons': ['Tech: SELL']},
            'ETHUSDT': {'score': 0.03, 'decision': 'HOLD', 'reasons': ['Volume: Accumulation']},
            'BNBUSDT': {'score': -0.15, 'decision': 'HOLD', 'reasons': ['Pattern: Shooting Star']},
            'SOLUSDT': {'score': -0.12, 'decision': 'HOLD', 'reasons': ['Volume: Distribution']},
            'ADAUSDT': {'score': -0.32, 'decision': 'SELL', 'reasons': ['Tech: SELL', 'Volume: Distribution']}
        })


# ============== Data API ==============

@app.route('/api/data/available')
def data_available():
    """Get available data"""
    return jsonify(get_data_manager().get_available_data())


@app.route('/api/data/stats')
def data_stats():
    """Get data statistics"""
    return jsonify(get_data_manager().get_data_stats())


@app.route('/api/data/download', methods=['POST'])
def data_download():
    """Start data download"""
    data = request.json
    symbols = data.get('symbols', config.trading.symbols)
    intervals = data.get('intervals', ['1h', '4h', '1d'])
    start_date = data.get('start_date', '2023-01-01')

    dm = get_data_manager()

    # Run in background
    def download_task():
        stats = dm.download_all_data(
            symbols=symbols,
            intervals=intervals,
            start_date=start_date,
            progress_callback=lambda info: socketio.emit('download_progress', info)
        )
        socketio.emit('download_complete', stats)

    socketio.start_background_task(download_task)
    return jsonify({'success': True, 'message': 'Download started'})


@app.route('/api/data/candles/<symbol>/<interval>')
def get_candles(symbol, interval):
    """Get candle data from prices.db"""
    import sqlite3

    limit = request.args.get('limit', 100, type=int)

    # Connect to prices.db where Price Scanner stores data
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'prices.db')

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get latest candles
        cursor.execute('''
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time DESC
            LIMIT ?
        ''', (symbol, interval, limit))

        rows = cursor.fetchall()
        conn.close()

        # Convert to list of dicts (reverse to chronological order)
        candles = []
        for row in reversed(rows):
            candles.append({
                'timestamp': datetime.fromtimestamp(row[0] / 1000).isoformat(),
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5]
            })

        return jsonify(candles)
    except Exception as e:
        logger.error(f"Error loading candles: {e}")
        return jsonify([])


# ============== Backtest API ==============

@app.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """Run backtest"""
    from backtesting.engine import BacktestEngine, BacktestConfig

    data = request.json

    symbol = data.get('symbol', 'BTCUSDT')
    interval = data.get('interval', '1h')
    strategy_name = data.get('strategy', 'mean_reversion')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date')

    # Load data
    df = get_data_manager().prepare_backtest_data(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        add_features=True
    )

    if df.empty:
        return jsonify({'success': False, 'error': 'No data available'}), 400

    # Create strategy
    strategy = get_strategy(strategy_name)

    # Create backtest config
    bt_config = BacktestConfig(
        initial_capital=data.get('initial_capital', 10000),
        fee_rate=data.get('fee_rate', 0.001),
        max_position_pct=data.get('max_position_pct', 0.02)
    )

    # Run backtest
    engine = BacktestEngine(bt_config)
    result = engine.run(strategy, df, symbol)

    return jsonify({
        'success': True,
        'result': result.to_dict(),
        'equity_curve': result.equity_curve[::10],  # Downsample
        'trades': result.trades[:50]  # Limit trades
    })


@app.route('/api/backtest/strategies')
def list_strategies():
    """List available strategies"""
    return jsonify(list(STRATEGIES.keys()))


# ============== Price API ==============

@app.route('/api/price/<symbol>')
def get_price(symbol):
    """Get current price"""
    try:
        loader = get_binance_loader()
        price = loader.client.get_symbol_ticker(symbol=symbol)
        return jsonify({
            'symbol': symbol,
            'price': float(price['price']),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/prices')
def get_prices():
    """Get all prices"""
    symbols = request.args.get('symbols', ','.join(config.trading.symbols))
    symbols = symbols.split(',')

    loader = get_binance_loader()
    prices = {}
    for symbol in symbols:
        try:
            ticker = loader.client.get_symbol_ticker(symbol=symbol)
            prices[symbol] = float(ticker['price'])
        except:
            pass

    return jsonify(prices)


# ============== WebSocket Events ==============

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('connected', {'status': 'ok'})


@socketio.on('subscribe_prices')
def handle_subscribe(data):
    """Subscribe to price updates"""
    symbols = data.get('symbols', config.trading.symbols)
    logger.info(f"Client subscribed to prices: {symbols}")

    # Start price update loop
    def price_loop():
        loader = get_binance_loader()
        engine = get_paper_engine()
        while True:
            prices = {}
            for symbol in symbols:
                try:
                    ticker = loader.client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    prices[symbol] = price
                    engine.update_price(symbol, price)
                except:
                    pass

            socketio.emit('prices', prices)
            socketio.sleep(5)  # Update every 5 seconds

    socketio.start_background_task(price_loop)


@socketio.on('update_price')
def handle_price_update(data):
    """Manual price update (for testing)"""
    symbol = data['symbol']
    price = data['price']
    get_paper_engine().update_price(symbol, price)


# ============== Auto Trading ==============

_auto_trading_enabled = False
_auto_trading_task = None


def get_strategy_engine():
    """Get strategy engine for signal generation"""
    try:
        import sys
        bots_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bots')
        if bots_path not in sys.path:
            sys.path.insert(0, bots_path)
        from strategy_engine import StrategyEngine
        return StrategyEngine()
    except Exception as e:
        logger.error(f"Failed to load strategy engine: {e}")
        return None


def calculate_live_signal(closes: list, volumes: list) -> float:
    """Calculate trading signal from live data"""
    import numpy as np

    if len(closes) < 50:
        return 0.0

    closes = np.array(closes)
    volumes = np.array(volumes)
    score = 0.0

    # RSI
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
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

    # Price momentum
    momentum = (closes[-1] - closes[-10]) / closes[-10]
    if momentum > 0.02:
        score += 0.15
    elif momentum < -0.02:
        score -= 0.15

    return max(-1.0, min(1.0, score))


def auto_trading_loop():
    """Background task for automatic trading based on bot signals"""
    global _auto_trading_enabled

    logger.info("Auto trading loop started")
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

    while _auto_trading_enabled:
        try:
            paper = get_paper_engine()
            loader = get_binance_loader()

            for symbol in symbols:
                # Get live candles from Binance
                try:
                    klines = loader.client.get_klines(symbol=symbol, interval='1h', limit=100)
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    price = closes[-1]
                    paper.update_price(symbol, price)
                except Exception as e:
                    logger.error(f"Failed to get klines for {symbol}: {e}")
                    continue

                # Calculate simple indicators
                score = calculate_live_signal(closes, volumes)

                # Check if we already have a position
                has_position = symbol in paper.positions

                logger.info(f"{symbol}: price={price:.2f}, score={score:.2f}, has_pos={has_position}")

                # BUY signal (score > 0.15) and no position
                if score > 0.15 and not has_position:
                    size_usd = paper.capital * 0.02

                    result = paper.open_position(
                        symbol=symbol,
                        side='LONG',
                        size_usd=size_usd
                    )
                    if result:
                        logger.info(f"AUTO: Opened LONG {symbol} at {price}, score={score:.2f}")
                        socketio.emit('auto_trade', {
                            'action': 'OPEN',
                            'symbol': symbol,
                            'side': 'LONG',
                            'price': price,
                            'score': score
                        })

                # SELL signal (score < -0.15) and no position
                elif score < -0.15 and not has_position:
                    size_usd = paper.capital * 0.02

                    result = paper.open_position(
                        symbol=symbol,
                        side='SHORT',
                        size_usd=size_usd
                    )
                    if result:
                        logger.info(f"AUTO: Opened SHORT {symbol} at {price}, score={score:.2f}")
                        socketio.emit('auto_trade', {
                            'action': 'OPEN',
                            'symbol': symbol,
                            'side': 'SHORT',
                            'price': price,
                            'score': score
                        })

                # Close position if signal reversed
                elif has_position:
                    position = paper.positions[symbol]
                    # Close LONG if signal turns negative
                    if position.side == 'LONG' and score < -0.1:
                        result = paper.close_position(symbol, reason='SIGNAL_REVERSED')
                        if result:
                            logger.info(f"AUTO: Closed LONG {symbol}, P&L: ${result.pnl:.2f}")
                            socketio.emit('auto_trade', {
                                'action': 'CLOSE',
                                'symbol': symbol,
                                'pnl': result.pnl
                            })
                    # Close SHORT if signal turns positive
                    elif position.side == 'SHORT' and score > 0.1:
                        result = paper.close_position(symbol, reason='SIGNAL_REVERSED')
                        if result:
                            logger.info(f"AUTO: Closed SHORT {symbol}, P&L: ${result.pnl:.2f}")
                            socketio.emit('auto_trade', {
                                'action': 'CLOSE',
                                'symbol': symbol,
                                'pnl': result.pnl
                            })

            # Wait before next check (every 30 seconds)
            socketio.sleep(30)

        except Exception as e:
            logger.error(f"Auto trading error: {e}")
            socketio.sleep(30)

    logger.info("Auto trading loop stopped")


@app.route('/api/auto/start', methods=['POST'])
def start_auto_trading():
    """Start automatic trading"""
    global _auto_trading_enabled, _auto_trading_task

    if _auto_trading_enabled:
        return jsonify({'success': True, 'message': 'Already running'})

    _auto_trading_enabled = True
    _auto_trading_task = socketio.start_background_task(auto_trading_loop)

    logger.info("Auto trading ENABLED")
    return jsonify({'success': True, 'message': 'Auto trading started'})


@app.route('/api/auto/stop', methods=['POST'])
def stop_auto_trading():
    """Stop automatic trading"""
    global _auto_trading_enabled

    _auto_trading_enabled = False
    logger.info("Auto trading DISABLED")
    return jsonify({'success': True, 'message': 'Auto trading stopped'})


@app.route('/api/auto/status')
def auto_trading_status():
    """Get auto trading status"""
    return jsonify({
        'enabled': _auto_trading_enabled,
        'message': 'Running' if _auto_trading_enabled else 'Stopped'
    })


# ============== Templates ==============

# Create templates directory
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(templates_dir, exist_ok=True)

# Create static directory
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)


def run_server(host='0.0.0.0', port=None, debug=False):
    """Run the server"""
    # Use PORT env variable for cloud deployment
    if port is None:
        port = int(os.environ.get('PORT', 5002))

    # Auto-start trading on server start
    global _auto_trading_enabled, _auto_trading_task
    if os.environ.get('AUTO_TRADE', 'true').lower() == 'true':
        _auto_trading_enabled = True
        _auto_trading_task = socketio.start_background_task(auto_trading_loop)
        logger.info("Auto trading ENABLED on startup")

    logger.info(f"Starting server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server()
