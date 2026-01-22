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
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Trading symbols (15 coins)
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',  # Original 5
    'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',  # New 5
    'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'NEARUSDT'   # New 5
]

# Strategy parameters (optimized)
RSI_LONG = 42    # Was 40
RSI_SHORT = 58   # Was 60
ADX_MIN = 18     # Was 20
EMA_PERIOD = 100

# Timeframes
TIMEFRAMES = ['15m', '1h']  # 15m for entries, 1h for confirmation

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
    """Get open positions with real-time prices"""
    engine = get_paper_engine()

    # Update prices for all open positions
    if engine.positions:
        try:
            loader = get_binance_loader()
            for symbol, position in engine.positions.items():
                try:
                    ticker = loader.client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    position.update_price(price)
                except:
                    pass
        except:
            pass

    response = jsonify([p.to_dict() for p in engine.positions.values()])
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


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


@app.route('/api/paper/restore', methods=['POST'])
def paper_restore():
    """Restore historical trades (for recovery after deploy)"""
    from core.paper_trading import PaperTrade
    data = request.json
    trades = data.get('trades', [])
    engine = get_paper_engine()

    restored = 0
    for t in trades:
        try:
            trade = PaperTrade(
                id=t['id'],
                symbol=t['symbol'],
                side=t['side'],
                entry_time=datetime.fromisoformat(t['entry_time']),
                exit_time=datetime.fromisoformat(t['exit_time']),
                entry_price=t['entry_price'],
                exit_price=t['exit_price'],
                quantity=t['quantity'],
                size_usd=t.get('size_usd', 200),
                pnl=t['pnl'],
                pnl_pct=t['pnl_pct'],
                exit_reason=t['exit_reason'],
                fees=t.get('fees', 0)
            )
            engine._save_trade(trade)
            engine.trades.insert(0, trade)
            engine.capital += trade.pnl
            restored += 1
        except Exception as e:
            logger.error(f"Error restoring trade: {e}")

    engine._save_state()
    logger.info(f"Restored {restored} trades, capital: ${engine.capital:.2f}")
    return jsonify({'restored': restored, 'capital': engine.capital})


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


def calculate_rsi(closes, period=14):
    """Calculate RSI"""
    import numpy as np
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def calculate_ema(prices, period):
    """Calculate EMA"""
    import numpy as np
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    mult = 2.0 / (period + 1)
    for i in range(1, len(prices)):
        ema[i] = prices[i] * mult + ema[i-1] * (1 - mult)
    return ema


def calculate_atr(highs, lows, closes, period=14):
    """Calculate ATR for position sizing"""
    import numpy as np
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)

    atr = np.mean(tr[:period])
    for i in range(period, len(closes)):
        atr = (atr * (period-1) + tr[i]) / period
    return atr


def calculate_adx(highs, lows, closes, period=14):
    """Calculate ADX for trend strength filter"""
    import numpy as np
    n = len(closes)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

    atr_safe = np.where(atr == 0, 0.0001, atr)

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
    if 2*period < n:
        adx[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

    return adx[-1] if len(adx) > 0 else 0


def calculate_live_signal_multitf(loader, symbol: str) -> tuple:
    """
    MULTI-TIMEFRAME STRATEGY: 15m entry + 1h trend confirmation

    - 15m: Entry signals (RSI + ADX) - 4x more opportunities
    - 1h: Trend confirmation (Price vs EMA100)
    - Only enter when both timeframes agree

    Returns: (signal, atr, rsi_15m, rsi_1h)
    """
    import numpy as np

    try:
        # Get 1h data for trend confirmation
        klines_1h = loader.client.get_klines(symbol=symbol, interval='1h', limit=120)
        closes_1h = np.array([float(k[4]) for k in klines_1h])
        highs_1h = np.array([float(k[2]) for k in klines_1h])
        lows_1h = np.array([float(k[3]) for k in klines_1h])

        ema_100_1h = calculate_ema(closes_1h, 100)
        trend_ema = ema_100_1h[-1]

        # Get 15m data for entry signals
        klines_15m = loader.client.get_klines(symbol=symbol, interval='15m', limit=120)
        closes_15m = np.array([float(k[4]) for k in klines_15m])
        highs_15m = np.array([float(k[2]) for k in klines_15m])
        lows_15m = np.array([float(k[3]) for k in klines_15m])

        rsi_15m = calculate_rsi(closes_15m, 14)
        atr_15m = calculate_atr(highs_15m, lows_15m, closes_15m, 14)
        adx_15m = calculate_adx(highs_15m, lows_15m, closes_15m, 14)
        rsi_1h = calculate_rsi(closes_1h, 14)

        price = closes_15m[-1]

        # 1h trend direction
        trend_bullish = price > trend_ema
        trend_bearish = price < trend_ema

        # 15m entry signal
        signal_15m = 0
        if adx_15m >= ADX_MIN:
            if rsi_15m < RSI_LONG:
                signal_15m = 1  # LONG signal
            elif rsi_15m > RSI_SHORT:
                signal_15m = -1  # SHORT signal

        # Combined signal: only enter if 15m and 1h agree
        signal = 0
        if signal_15m == 1 and trend_bullish:
            signal = 1  # LONG confirmed
        elif signal_15m == -1 and trend_bearish:
            signal = -1  # SHORT confirmed

        return signal, atr_15m, rsi_15m, rsi_1h, price

    except Exception as e:
        logger.error(f"Error calculating signal for {symbol}: {e}")
        return 0, 0, 50, 50, 0


def calculate_live_signal(closes: list, highs: list, lows: list) -> tuple:
    """
    LEGACY: Single timeframe signal (1h only)
    Kept for backwards compatibility
    """
    import numpy as np

    if len(closes) < 100:
        return 0, 0, 0

    closes = np.array(closes)
    highs = np.array(highs)
    lows = np.array(lows)

    rsi = calculate_rsi(closes, 14)
    ema_100 = calculate_ema(closes, 100)
    atr = calculate_atr(highs, lows, closes, 14)
    adx = calculate_adx(highs, lows, closes, 14)

    price = closes[-1]
    trend_ema = ema_100[-1]

    signal = 0

    if adx < ADX_MIN:
        return 0, atr, rsi

    if rsi < RSI_LONG and price > trend_ema:
        signal = 1
    elif rsi > RSI_SHORT and price < trend_ema:
        signal = -1

    return signal, atr, rsi


def auto_trading_loop():
    """
    MULTI-TIMEFRAME Auto Trading Loop

    Strategy: 15m entry + 1h trend confirmation
    - 15m: RSI + ADX for entry signals (4x more opportunities)
    - 1h: EMA100 trend confirmation
    - Only enter when both timeframes agree

    Parameters:
    - RSI Long: < 42, RSI Short: > 58
    - ADX threshold: 18
    - Stop: 2.0 × ATR
    - Target: 4.0 × ATR
    """
    global _auto_trading_enabled

    logger.info("Auto trading loop started - MULTI-TIMEFRAME (15m + 1h)")
    symbols = SYMBOLS

    # Strategy parameters
    STOP_ATR_MULT = 2.0
    TARGET_ATR_MULT = 4.0
    POSITION_SIZE_PCT = 0.02  # 2% of capital per trade

    while _auto_trading_enabled:
        try:
            paper = get_paper_engine()
            loader = get_binance_loader()

            for symbol in symbols:
                try:
                    # Use multi-timeframe signal calculation
                    signal, atr, rsi_15m, rsi_1h, price = calculate_live_signal_multitf(loader, symbol)
                    paper.update_price(symbol, price)
                    rsi = rsi_15m  # Use 15m RSI for logging
                except Exception as e:
                    logger.error(f"Failed to get signal for {symbol}: {e}")
                    continue

                # Check if we already have a position
                has_position = symbol in paper.positions

                logger.info(f"{symbol}: price={price:.2f}, RSI={rsi:.1f}, signal={signal}, ATR={atr:.2f}")

                # LONG signal and no position
                if signal == 1 and not has_position:
                    size_usd = paper.capital * POSITION_SIZE_PCT
                    stop_loss = price - atr * STOP_ATR_MULT
                    take_profit = price + atr * TARGET_ATR_MULT

                    result = paper.open_position(
                        symbol=symbol,
                        side='LONG',
                        size_usd=size_usd,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    if result:
                        logger.info(f"AUTO: Opened LONG {symbol} at {price}, SL={stop_loss:.2f}, TP={take_profit:.2f}, RSI={rsi:.1f}")
                        socketio.emit('auto_trade', {
                            'action': 'OPEN',
                            'symbol': symbol,
                            'side': 'LONG',
                            'price': price,
                            'rsi': rsi,
                            'atr': atr
                        })

                # SHORT signal and no position
                elif signal == -1 and not has_position:
                    size_usd = paper.capital * POSITION_SIZE_PCT
                    stop_loss = price + atr * STOP_ATR_MULT
                    take_profit = price - atr * TARGET_ATR_MULT

                    result = paper.open_position(
                        symbol=symbol,
                        side='SHORT',
                        size_usd=size_usd,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    if result:
                        logger.info(f"AUTO: Opened SHORT {symbol} at {price}, SL={stop_loss:.2f}, TP={take_profit:.2f}, RSI={rsi:.1f}")
                        socketio.emit('auto_trade', {
                            'action': 'OPEN',
                            'symbol': symbol,
                            'side': 'SHORT',
                            'price': price,
                            'rsi': rsi,
                            'atr': atr
                        })

                # Check existing position for exit conditions
                elif has_position:
                    position = paper.positions[symbol]

                    # Get current price and check against SL/TP
                    should_close = False
                    reason = None

                    if hasattr(position, 'stop_loss') and hasattr(position, 'take_profit'):
                        if position.side == 'LONG':
                            if price <= position.stop_loss:
                                should_close = True
                                reason = 'STOP_LOSS'
                            elif price >= position.take_profit:
                                should_close = True
                                reason = 'TAKE_PROFIT'
                        else:  # SHORT
                            if price >= position.stop_loss:
                                should_close = True
                                reason = 'STOP_LOSS'
                            elif price <= position.take_profit:
                                should_close = True
                                reason = 'TAKE_PROFIT'

                    # Signal reversal exit (opposite RSI signal)
                    if not should_close:
                        if position.side == 'LONG' and signal == -1:
                            should_close = True
                            reason = 'SIGNAL_REVERSED'
                        elif position.side == 'SHORT' and signal == 1:
                            should_close = True
                            reason = 'SIGNAL_REVERSED'

                    if should_close and reason:
                        result = paper.close_position(symbol, reason=reason)
                        if result:
                            logger.info(f"AUTO: {reason} {symbol}, P&L: ${result.pnl:.2f}")
                            socketio.emit('auto_trade', {'action': 'CLOSE', 'symbol': symbol, 'pnl': result.pnl, 'reason': reason})

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


# ============== Trade Logger API ==============

def get_trade_logger():
    """Get trade logger instance"""
    try:
        from core.trade_logger import trade_logger
        return trade_logger
    except Exception as e:
        logger.error(f"Failed to import trade_logger: {e}")
        return None


@app.route('/api/trades/today')
def trades_today():
    """Get today's trades"""
    tl = get_trade_logger()
    if tl:
        trades = tl.get_today_trades()
        return jsonify([{
            'id': t.id,
            'symbol': t.symbol,
            'side': t.side,
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'entry_slippage': t.entry_slippage,
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'pnl_usd': t.pnl_usd,
            'pnl_pct': t.pnl_pct,
            'duration_hours': t.duration_hours
        } for t in trades])
    return jsonify([])


@app.route('/api/trades/recent')
def trades_recent():
    """Get recent trades"""
    limit = request.args.get('limit', 50, type=int)
    tl = get_trade_logger()
    if tl:
        trades = tl.get_recent_trades(limit=limit)
        return jsonify([{
            'id': t.id,
            'symbol': t.symbol,
            'side': t.side,
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'entry_slippage': t.entry_slippage,
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'pnl_usd': t.pnl_usd,
            'pnl_pct': t.pnl_pct,
            'duration_hours': t.duration_hours
        } for t in trades])
    return jsonify([])


@app.route('/api/trades/stats')
def trades_stats():
    """Get trading statistics"""
    days = request.args.get('days', 30, type=int)
    tl = get_trade_logger()
    if tl:
        return jsonify(tl.get_statistics(days=days))
    return jsonify({
        'total_trades': 0,
        'win_rate': 0,
        'profit_factor': 0,
        'total_pnl': 0,
        'avg_pnl': 0,
        'avg_duration': 0,
        'avg_slippage': 0
    })


# Cache for signals data
_signals_cache = {}
_signals_cache_time = 0
SIGNALS_CACHE_TTL = 3  # seconds - matches frontend update frequency


@app.route('/api/signals/all')
def signals_all():
    """Get signals and indicators for all symbols (multi-timeframe: 15m + 1h) with caching"""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    global _signals_cache, _signals_cache_time

    # Return cached data if fresh
    if time.time() - _signals_cache_time < SIGNALS_CACHE_TTL and _signals_cache:
        return jsonify(_signals_cache)

    symbols = SYMBOLS
    result = {}
    loader = get_binance_loader()

    def fetch_symbol_data(symbol):
        """Fetch and calculate signals for one symbol"""
        try:
            # Get both timeframes
            klines_1h = loader.client.get_klines(symbol=symbol, interval='1h', limit=120)
            klines_15m = loader.client.get_klines(symbol=symbol, interval='15m', limit=120)

            closes_1h = np.array([float(k[4]) for k in klines_1h])
            highs_1h = np.array([float(k[2]) for k in klines_1h])
            lows_1h = np.array([float(k[3]) for k in klines_1h])

            closes_15m = np.array([float(k[4]) for k in klines_15m])
            highs_15m = np.array([float(k[2]) for k in klines_15m])
            lows_15m = np.array([float(k[3]) for k in klines_15m])

            rsi_1h = calculate_rsi(closes_1h, 14)
            adx_1h = calculate_adx(highs_1h, lows_1h, closes_1h, 14)
            ema_100_1h = calculate_ema(closes_1h, 100)

            rsi_15m = calculate_rsi(closes_15m, 14)
            atr_15m = calculate_atr(highs_15m, lows_15m, closes_15m, 14)
            adx_15m = calculate_adx(highs_15m, lows_15m, closes_15m, 14)

            price = closes_15m[-1]
            trend_ema = ema_100_1h[-1]

            # Calculate signals
            signal = 0
            signal_15m = 0
            signal_1h = 1 if price > trend_ema else -1

            if adx_15m >= ADX_MIN:
                if rsi_15m < RSI_LONG:
                    signal_15m = 1
                elif rsi_15m > RSI_SHORT:
                    signal_15m = -1

            if signal_15m == 1 and signal_1h == 1:
                signal = 1
            elif signal_15m == -1 and signal_1h == -1:
                signal = -1

            return symbol, {
                'rsi': float(rsi_15m),
                'rsi_1h': float(rsi_1h),
                'adx': float(adx_15m),
                'adx_1h': float(adx_1h),
                'atr': float(atr_15m),
                'ema100': float(trend_ema),
                'price': float(price),
                'signal': signal,
                'signal_15m': signal_15m,
                'signal_1h': signal_1h
            }
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return symbol, {'rsi': 50, 'adx': 0, 'atr': 0, 'signal': 0}

    try:
        # Parallel fetch - 5 threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_symbol_data, sym): sym for sym in symbols}
            for future in as_completed(futures):
                symbol, data = future.result()
                result[symbol] = data

        # Update cache
        _signals_cache = result
        _signals_cache_time = time.time()

    except Exception as e:
        logger.error(f"Error getting signals: {e}")

    return jsonify(result)


@app.route('/api/chart/<symbol>')
def chart_data(symbol):
    """Get historical price data for chart"""
    interval = request.args.get('interval', '1h')
    limit = request.args.get('limit', 48, type=int)  # 48 hours default

    try:
        loader = get_binance_loader()
        klines = loader.client.get_klines(symbol=symbol, interval=interval, limit=limit)

        data = []
        for k in klines:
            data.append({
                'time': k[0],  # Open time in ms
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })

        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'data': data
        })
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades/comparison')
def trades_comparison():
    """Compare live results with backtest"""
    # Backtest results (from our optimization)
    backtest_stats = {
        'win_rate': 59.2,
        'profit_factor': 1.33,
        'avg_pnl': 0,
        'trades': 395
    }

    tl = get_trade_logger()
    if tl:
        return jsonify(tl.compare_with_backtest(backtest_stats))

    return jsonify({
        'backtest': backtest_stats,
        'live': {},
        'comparison': {}
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
