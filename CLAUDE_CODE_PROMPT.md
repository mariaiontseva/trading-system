# PROMPT FOR CLAUDE CODE

## Project Context

I have a cryptocurrency trading bot system that needs to be completed. The project was started in Claude.ai and I need you to continue development.

## What's Already Done

A trading system with the following structure:

```
trading-system/
├── config/
│   └── settings.py          # Configuration (Binance, trading params, risk management)
├── core/
│   ├── binance_client.py    # Binance API client (data loading, trading, WebSocket)
│   └── paper_trading.py     # Paper trading engine with SQLite persistence
├── data/
│   └── data_manager.py      # Historical data manager (download, store in Parquet)
├── backtesting/
│   └── engine.py            # Backtesting engine with walk-forward validation
├── bots/
│   └── strategies.py        # Trading strategies (Mean Reversion, Trend Following)
├── dashboard/
│   ├── app.py               # Flask + SocketIO web dashboard
│   └── templates/
│       └── index.html       # Dashboard UI (Tailwind CSS, Chart.js)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Key Features Implemented

1. **Binance Client** - loads historical OHLCV data, WebSocket for real-time prices
2. **Data Manager** - downloads and stores data in Parquet format, SQLite registry
3. **Backtest Engine** - event-driven, calculates Sharpe, Sortino, max drawdown, win rate
4. **Paper Trading** - virtual trading with persistent state, stop-loss/take-profit
5. **Web Dashboard** - real-time prices, positions, trades, backtest runner

## What Needs To Be Done

1. **Test and fix any bugs** - run the system, fix import errors, test all endpoints

2. **Add missing __init__.py files** if needed

3. **Complete the data download flow**:
   - Test downloading historical data from Binance
   - Verify Parquet storage works correctly

4. **Enhance backtesting**:
   - Add Monte Carlo simulation
   - Add walk-forward validation UI
   - Save backtest results to database

5. **Add ML-based strategy** (optional):
   - Feature engineering (200+ features)
   - XGBoost/LightGBM ensemble
   - Bayesian signal combination

6. **Live trading preparation**:
   - Binance testnet integration
   - Order execution with retry logic
   - Position management

7. **Docker setup**:
   - Verify docker-compose works
   - Add Redis for message queue
   - Add TimescaleDB for time-series data

## How To Start

```bash
# Navigate to project
cd trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python -m dashboard.app

# Or run data download
python -m data.data_manager --start 2023-01-01
```

## API Endpoints

- `GET /api/status` - system status
- `GET /api/paper/status` - paper trading status
- `POST /api/paper/open` - open position
- `POST /api/paper/close` - close position
- `POST /api/backtest/run` - run backtest
- `POST /api/data/download` - start data download
- `GET /api/prices` - current prices

## Tech Stack

- Python 3.11+
- Flask + Flask-SocketIO
- python-binance
- pandas, numpy, pandas-ta
- SQLite (paper trading state)
- Parquet (historical data)
- Chart.js (frontend charts)
- Tailwind CSS (styling)

## Priority Tasks

1. First, verify the project runs without errors
2. Test the data download from Binance
3. Test backtesting with sample data
4. Test paper trading flow
5. Then add new features

Please start by examining the code structure and running the dashboard to identify any issues.
