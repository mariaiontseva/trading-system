"""
Trading System Configuration
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BinanceConfig:
    """Binance API Configuration"""
    api_key: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    api_secret: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))
    testnet: bool = True  # ВАЖНО: начинаем с testnet!
    testnet_api_key: str = field(default_factory=lambda: os.getenv('BINANCE_TESTNET_API_KEY', ''))
    testnet_api_secret: str = field(default_factory=lambda: os.getenv('BINANCE_TESTNET_API_SECRET', ''))


@dataclass
class TradingConfig:
    """Trading Parameters"""
    # Монеты для торговли
    symbols: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
        'MATICUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'APTUSDT',
        'ARBUSDT', 'OPUSDT', 'INJUSDT', 'SUIUSDT', 'NEARUSDT'
    ])
    
    # Таймфреймы для анализа
    timeframes: List[str] = field(default_factory=lambda: [
        '1m', '5m', '15m', '1h', '4h', '1d'
    ])
    
    # Risk Management
    max_position_pct: float = 0.02  # 2% на сделку
    max_total_exposure_pct: float = 0.10  # 10% всего капитала
    max_daily_trades: int = 10
    max_daily_loss_pct: float = 0.05  # 5% дневной лимит потерь
    
    # Stop Loss / Take Profit
    default_stop_loss_pct: float = 0.02  # 2%
    default_take_profit_pct: float = 0.04  # 4% (1:2 risk:reward)
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.015  # 1.5%


@dataclass
class BacktestConfig:
    """Backtesting Parameters"""
    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    
    # Walk-forward
    train_days: int = 90
    test_days: int = 30
    
    # Minimum requirements
    min_trades: int = 100
    min_win_rate: float = 0.55
    min_sharpe: float = 1.5
    max_drawdown: float = 0.15


@dataclass
class DatabaseConfig:
    """Database Configuration"""
    # PostgreSQL (TimescaleDB)
    db_host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    db_port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))
    db_name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'trading_db'))
    db_user: str = field(default_factory=lambda: os.getenv('DB_USER', 'trading'))
    db_password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', ''))
    
    # Redis
    redis_host: str = field(default_factory=lambda: os.getenv('REDIS_HOST', 'localhost'))
    redis_port: int = field(default_factory=lambda: int(os.getenv('REDIS_PORT', '6379')))
    
    @property
    def db_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@dataclass
class DashboardConfig:
    """Web Dashboard Configuration"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = True
    secret_key: str = field(default_factory=lambda: os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-me'))


@dataclass
class Config:
    """Main Configuration"""
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Mode
    mode: str = 'paper'  # 'paper', 'live', 'backtest'
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'logs/trading.log'


# Global config instance
config = Config()
