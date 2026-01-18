"""
Backtesting Engine - полный движок для тестирования стратегий
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
from datetime import datetime
import json
from loguru import logger


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class Trade:
    """Информация о сделке"""
    id: int
    symbol: str
    side: Side
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    size_usd: float
    
    # P&L
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Costs
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage: float = 0.0
    
    # Exit reason
    exit_reason: str = ""
    
    # Stop/Take
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Status
    is_open: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'size_usd': self.size_usd,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'fees': self.entry_fee + self.exit_fee,
            'exit_reason': self.exit_reason,
            'is_open': self.is_open
        }


@dataclass
class BacktestConfig:
    """Конфигурация бэктеста"""
    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    
    # Position sizing
    max_position_pct: float = 0.02  # 2% per trade
    max_positions: int = 5
    
    # Risk management
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.015
    
    # Circuit breakers
    max_daily_loss_pct: float = 0.05
    max_consecutive_losses: int = 5


@dataclass
class BacktestResult:
    """Результаты бэктеста"""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Ratios
    profit_factor: float = 0.0
    risk_reward: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    
    # Costs
    total_fees: float = 0.0
    total_slippage: float = 0.0
    
    # Time
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0
    
    # Equity
    final_equity: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    
    # Trades
    trades: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 4),
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 4),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'risk_reward': round(self.risk_reward, 2),
            'expectancy': round(self.expectancy, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 4),
            'total_fees': round(self.total_fees, 2),
            'final_equity': round(self.final_equity, 2),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'trading_days': self.trading_days
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BacktestEngine:
    """
    Event-driven backtesting engine
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        """Сброс состояния"""
        self.capital = self.config.initial_capital
        self.equity = self.config.initial_capital
        self.equity_curve = [self.equity]
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.trade_counter = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.is_halted = False
    
    def run(
        self,
        strategy,
        data: pd.DataFrame,
        symbol: str = 'BTCUSDT'
    ) -> BacktestResult:
        """
        Запуск бэктеста
        
        Args:
            strategy: Объект стратегии с методом generate_signal(data) -> Dict
            data: OHLCV DataFrame с индексом datetime
            symbol: Торговая пара
        
        Returns:
            BacktestResult с метриками
        """
        self.reset()
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        
        # Минимум данных для начала
        warmup_period = getattr(strategy, 'warmup_period', 100)
        
        for i in range(warmup_period, len(data)):
            if self.is_halted:
                logger.warning("Trading halted due to circuit breaker")
                break
            
            # Текущий и исторические бары
            current_bar = data.iloc[i]
            historical_data = data.iloc[:i+1]
            
            # Обновление открытых позиций (stop-loss, take-profit)
            self._update_positions(current_bar, symbol)
            
            # Генерация сигнала
            signal = strategy.generate_signal(historical_data)
            
            # Обработка сигнала
            if signal and signal.get('action') != 'HOLD':
                self._process_signal(signal, current_bar, symbol)
            
            # Обновление equity curve
            unrealized_pnl = self._calculate_unrealized_pnl(current_bar)
            self.equity = self.capital + unrealized_pnl
            self.equity_curve.append(self.equity)
            
            # Проверка circuit breakers
            self._check_circuit_breakers()
        
        # Закрытие открытых позиций
        if self.open_positions:
            last_bar = data.iloc[-1]
            for symbol in list(self.open_positions.keys()):
                self._close_position(symbol, last_bar, "END_OF_BACKTEST")
        
        # Расчёт результатов
        result = self._calculate_results(data)
        
        logger.info(f"Backtest complete. Trades: {result.total_trades}, "
                   f"Win rate: {result.win_rate:.1%}, "
                   f"Return: {result.total_return_pct:.2%}")
        
        return result
    
    def _process_signal(self, signal: Dict, bar: pd.Series, symbol: str):
        """
        Обработка торгового сигнала
        """
        action = signal.get('action')
        
        # Закрытие противоположной позиции
        if symbol in self.open_positions:
            position = self.open_positions[symbol]
            if (action == 'BUY' and position.side == Side.SHORT) or \
               (action == 'SELL' and position.side == Side.LONG):
                self._close_position(symbol, bar, "SIGNAL_REVERSE")
        
        # Открытие новой позиции
        if action in ['BUY', 'SELL'] and symbol not in self.open_positions:
            # Проверка лимита позиций
            if len(self.open_positions) >= self.config.max_positions:
                return
            
            self._open_position(signal, bar, symbol)
    
    def _open_position(self, signal: Dict, bar: pd.Series, symbol: str):
        """
        Открытие позиции
        """
        side = Side.LONG if signal['action'] == 'BUY' else Side.SHORT
        
        # Размер позиции
        position_pct = signal.get('position_size', self.config.max_position_pct)
        size_usd = self.capital * position_pct
        
        # Slippage
        slippage = bar['close'] * self.config.slippage_pct
        if side == Side.LONG:
            entry_price = bar['close'] + slippage
        else:
            entry_price = bar['close'] - slippage
        
        # Fee
        entry_fee = size_usd * self.config.fee_rate
        
        # Quantity
        quantity = (size_usd - entry_fee) / entry_price
        
        # Stop-loss и Take-profit
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        if not stop_loss:
            if side == Side.LONG:
                stop_loss = entry_price * (1 - self.config.default_stop_loss_pct)
            else:
                stop_loss = entry_price * (1 + self.config.default_stop_loss_pct)
        
        if not take_profit:
            if side == Side.LONG:
                take_profit = entry_price * (1 + self.config.default_take_profit_pct)
            else:
                take_profit = entry_price * (1 - self.config.default_take_profit_pct)
        
        # Создание trade
        self.trade_counter += 1
        trade = Trade(
            id=self.trade_counter,
            symbol=symbol,
            side=side,
            entry_time=bar.name,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            size_usd=size_usd,
            entry_fee=entry_fee,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.open_positions[symbol] = trade
        logger.debug(f"Opened {side.value} position: {symbol} @ {entry_price:.4f}")
    
    def _close_position(self, symbol: str, bar: pd.Series, reason: str, exit_price: float = None):
        """
        Закрытие позиции
        """
        if symbol not in self.open_positions:
            return
        
        trade = self.open_positions[symbol]
        
        if exit_price is None:
            exit_price = bar['close']
        
        # Slippage при выходе
        slippage = exit_price * self.config.slippage_pct
        if trade.side == Side.LONG:
            exit_price = exit_price - slippage
        else:
            exit_price = exit_price + slippage
        
        # Exit fee
        exit_value = trade.quantity * exit_price
        exit_fee = exit_value * self.config.fee_rate
        
        # P&L
        if trade.side == Side.LONG:
            gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.quantity
        
        net_pnl = gross_pnl - trade.entry_fee - exit_fee
        pnl_pct = net_pnl / trade.size_usd
        
        # Обновление trade
        trade.exit_time = bar.name
        trade.exit_price = exit_price
        trade.exit_fee = exit_fee
        trade.pnl = net_pnl
        trade.pnl_pct = pnl_pct
        trade.exit_reason = reason
        trade.is_open = False
        trade.slippage = slippage * 2 * trade.quantity
        
        # Обновление капитала
        self.capital += net_pnl
        self.daily_pnl += net_pnl
        
        # Обновление consecutive losses
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Сохранение и удаление из открытых
        self.trades.append(trade)
        del self.open_positions[symbol]
        
        logger.debug(f"Closed {trade.side.value} position: {symbol} @ {exit_price:.4f}, "
                    f"PnL: ${net_pnl:.2f} ({pnl_pct:.2%}), Reason: {reason}")
    
    def _update_positions(self, bar: pd.Series, symbol: str):
        """
        Обновление позиций (проверка stop-loss, take-profit)
        """
        if symbol not in self.open_positions:
            return
        
        trade = self.open_positions[symbol]
        
        if trade.side == Side.LONG:
            # Stop-loss hit
            if trade.stop_loss and bar['low'] <= trade.stop_loss:
                self._close_position(symbol, bar, "STOP_LOSS", trade.stop_loss)
                return
            # Take-profit hit
            if trade.take_profit and bar['high'] >= trade.take_profit:
                self._close_position(symbol, bar, "TAKE_PROFIT", trade.take_profit)
                return
            # Trailing stop update
            if self.config.use_trailing_stop:
                new_stop = bar['high'] * (1 - self.config.trailing_stop_pct)
                if new_stop > trade.stop_loss:
                    trade.stop_loss = new_stop
        
        else:  # SHORT
            # Stop-loss hit
            if trade.stop_loss and bar['high'] >= trade.stop_loss:
                self._close_position(symbol, bar, "STOP_LOSS", trade.stop_loss)
                return
            # Take-profit hit
            if trade.take_profit and bar['low'] <= trade.take_profit:
                self._close_position(symbol, bar, "TAKE_PROFIT", trade.take_profit)
                return
            # Trailing stop update
            if self.config.use_trailing_stop:
                new_stop = bar['low'] * (1 + self.config.trailing_stop_pct)
                if new_stop < trade.stop_loss:
                    trade.stop_loss = new_stop
    
    def _calculate_unrealized_pnl(self, bar: pd.Series) -> float:
        """
        Расчёт нереализованного P&L
        """
        unrealized = 0.0
        for symbol, trade in self.open_positions.items():
            if trade.side == Side.LONG:
                unrealized += (bar['close'] - trade.entry_price) * trade.quantity
            else:
                unrealized += (trade.entry_price - bar['close']) * trade.quantity
        return unrealized
    
    def _check_circuit_breakers(self):
        """
        Проверка circuit breakers
        """
        # Max daily loss
        if self.daily_pnl < -self.config.initial_capital * self.config.max_daily_loss_pct:
            self.is_halted = True
            logger.warning(f"Circuit breaker: Max daily loss reached (${self.daily_pnl:.2f})")
        
        # Max consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.is_halted = True
            logger.warning(f"Circuit breaker: {self.consecutive_losses} consecutive losses")
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """
        Расчёт всех метрик
        """
        result = BacktestResult()
        
        if not self.trades:
            result.equity_curve = self.equity_curve
            result.final_equity = self.equity
            return result
        
        # Trade statistics
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        result.total_trades = len(self.trades)
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        # P&L
        result.total_pnl = sum(t.pnl for t in self.trades)
        result.total_return_pct = result.total_pnl / self.config.initial_capital
        result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        result.avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
        result.largest_win = max(t.pnl for t in self.trades)
        result.largest_loss = min(t.pnl for t in self.trades)
        
        # Ratios
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        result.risk_reward = result.avg_win / result.avg_loss if result.avg_loss > 0 else float('inf')
        result.expectancy = (result.win_rate * result.avg_win) - ((1 - result.win_rate) * result.avg_loss)
        
        # Equity curve metrics
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Sharpe (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            result.sharpe_ratio = np.sqrt(365) * np.mean(returns) / np.std(returns)
        
        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0:
            result.sortino_ratio = np.sqrt(365) * np.mean(returns) / np.std(downside)
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        result.max_drawdown_pct = np.max(drawdown)
        result.drawdown_curve = drawdown.tolist()
        
        # Calmar
        if result.max_drawdown_pct > 0:
            result.calmar_ratio = result.total_return_pct / result.max_drawdown_pct
        
        # Costs
        result.total_fees = sum(t.entry_fee + t.exit_fee for t in self.trades)
        result.total_slippage = sum(t.slippage for t in self.trades)
        
        # Time
        result.start_date = data.index[0].isoformat()
        result.end_date = data.index[-1].isoformat()
        result.trading_days = (data.index[-1] - data.index[0]).days
        
        # Final
        result.final_equity = self.equity
        result.equity_curve = self.equity_curve
        result.trades = [t.to_dict() for t in self.trades]
        
        return result


class WalkForwardValidator:
    """
    Walk-Forward Analysis
    """
    
    def __init__(
        self,
        train_period: int = 90,
        test_period: int = 30,
        n_splits: Optional[int] = None
    ):
        self.train_period = train_period
        self.test_period = test_period
        self.n_splits = n_splits
    
    def validate(
        self,
        strategy_class,
        data: pd.DataFrame,
        symbol: str = 'BTCUSDT',
        config: BacktestConfig = None
    ) -> Dict:
        """
        Walk-forward validation
        """
        results = []
        window_size = self.train_period + self.test_period
        
        if self.n_splits:
            step = max(1, (len(data) - window_size) // self.n_splits)
        else:
            step = self.test_period
        
        i = 0
        split_num = 0
        
        while i + window_size <= len(data):
            split_num += 1
            
            # Windows
            train_data = data.iloc[i:i + self.train_period]
            test_data = data.iloc[i + self.train_period:i + self.train_period + self.test_period]
            
            logger.info(f"Split {split_num}: Train {train_data.index[0].date()} to {train_data.index[-1].date()}, "
                       f"Test {test_data.index[0].date()} to {test_data.index[-1].date()}")
            
            # Train strategy
            strategy = strategy_class()
            if hasattr(strategy, 'train'):
                strategy.train(train_data)
            
            # Test
            engine = BacktestEngine(config or BacktestConfig())
            result = engine.run(strategy, test_data, symbol)
            
            results.append({
                'split': split_num,
                'train_start': train_data.index[0].isoformat(),
                'train_end': train_data.index[-1].isoformat(),
                'test_start': test_data.index[0].isoformat(),
                'test_end': test_data.index[-1].isoformat(),
                'win_rate': result.win_rate,
                'total_return': result.total_return_pct,
                'sharpe': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown_pct,
                'trades': result.total_trades
            })
            
            i += step
        
        # Aggregate
        return {
            'splits': results,
            'avg_win_rate': np.mean([r['win_rate'] for r in results]),
            'std_win_rate': np.std([r['win_rate'] for r in results]),
            'min_win_rate': np.min([r['win_rate'] for r in results]),
            'max_win_rate': np.max([r['win_rate'] for r in results]),
            'avg_return': np.mean([r['total_return'] for r in results]),
            'avg_sharpe': np.mean([r['sharpe'] for r in results]),
            'avg_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'consistency': sum(1 for r in results if r['win_rate'] > 0.5) / len(results) if results else 0
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation для оценки рисков
    """
    
    def simulate(
        self,
        trades: List[Trade],
        n_simulations: int = 1000,
        initial_capital: float = 10000
    ) -> Dict:
        """
        Случайные перестановки сделок
        """
        if not trades:
            return {}
        
        pnls = [t.pnl for t in trades]
        
        final_equities = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            shuffled = np.random.permutation(pnls)
            
            equity = [initial_capital]
            for pnl in shuffled:
                equity.append(equity[-1] + pnl)
            equity = np.array(equity)
            
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            
            final_equities.append(equity[-1])
            max_drawdowns.append(np.max(dd))
        
        return {
            'median_final_equity': np.median(final_equities),
            'mean_final_equity': np.mean(final_equities),
            'std_final_equity': np.std(final_equities),
            'percentile_5': np.percentile(final_equities, 5),
            'percentile_25': np.percentile(final_equities, 25),
            'percentile_75': np.percentile(final_equities, 75),
            'percentile_95': np.percentile(final_equities, 95),
            'prob_profit': sum(1 for e in final_equities if e > initial_capital) / n_simulations,
            'prob_double': sum(1 for e in final_equities if e > initial_capital * 2) / n_simulations,
            'median_max_dd': np.median(max_drawdowns),
            'worst_case_dd': np.percentile(max_drawdowns, 95),
            'risk_of_ruin_50': sum(1 for e in final_equities if e < initial_capital * 0.5) / n_simulations,
            'risk_of_ruin_25': sum(1 for e in final_equities if e < initial_capital * 0.25) / n_simulations
        }
