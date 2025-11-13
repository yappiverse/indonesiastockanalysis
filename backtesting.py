#!/usr/bin/env python3
"""
Backtesting module for stock analysis.
Provides portfolio backtesting with regime-based strategies.
"""

import vectorbt as vbt
import pandas as pd
from typing import List, Dict, Any, Optional
from config import BACKTEST_CONFIG


class Backtester:
    """Handles backtesting of regime-based trading strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or BACKTEST_CONFIG["default"]
        
    def backtest_regime_strategy(
        self,
        df_intraday: pd.DataFrame,
        regime_column: str = "regime_ml",
        trade_regimes: Optional[List[str]] = None,
        **kwargs
    ) -> vbt.Portfolio:
        """
        Backtest a regime-based trading strategy.
        
        Args:
            df_intraday: Intraday dataframe with regime data
            regime_column: Name of the regime column
            trade_regimes: List of regimes to trade in
            **kwargs: Override backtest parameters
            
        Returns:
            VectorBT Portfolio object with backtest results
        """
        # Merge config with overrides
        backtest_params = self._get_backtest_params(kwargs)
        trade_regimes = trade_regimes or backtest_params["trade_regimes"]
        
        close = df_intraday["Close"]
        regime = df_intraday[regime_column]
        
        # Generate trading signals
        entries, exits = self._generate_regime_signals(regime, trade_regimes)
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            freq=backtest_params.get("frequency"),
            **{k: v for k, v in backtest_params.items() if k not in ["trade_regimes", "frequency"]}
        )
        
        return portfolio
    
    def _get_backtest_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Get backtest parameters with overrides applied."""
        params = self.config.copy()
        params.update(overrides)
        return params
    
    def _generate_regime_signals(
        self, 
        regime: pd.Series, 
        trade_regimes: List[str]
    ) -> tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on regime changes.
        
        Args:
            regime: Series with regime labels
            trade_regimes: List of regimes to trade in
            
        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        # Identify tradable periods
        tradable = regime.isin(trade_regimes)
        tradable_prev = tradable.shift(1).fillna(False)
        
        # Ensure boolean type without downcasting warnings
        tradable_prev = tradable_prev.astype(bool)
        
        # Generate signals
        entries = (tradable & ~tradable_prev)
        exits = (~tradable & tradable_prev)
        
        return entries, exits
    
    def analyze_portfolio_performance(
        self, 
        portfolio: vbt.Portfolio,
        include_detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze portfolio performance with comprehensive metrics.
        
        Args:
            portfolio: VectorBT Portfolio object
            include_detailed: Whether to include detailed metrics
            
        Returns:
            Dictionary with performance metrics
        """
        stats = portfolio.stats()
        
        # Basic performance metrics
        performance = {
            "total_return": stats["Total Return [%]"],
            "sharpe_ratio": stats["Sharpe Ratio"],
            "max_drawdown": stats["Max Drawdown [%]"],
            "win_rate": stats["Win Rate [%]"],
            "total_trades": stats["Total Trades"],
            "total_closed_trades": stats["Total Closed Trades"],
            "total_open_trades": stats["Total Open Trades"]
        }
        
        if include_detailed:
            # Add detailed metrics
            performance.update({
                "avg_trade_return": stats["Avg. Trade [%]"],
                "avg_winning_trade": stats["Avg. Winning Trade [%]"],
                "avg_losing_trade": stats["Avg. Losing Trade [%]"],
                "profit_factor": stats["Profit Factor"],
                "expectancy": stats["Expectancy"],
                "calmar_ratio": stats["Calmar Ratio"]
            })
        
        return performance
    
    def compare_strategies(
        self,
        df_intraday: pd.DataFrame,
        regime_columns: List[str],
        strategy_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple regime strategies.
        
        Args:
            df_intraday: Intraday dataframe with regime data
            regime_columns: List of regime column names to test
            strategy_names: Optional names for each strategy
            
        Returns:
            DataFrame with comparison results
        """
        if strategy_names is None:
            strategy_names = [f"Strategy_{i+1}" for i in range(len(regime_columns))]
        
        if len(regime_columns) != len(strategy_names):
            raise ValueError("regime_columns and strategy_names must have same length")
        
        results = []
        
        for regime_col, strategy_name in zip(regime_columns, strategy_names):
            try:
                portfolio = self.backtest_regime_strategy(df_intraday, regime_column=regime_col)
                performance = self.analyze_portfolio_performance(portfolio)
                performance["strategy"] = strategy_name
                performance["regime_column"] = regime_col
                results.append(performance)
            except Exception as e:
                print(f"Error testing strategy {strategy_name}: {e}")
        
        return pd.DataFrame(results)


class PerformanceReporter:
    """Handles reporting and visualization of backtest results."""
    
    def __init__(self):
        pass
    
    def generate_summary_report(
        self, 
        portfolio: vbt.Portfolio, 
        ticker: str,
        regime: str,
        confidence: float
    ) -> str:
        """
        Generate a summary report for backtest results.
        
        Args:
            portfolio: VectorBT Portfolio object
            ticker: Stock ticker
            regime: Current regime
            confidence: Regime confidence score
            
        Returns:
            Formatted summary report string
        """
        stats = portfolio.stats()
        
        report = f"""
===== BACKTEST RESULTS for {ticker} =====

📊 Performance Summary:
  Total Return: {stats['Total Return [%]']:.2f}%
  Sharpe Ratio: {stats['Sharpe Ratio']:.2f}
  Max Drawdown: {stats['Max Drawdown [%]']:.2f}%
  Win Rate: {stats['Win Rate [%]']:.2f}%

📈 Trading Activity:
  Total Trades: {stats['Total Trades']}
  Closed Trades: {stats['Total Closed Trades']}
  Open Trades: {stats['Total Open Trades']}

🎯 Current Regime:
  Regime: {regime}
  Confidence: {confidence:.2f}
"""
        
        if regime in ["Bull", "Recovery"]:
            report += "\n🔥 BUY SIGNAL - Favorable regime detected"
        else:
            report += "\n❌ No long setup - Wait for better conditions"
        
        return report
    
    def format_portfolio_stats(self, portfolio: vbt.Portfolio) -> str:
        """
        Format portfolio statistics for display.
        
        Args:
            portfolio: VectorBT Portfolio object
            
        Returns:
            Formatted statistics string
        """
        stats = portfolio.stats()
        
        # Select key metrics for display
        key_metrics = [
            "Start", "End", "Period", "Start Value", "End Value",
            "Total Return [%]", "Benchmark Return [%]", "Max Drawdown [%]",
            "Sharpe Ratio", "Calmar Ratio", "Win Rate [%]", "Profit Factor"
        ]
        
        formatted = "Portfolio Statistics:\n"
        for metric in key_metrics:
            if metric in stats:
                formatted += f"  {metric}: {stats[metric]}\n"
        
        return formatted