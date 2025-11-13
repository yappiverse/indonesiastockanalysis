#!/usr/bin/env python3
"""
Summary Analyzer module for stock analysis system.
Provides clear buy/sell/hold recommendations with confidence levels and risk assessment.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple
from enum import Enum


class Recommendation(Enum):
    """Trading recommendation types."""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


class RiskLevel(Enum):
    """Risk assessment levels."""
    VERY_LOW = "VERY LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY HIGH"


class ConfidenceLevel(Enum):
    """Confidence assessment levels."""
    VERY_HIGH = "VERY HIGH"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY LOW"


class SummaryAnalyzer:
    """
    Analyzes stock analysis results and generates clear, actionable recommendations
    with confidence levels and risk assessment.
    """
    
    def __init__(self):
        self.buy_regimes = ["Bull", "Recovery"]
        self.sell_regimes = ["Bear", "Crash"]
        self.high_risk_regimes = ["HighVol", "Crash"]
        
    def generate_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive summary with recommendation, confidence, and risk assessment.
        
        Args:
            analysis_result: Dictionary from StockAnalyzer.analyze_ticker()
            
        Returns:
            Dictionary with summary information
        """
        if not analysis_result.get("success", False):
            return self._generate_error_summary(analysis_result)
        
        # Extract key metrics
        ticker = analysis_result["ticker"]
        regime = analysis_result["regime"]
        confidence = analysis_result["conf"]
        stats = analysis_result["stats"]
        
        # Generate recommendations
        recommendation, rec_confidence = self._generate_recommendation(regime, confidence, stats)
        risk_level = self._assess_risk(regime, stats)
        confidence_level = self._assess_confidence(confidence, stats)
        
        # Get supporting metrics
        key_metrics = self._extract_key_metrics(stats)
        rationale = self._generate_rationale(regime, recommendation, key_metrics)
        
        summary = {
            "ticker": ticker,
            "recommendation": recommendation.value,
            "confidence": confidence_level.value,
            "risk": risk_level.value,
            "rationale": rationale,
            "key_metrics": key_metrics,
            "regime": regime,
            "regime_confidence": confidence,
            "next_day_action": self._get_next_day_action(recommendation),
            "summary_score": self._calculate_summary_score(recommendation, confidence_level, risk_level)
        }
        
        return summary
    
    def _generate_recommendation(self, regime: str, confidence: float, stats: Dict) -> Tuple[Recommendation, float]:
        """Generate buy/sell/hold recommendation based on regime and performance."""
        
        # Base recommendation from regime
        if regime in self.buy_regimes:
            base_rec = Recommendation.BUY
            base_score = 0.7
        elif regime in self.sell_regimes:
            base_rec = Recommendation.SELL
            base_score = 0.7
        else:
            base_rec = Recommendation.HOLD
            base_score = 0.5
        
        # Adjust based on performance metrics
        performance_score = self._calculate_performance_score(stats)
        regime_confidence_score = min(confidence, 1.0)  # Normalize confidence
        
        # Combine scores
        final_score = (base_score * 0.4 + performance_score * 0.4 + regime_confidence_score * 0.2)
        
        # Map to recommendation
        if final_score >= 0.8:
            if base_rec == Recommendation.BUY:
                return Recommendation.STRONG_BUY, final_score
            elif base_rec == Recommendation.SELL:
                return Recommendation.STRONG_SELL, final_score
            else:
                return Recommendation.BUY, final_score
        elif final_score >= 0.6:
            if base_rec == Recommendation.HOLD:
                return Recommendation.BUY, final_score
            else:
                return base_rec, final_score
        elif final_score >= 0.4:
            return Recommendation.HOLD, final_score
        elif final_score >= 0.2:
            if base_rec == Recommendation.HOLD:
                return Recommendation.SELL, final_score
            else:
                return base_rec, final_score
        else:
            if base_rec == Recommendation.BUY:
                return Recommendation.SELL, final_score
            elif base_rec == Recommendation.SELL:
                return Recommendation.STRONG_SELL, final_score
            else:
                return Recommendation.SELL, final_score
    
    def _calculate_performance_score(self, stats: Dict) -> float:
        """Calculate performance score from backtest statistics."""
        score = 0.5  # Neutral starting point
        
        # Total Return impact
        total_return = stats.get('Total Return [%]', 0)
        if total_return > 10:
            score += 0.2
        elif total_return > 5:
            score += 0.1
        elif total_return < -5:
            score -= 0.1
        elif total_return < -10:
            score -= 0.2
        
        # Win Rate impact
        win_rate = stats.get('Win Rate [%]', 50)
        if win_rate > 60:
            score += 0.15
        elif win_rate > 55:
            score += 0.1
        elif win_rate < 40:
            score -= 0.1
        elif win_rate < 30:
            score -= 0.15
        
        # Sharpe Ratio impact
        sharpe = stats.get('Sharpe Ratio', 1.0)
        if sharpe > 1.5:
            score += 0.15
        elif sharpe > 1.0:
            score += 0.1
        elif sharpe < 0.5:
            score -= 0.1
        elif sharpe < 0:
            score -= 0.15
        
        # Max Drawdown impact
        max_dd = abs(stats.get('Max Drawdown [%]', 0))
        if max_dd < 5:
            score += 0.1
        elif max_dd < 10:
            score += 0.05
        elif max_dd > 20:
            score -= 0.1
        elif max_dd > 15:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def _assess_risk(self, regime: str, stats: Dict) -> RiskLevel:
        """Assess overall risk level."""
        risk_score = 0.5  # Moderate starting point
        
        # Regime-based risk
        if regime in self.high_risk_regimes:
            risk_score += 0.3
        elif regime in ["Bull", "Recovery"]:
            risk_score -= 0.2
        elif regime in ["Bear"]:
            risk_score += 0.2
        
        # Performance-based risk
        max_dd = abs(stats.get('Max Drawdown [%]', 0))
        if max_dd > 15:
            risk_score += 0.3
        elif max_dd > 10:
            risk_score += 0.2
        elif max_dd < 5:
            risk_score -= 0.2
        
        volatility = stats.get('Sharpe Ratio', 1.0)
        if volatility < 0.5:
            risk_score += 0.1
        elif volatility > 2.0:
            risk_score -= 0.1
        
        # Map to risk levels
        if risk_score >= 0.8:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MODERATE
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _assess_confidence(self, regime_confidence: float, stats: Dict) -> ConfidenceLevel:
        """Assess overall confidence level."""
        confidence_score = regime_confidence
        
        # Adjust based on data quality and performance consistency
        total_trades = stats.get('Total Trades', 0)
        if total_trades >= 20:
            confidence_score += 0.1
        elif total_trades >= 10:
            confidence_score += 0.05
        elif total_trades < 5:
            confidence_score -= 0.1
        
        win_rate = stats.get('Win Rate [%]', 50)
        if 40 <= win_rate <= 70:  # Reasonable win rate range
            confidence_score += 0.05
        
        # Map to confidence levels
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MODERATE
        elif confidence_score >= 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _extract_key_metrics(self, stats: Dict) -> Dict[str, Any]:
        """Extract and format key supporting metrics."""
        return {
            "total_return": stats.get('Total Return [%]', 'N/A'),
            "sharpe_ratio": stats.get('Sharpe Ratio', 'N/A'),
            "max_drawdown": stats.get('Max Drawdown [%]', 'N/A'),
            "win_rate": stats.get('Win Rate [%]', 'N/A'),
            "total_trades": stats.get('Total Trades', 'N/A'),
            "profit_factor": stats.get('Profit Factor', 'N/A'),
            "calmar_ratio": stats.get('Calmar Ratio', 'N/A')
        }
    
    def _generate_rationale(self, regime: str, recommendation: Recommendation, metrics: Dict) -> str:
        """Generate rationale text explaining the recommendation."""
        rationale_parts = []
        
        # Regime-based rationale
        if regime in self.buy_regimes:
            rationale_parts.append(f"Current regime ({regime}) is favorable for buying.")
        elif regime in self.sell_regimes:
            rationale_parts.append(f"Current regime ({regime}) suggests caution.")
        else:
            rationale_parts.append(f"Current regime ({regime}) indicates sideways movement.")
        
        # Performance-based rationale
        total_return = metrics.get("total_return", 0)
        if total_return != 'N/A':
            if total_return > 5:
                rationale_parts.append(f"Strong historical performance ({total_return:.1f}% return).")
            elif total_return < -5:
                rationale_parts.append(f"Weak historical performance ({total_return:.1f}% return).")
        
        win_rate = metrics.get("win_rate", 50)
        if win_rate != 'N/A':
            if win_rate > 60:
                rationale_parts.append(f"High win rate ({win_rate:.1f}%) supports strategy effectiveness.")
            elif win_rate < 40:
                rationale_parts.append(f"Low win rate ({win_rate:.1f}%) suggests strategy inconsistency.")
        
        max_dd = metrics.get("max_drawdown", 0)
        if max_dd != 'N/A':
            if abs(max_dd) < 8:
                rationale_parts.append(f"Controlled risk profile ({abs(max_dd):.1f}% max drawdown).")
            elif abs(max_dd) > 15:
                rationale_parts.append(f"Elevated risk ({abs(max_dd):.1f}% max drawdown).")
        
        return " ".join(rationale_parts)
    
    def _get_next_day_action(self, recommendation: Recommendation) -> str:
        """Get clear next-day action based on recommendation."""
        if recommendation == Recommendation.STRONG_BUY:
            return "BUY on next trading day"
        elif recommendation == Recommendation.BUY:
            return "Consider buying on next trading day"
        elif recommendation == Recommendation.HOLD:
            return "Hold current position, monitor for changes"
        elif recommendation == Recommendation.SELL:
            return "Consider selling on next trading day"
        elif recommendation == Recommendation.STRONG_SELL:
            return "SELL on next trading day"
        else:
            return "No clear action - monitor market conditions"
    
    def _calculate_summary_score(self, recommendation: Recommendation, 
                               confidence: ConfidenceLevel, risk: RiskLevel) -> int:
        """Calculate overall summary score (1-5 stars)."""
        score = 3  # Neutral starting point
        
        # Recommendation impact
        if recommendation in [Recommendation.STRONG_BUY, Recommendation.STRONG_SELL]:
            score += 1
        elif recommendation in [Recommendation.BUY, Recommendation.SELL]:
            score += 0.5
        
        # Confidence impact
        if confidence == ConfidenceLevel.VERY_HIGH:
            score += 1
        elif confidence == ConfidenceLevel.HIGH:
            score += 0.5
        elif confidence == ConfidenceLevel.VERY_LOW:
            score -= 1
        
        # Risk impact (lower risk is better)
        if risk == RiskLevel.VERY_LOW:
            score += 1
        elif risk == RiskLevel.LOW:
            score += 0.5
        elif risk == RiskLevel.VERY_HIGH:
            score -= 1
        
        return max(1, min(5, int(round(score))))
    
    def _generate_error_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for failed analysis."""
        return {
            "ticker": analysis_result.get("ticker", "Unknown"),
            "recommendation": "UNAVAILABLE",
            "confidence": "LOW",
            "risk": "UNKNOWN",
            "rationale": f"Analysis failed: {analysis_result.get('error', 'Unknown error')}",
            "key_metrics": {},
            "regime": "UNKNOWN",
            "regime_confidence": 0.0,
            "next_day_action": "Cannot provide recommendation - analysis failed",
            "summary_score": 1
        }
    
    def format_summary_report(self, summary: Dict[str, Any]) -> str:
        """
        Format summary as a clean, readable report.
        
        Args:
            summary: Summary dictionary from generate_summary()
            
        Returns:
            Formatted report string
        """
        if summary["recommendation"] == "UNAVAILABLE":
            return self._format_error_report(summary)
        
        # Recommendation emoji mapping
        rec_emoji = {
            "STRONG BUY": "🚀",
            "BUY": "📈", 
            "HOLD": "⚖️",
            "SELL": "📉",
            "STRONG SELL": "💥"
        }
        
        # Risk emoji mapping
        risk_emoji = {
            "VERY LOW": "🟢",
            "LOW": "🟡",
            "MODERATE": "🟠", 
            "HIGH": "🔴",
            "VERY HIGH": "💀"
        }
        
        emoji = rec_emoji.get(summary["recommendation"], "📊")
        risk_icon = risk_emoji.get(summary["risk"], "⚫")
        
        report = f"""
{emoji} STOCK SUMMARY: {summary['ticker']} {emoji}

🎯 RECOMMENDATION: {summary['recommendation']}
📅 Next Day Action: {summary['next_day_action']}

📊 CONFIDENCE LEVEL: {summary['confidence']}
{risk_icon} RISK LEVEL: {summary['risk']}
⭐ OVERALL SCORE: {'★' * summary['summary_score']}{'☆' * (5 - summary['summary_score'])}

📈 KEY METRICS:
  • Total Return: {summary['key_metrics'].get('total_return', 'N/A'):.2f}%
  • Sharpe Ratio: {summary['key_metrics'].get('sharpe_ratio', 'N/A'):.2f}
  • Max Drawdown: {summary['key_metrics'].get('max_drawdown', 'N/A'):.2f}%
  • Win Rate: {summary['key_metrics'].get('win_rate', 'N/A'):.1f}%
  • Total Trades: {summary['key_metrics'].get('total_trades', 'N/A')}

🎭 MARKET REGIME:
  • Current: {summary['regime']}
  • Confidence: {summary['regime_confidence']:.2f}

💡 RATIONALE:
  {summary['rationale']}
"""
        return report
    
    def _format_error_report(self, summary: Dict[str, Any]) -> str:
        """Format error report."""
        return f"""
❌ ANALYSIS FAILED: {summary['ticker']}

Error: {summary['rationale']}

Next Day Action: {summary['next_day_action']}
"""