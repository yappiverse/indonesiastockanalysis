# Indonesian Stock Analysis Project

A sophisticated Python-based system for automated stock analysis, regime detection, and trading recommendations for Indonesian stocks. This system combines technical analysis, machine learning, and backtesting to generate clear buy/sell/hold recommendations with confidence scoring and risk assessment.

## 🚀 Features

### Core Capabilities

- **Automated Stock Analysis** - Comprehensive analysis pipeline for Indonesian stocks
- **Regime Detection** - Rule-based and ML-based market regime classification
- **Backtesting Engine** - Portfolio backtesting using VectorBT
- **Trading Recommendations** - Clear buy/sell/hold signals with confidence levels
- **Risk Assessment** - Comprehensive risk evaluation with visual indicators
- **Parallel Processing** - Multi-core analysis for multiple stocks
- **CSV Export** - Automatic timestamped export of analysis results

### Recommendation System

- **STRONG BUY** 🚀 - High conviction buy signal
- **BUY** 📈 - Favorable conditions for buying
- **HOLD** ⚖️ - Maintain current position
- **SELL** 📉 - Consider selling
- **STRONG SELL** 💥 - High conviction sell signal

### Confidence Levels

- **VERY HIGH** - Highly reliable analysis
- **HIGH** - Reliable analysis
- **MODERATE** - Reasonable confidence
- **LOW** - Limited confidence
- **VERY LOW** - Low reliability

### Risk Assessment

- **VERY LOW** 🟢 - Minimal risk
- **LOW** 🟡 - Low risk
- **MODERATE** 🟠 - Moderate risk
- **HIGH** 🔴 - High risk
- **VERY HIGH** 💀 - Very high risk

## 📋 System Architecture

### Core Components

| Component                    | File                                                 | Description                                                 |
| ---------------------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| **Main Orchestrator**        | [`main.py`](main.py:1)                               | Entry point for the analysis system                         |
| **Configuration Management** | [`config.py`](config.py:1)                           | Centralized configuration for all system parameters         |
| **Data Layer**               | [`data_layer.py`](data_layer.py:1)                   | Data fetching from Yahoo Finance with robust error handling |
| **Feature Engineering**      | [`feature_engineering.py`](feature_engineering.py:1) | Technical indicator calculation                             |
| **Regime Detection**         | [`regime_detection.py`](regime_detection.py:1)       | Rule-based and ML-based regime classification               |
| **Backtesting Engine**       | [`backtesting.py`](backtesting.py:1)                 | Portfolio backtesting using VectorBT                        |
| **Stock Analyzer**           | [`stock_analyzer.py`](stock_analyzer.py:1)           | Main orchestrator that coordinates all components           |
| **Summary Analyzer**         | [`summary_analyzer.py`](summary_analyzer.py:1)       | Generates clear, actionable trading recommendations         |

### Analysis Pipeline

```
Data Fetching → Feature Engineering → Regime Detection → Backtesting → Summary Generation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone or download the project**

   ```bash
   git clone <repository-url>
   cd indostockanalysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

Key dependencies from [`requirements.txt`](requirements.txt:1):

- `yfinance==0.2.66` - Stock data fetching
- `pandas==2.3.3` - Data manipulation
- `numpy==1.23.5` - Numerical computing
- `scikit-learn==1.7.2` - Machine learning
- `vectorbt==0.28.1` - Backtesting engine
- `joblib==1.5.2` - Parallel processing

## 🚀 Usage

### Basic Commands

**Single stock analysis:**

```bash
python main.py BBCA
```

**Multiple stocks analysis:**

```bash
python main.py BBCA ASII TLKM
```

**Parallel processing (4 workers):**

```bash
python main.py BBCA ASII TLKM --jobs 4
```

**CSV export:**

```bash
python main.py BBCA ASII TLKM --csv
```

**Detailed backtest output:**

```bash
python main.py BBCA ASII TLKM --detailed
```

**Disable data caching:**

```bash
python main.py BBCA ASII TLKM --no-cache
```

### Command Line Options

| Option       | Description                       | Default        |
| ------------ | --------------------------------- | -------------- |
| `tickers`    | List of ticker symbols (required) | -              |
| `-j, --jobs` | Number of parallel workers        | -1 (all cores) |
| `--backend`  | Parallel backend (MP/LOCKY)       | LOCKY          |
| `--no-cache` | Disable data caching              | False          |
| `--detailed` | Print detailed backtest tables    | False          |
| `--csv`      | Export results to CSV             | False          |

## 🧠 Algorithm and Workflow

### Regime Detection Logic

The system uses a hybrid approach for regime detection:

**Rule-based Detection:**

- Volatility thresholds using quantile analysis
- Trend analysis with moving averages
- Price movement analysis for crash/recovery detection
- Volume anomaly detection

**ML-based Detection:**

- Ensemble of multiple classifiers:
  - Random Forest (300 estimators)
  - Gradient Boosting (200 estimators)
  - Support Vector Classifier (RBF kernel)
  - Logistic Regression
- Features include trend differences, returns, volatility, and ATR
- Confidence scoring based on prediction probabilities

**Regime Types:**

- **Bull** - Upward trending market
- **Bear** - Downward trending market
- **HighVol** - High volatility sideways movement
- **Crash** - Sharp price decline
- **Recovery** - Post-crash recovery phase
- **Sideways** - Low volatility sideways movement

### Recommendation Logic

The trading recommendation system combines multiple factors:

| Factor                  | Weight | Description                     |
| ----------------------- | ------ | ------------------------------- |
| **Regime-based**        | 40%    | Current market regime analysis  |
| **Performance Metrics** | 40%    | Returns, Sharpe ratio, win rate |
| **Confidence Scoring**  | 20%    | Analysis reliability indicators |

### Risk Assessment

Risk evaluation considers:

- Maximum drawdown analysis
- Volatility metrics
- Regime-based risk factors
- Performance consistency
- Trade frequency and reliability

## 📊 Expected Output

### Sample Output Format

```
📊 SUCCESSFUL ANALYSES (3 stocks):
================================================================================

🚀 BBCA.JK - STRONG BUY 🚀
   Next Day: BUY on next trading day
   Confidence: VERY HIGH | 🟢 Risk: VERY LOW
   Score: ★★★★★
   Regime: Bull (confidence: 0.92)
   Returns: 15.23% | Sharpe: 2.15 | Win Rate: 72.8%
   Rationale: Strong bullish regime with high confidence. Excellent historical performance...

📈 ASII.JK - BUY 📈
   Next Day: Consider buying on next trading day
   Confidence: HIGH | 🟡 Risk: LOW
   Score: ★★★★☆
   Regime: Recovery (confidence: 0.78)
   Returns: 8.45% | Sharpe: 1.42 | Win Rate: 65.3%
   Rationale: Recovery regime with good momentum. Moderate risk profile...

📉 TLKM.JK - SELL 📉
   Next Day: Consider selling on next trading day
   Confidence: MODERATE | 🔴 Risk: HIGH
   Score: ★★☆☆☆
   Regime: HighVol (confidence: 0.65)
   Returns: -3.21% | Sharpe: 0.45 | Win Rate: 48.2%
   Rationale: High volatility regime with negative returns. Elevated risk...
```

### CSV Export Format

The system exports results with the following columns:

- `no` - Sequential number
- `ticker` - Stock ticker symbol
- `regime` - Current market regime
- `confidence` - Regime confidence score
- `action` - Trading recommendation
- `risk` - Risk assessment level
- `score` - Overall score (1-5 stars)

Files are automatically named using timestamp format: `output/DDMMYY - HHMMSS.csv`

## ⚙️ Configuration

### Data Configuration ([`config.py`](config.py:7))

```python
DATA_CONFIG = {
    "period_daily": "5y",      # 5 years of daily data
    "period_intraday": "2y",   # 2 years of intraday data
    "interval_daily": "1d",    # Daily intervals
    "interval_intraday": "1h", # Hourly intervals for intraday
    "timezone": "Asia/Jakarta",# Jakarta timezone
    "ticker_suffix": ".JK"     # Indonesian stock suffix
}
```

### Feature Engineering ([`config.py`](config.py:18))

```python
FEATURE_CONFIG = {
    "moving_averages": [20, 50, 200],     # MA periods
    "return_periods": [1, 5, 20],         # Return calculation periods
    "volatility_periods": [20, 50],       # Volatility calculation
    "atr_period": 14,                     # Average True Range period
    "volume_zscore_period": 50            # Volume anomaly detection
}
```

### Backtesting Configuration ([`config.py`](config.py:68))

```python
BACKTEST_CONFIG = {
    "default": {
        "init_cash": 1_000_000,    # Initial capital
        "fees": 0.0005,            # Trading fees (0.05%)
        "slippage": 0.0005,        # Slippage (0.05%)
        "sl_stop": 0.02,           # Stop loss (2%)
        "tp_stop": 0.05,           # Take profit (5%)
        "frequency": "1h",         # Trading frequency
        "trade_regimes": ["Bull", "Recovery"]  # Regimes to trade in
    }
}
```

## 📈 Performance Characteristics

### Processing Times

- **Single stock analysis**: ~30-60 seconds
- **Parallel processing**: Scales linearly with number of cores
- **Memory usage**: Moderate (depends on data period)
- **Data caching**: Reduces subsequent analysis time by ~70%

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **CPU**: Multi-core processor recommended for parallel processing
- **Storage**: ~100MB for system + data cache
- **Internet**: Required for data fetching

## 🔧 Advanced Usage

### Custom Regime Detection

Modify [`regime_detection.py`](regime_detection.py:1) to implement custom regime logic:

```python
# Add custom regime rules
def custom_regime_rule(data):
    # Your custom logic here
    pass
```

### Custom Recommendation Logic

Modify [`summary_analyzer.py`](summary_analyzer.py:1) to adjust recommendation weights:

```python
def _generate_recommendation(self, regime_data, performance_data, confidence_data):
    # Custom weighting logic
    regime_weight = 0.4    # Regime-based recommendation
    performance_weight = 0.4  # Performance metrics
    confidence_weight = 0.2   # Confidence scoring
    # Your custom logic...
```

### Parallel Processing Backends

The system supports multiple parallel backends:

- **MP** (multiprocessing) - Process-based parallelism
- **LOCKY** (loky) - Joblib's default backend with better memory management

```bash
# Use multiprocessing backend
python main.py BBCA ASII TLKM --backend MP --jobs 4

# Use loky backend (default)
python main.py BBCA ASII TLKM --backend LOCKY --jobs 4
```

## 🐛 Troubleshooting

### Common Issues

**Data Fetching Errors:**

- Check internet connection
- Verify ticker symbols exist on Yahoo Finance
- Ensure ticker format includes `.JK` suffix for Indonesian stocks

**Memory Issues:**

- Reduce number of parallel workers with `--jobs`
- Clear cache with `--no-cache` flag
- Reduce data period in configuration

**Performance Issues:**

- Use fewer technical indicators in feature configuration
- Reduce backtesting complexity
- Use appropriate parallel backend for your system

### Debug Mode

Enable detailed logging by modifying the main script or using the `--detailed` flag for comprehensive backtest output.

## 📝 Limitations

### Current Limitations

- **Data Source**: Relies on Yahoo Finance API
- **Market Coverage**: Primarily Indonesian stocks (.JK suffix)
- **Timeframe**: Optimized for medium-term analysis (weeks to months)
- **Regime Detection**: May not capture very short-term market movements
- **ML Models**: Pre-trained on historical data, may require retraining for new market conditions

### Future Enhancements

- Additional data sources (Bloomberg, Reuters)
- Real-time analysis capabilities
- Portfolio optimization features
- Advanced risk management tools
- Web interface for easier access

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with data provider terms of service and local regulations when using this system for actual trading decisions.

## 📞 Support

For questions, issues, or suggestions:

1. Check the existing documentation
2. Review the code comments and configuration files
3. Open an issue in the project repository

---

**Disclaimer**: This system is for educational and research purposes only. Trading involves risk, and past performance is not indicative of future results. Always conduct your own research and consider consulting with a qualified financial advisor before making investment decisions.
