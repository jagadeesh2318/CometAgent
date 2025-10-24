# CometAgent - Perplexity Comet Trading Signal Generator

An intelligent trading signal generator that analyzes stocks and cryptocurrencies using technical analysis, news sentiment, and social media sentiment to produce actionable trading recommendations. The application generates browser-automation prompts specifically optimized for Perplexity Comet browser to execute trades on popular brokerage platforms.

## Overview

CometAgent combines multiple data sources and analysis techniques to generate comprehensive trading signals:

- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, ATR
- **News Sentiment**: Yahoo Finance news headlines analyzed with VADER sentiment
- **Social Media Sentiment**: X (Twitter) posts from trusted financial accounts
- **Multi-Platform Support**: Generates execution prompts for Coinbase, Fidelity, Schwab, E*TRADE, Robinhood, and IBKR
- **Multiple Time Horizons**: Short-term (3 months), medium-term (6 months), and long-term (2 years) analysis

## Features

### Core Functionality

1. **Portfolio Analysis**:
   - Reads portfolio data from CSV/Excel files
   - Supports both stocks and cryptocurrency portfolios
   - Handles flexible column naming (ticker/symbol, quantity/shares, etc.)

2. **Technical Indicators Calculation**:
   - **Moving Averages**: SMA50, SMA200, EMA12, EMA26
   - **Momentum Indicators**: MACD, RSI14
   - **Volatility Indicators**: Bollinger Bands, ATR14
   - **Trend Analysis**: Price vs. moving averages for trend determination

3. **Sentiment Analysis**:
   - **News Sentiment**: Extracts and analyzes headlines from Yahoo Finance
   - **Social Media Sentiment**: Scrapes relevant X posts from trusted financial accounts
   - **VADER Sentiment Scoring**: Converts text to numerical sentiment scores (-1 to +1)

4. **Multi-Factor Scoring System**:
   - Combines technical, news, and social sentiment scores
   - Horizon-based weighting (short-term favors technical, long-term favors news)
   - Generates final scores from -1.0 (strong sell) to +1.0 (strong buy)

5. **Trading Decision Engine**:
   - Maps scores to discrete actions: Strong Buy, Buy, Hold, Trim, Sell
   - Calculates target percentage allocation changes
   - Provides detailed rationale for each recommendation

6. **Comet Browser Optimization**:
   - Creates platform-specific browser automation prompts
   - Optimized for Perplexity Comet autonomous execution
   - Voice-command friendly prompt generation
   - Leverages Coinbase partnership for real-time crypto data
   - Includes step-by-step execution instructions
   - Handles order types (Market/Limit) and risk management

### Supported Platforms

- **Coinbase** (Crypto trading)
- **Fidelity** (Stocks/ETFs)
- **Charles Schwab** (Stocks/ETFs)
- **E*TRADE** (Stocks/ETFs)
- **Robinhood** (Stocks/Crypto)
- **Interactive Brokers (IBKR)** (Stocks/Options/Futures)

## Installation

### Prerequisites

```bash
pip install pandas numpy yfinance feedparser vaderSentiment
```

### Optional Dependencies

For enhanced functionality:
```bash
# For X (Twitter) scraping
pip install snscrape

# For additional RSS feed parsing
# feedparser is already included above
```

## Usage

### Basic Command Structure

```bash
python agentic-trader.py \
  --portfolio-type {stocks|crypto} \
  --horizon {short|medium|long} \
  --platform {coinbase|fidelity|schwab|etrade|robinhood|ibkr} \
  --file portfolio.csv
```

### Examples

#### Stock Portfolio Analysis
```bash
python agentic-trader.py \
  --portfolio-type stocks \
  --horizon medium \
  --platform fidelity \
  --file my_stocks.csv
```

#### Cryptocurrency Portfolio Analysis
```bash
python agentic-trader.py \
  --portfolio-type crypto \
  --horizon short \
  --platform coinbase \
  --file crypto_holdings.csv
```

### Portfolio File Format

Your portfolio file should be in CSV or Excel format with the following columns (case-insensitive):

#### Required:
- `ticker` or `symbol`: Stock ticker (AAPL, MSFT) or crypto symbol (BTC, ETH)

#### Optional:
- `quantity` or `shares`: Number of shares/coins owned (defaults to 0)
- `purchase_price` or `price`: Original purchase price
- `purchase_date` or `date`: Date of purchase
- `notes`: Additional notes about the position

#### Example CSV:
```csv
ticker,quantity,purchase_price,purchase_date,notes
AAPL,100,150.00,2024-01-15,Core holding
MSFT,50,300.00,2024-02-01,Tech position
BTC,0.5,45000,2024-01-10,Crypto allocation
```

## Algorithm Details

### Technical Analysis Scoring

The application calculates a comprehensive technical score using:

1. **Trend Analysis** (±0.5 points):
   - Compares current price to SMA200 for long-term trend
   - Falls back to SMA50 for shorter periods

2. **Momentum Analysis** (±0.25 points):
   - Uses MACD line vs. signal line crossovers
   - Positive when MACD > signal line

3. **RSI Mean Reversion** (±0.2 points):
   - Lower RSI values increase buy signal strength
   - Formula: `((50 - RSI) / 50) * 0.2`

4. **Bollinger Band Extremes** (±0.1 points):
   - Below lower band = potential bounce (positive)
   - Above upper band = potential pullback (negative)

### Sentiment Analysis

#### News Sentiment Process:
1. Fetches recent headlines from Yahoo Finance for the symbol
2. Optionally supplements with RSS feeds (CNBC, MarketWatch)
3. Filters headlines for symbol relevance
4. Applies VADER sentiment analysis
5. Averages sentiment scores across all headlines

#### Social Media Sentiment:
1. Uses snscrape to fetch recent X posts from trusted accounts
2. Searches for symbol mentions from financial influencers/publications
3. Applies VADER sentiment analysis to post content
4. Averages sentiment across all relevant posts

### Multi-Factor Score Combination

Final scores are weighted based on investment horizon:

#### Short-term (3 months):
- Technical Analysis: 60%
- News Sentiment: 25%
- Social Sentiment: 15%

#### Medium-term (6 months):
- Technical Analysis: 50%
- News Sentiment: 35%
- Social Sentiment: 15%

#### Long-term (2 years):
- Technical Analysis: 40%
- News Sentiment: 45%
- Social Sentiment: 15%

### Decision Mapping

Total scores are mapped to trading actions:

- **≥ 0.60**: Strong Buy (+3.0% allocation)
- **≥ 0.30**: Buy (+1.5% allocation)
- **-0.30 to 0.30**: Hold (0% change)
- **≤ -0.30**: Trim (-1.5% allocation)
- **≤ -0.60**: Sell (-3.0% allocation)

## Output Files

The application generates two output files:

### 1. Signals CSV (`signals_YYYYMMDD_HHMMSS.csv`)
Contains detailed analysis results:
- Symbol and current price
- Technical indicators (RSI, MACD, moving averages)
- Individual scores (technical, news, social)
- Final decision and target allocation change
- Detailed reasoning for each signal

### 2. Comet Browser Prompts (`prompts_YYYYMMDD_HHMMSS.txt`)
Contains browser automation instructions optimized for Perplexity Comet:
- **Voice-Command Ready**: Natural language prompts for voice execution
- **Autonomous Workflows**: Complete trading sequences for hands-free operation
- **Platform-Specific Instructions**: Tailored for each brokerage platform
- **Coinbase Integration**: Leverages Perplexity's official partnership
- **Risk Management**: Built-in safety checks and confirmation steps

## Data Sources

### Market Data:
- **Yahoo Finance**: Primary source for price data and technical indicators
- **yfinance Python library**: Historical price data and news headlines

### News Sources:
- Yahoo Finance news feeds
- Optional RSS integration (CNBC, MarketWatch)
- Configurable for additional financial news sources

### Social Media Sources:
#### Stock-focused X accounts:
- @markets (Bloomberg Markets)
- @WSJmarkets (WSJ Markets)
- @ReutersBiz (Reuters Business)
- @CNBCnow (CNBC breaking news)
- Financial influencers and analysts

#### Crypto-focused X accounts:
- @CoinDesk, @Cointelegraph
- @TheBlock__, @decryptmedia
- @glassnode, @cryptoquant_com
- DeFi and blockchain analysts

## Risk Management & Disclaimers

### Built-in Safeguards:
1. **Two-factor Authentication Handling**: Prompts pause for user confirmation
2. **Order Verification**: Requires confirmation before submission
3. **Position Size Limits**: Uses percentage-based allocation changes
4. **Error Handling**: Graceful degradation when data is unavailable

### Important Disclaimers:
- **NOT FINANCIAL ADVICE**: This is a research and automation tool
- **Educational Purpose**: Designed for learning and experimentation
- **User Responsibility**: Always review signals and prompts before execution
- **No Warranty**: Past performance doesn't guarantee future results
- **Risk Warning**: Trading involves substantial risk of loss

## Technical Architecture

### Key Classes:

#### `Position`
```python
@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    purchase_price: Optional[float] = None
    purchase_date: Optional[pd.Timestamp] = None
    notes: str = ""
```

#### `Indicators`
```python
@dataclass
class Indicators:
    close: float
    sma50: float
    sma200: float
    ema12: float
    ema26: float
    macd: float
    macd_signal: float
    rsi14: float
    bb_mid: float
    bb_up: float
    bb_dn: float
    atr14: float
```

#### `Decision`
```python
@dataclass
class Decision:
    action: str   # "Strong Buy", "Buy", "Hold", "Trim", "Sell"
    target_pct_delta: float  # Percentage allocation change
    rationale: str
```

### Core Functions:

- `analyze_positions()`: Main analysis orchestrator
- `_indicators()`: Technical indicator calculations
- `_score_ta()`: Technical analysis scoring
- `_score_news_and_x()`: Sentiment analysis
- `_combine_scores()`: Multi-factor score combination
- `_decide()`: Decision mapping
- `generate_prompts()`: Agentic prompt creation

## Configuration

### Customizable Elements:

1. **News Sources**: Modify `DEFAULT_STOCK_SOURCES` and `DEFAULT_CRYPTO_SOURCES`
2. **Social Media Accounts**: Update `DEFAULT_STOCK_X_HANDLES` and `DEFAULT_CRYPTO_X_HANDLES`
3. **Technical Indicators**: Adjust periods and weights in scoring functions
4. **Decision Thresholds**: Modify score-to-action mappings in `_decide()`
5. **Platform Instructions**: Customize broker-specific prompts

## Contributing

This tool is designed for educational and research purposes. When extending or modifying:

1. Maintain the modular architecture
2. Add comprehensive error handling
3. Include appropriate risk warnings
4. Test with paper trading first
5. Follow responsible AI development practices

## License

Educational use only. Not for commercial distribution without proper licensing.

---

**Remember**: This tool generates suggestions and automation prompts. Always verify signals, review prompts, and understand the risks before executing any trades. The financial markets are inherently risky, and automated trading amplifies both opportunities and risks.
