#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agentic_trader.py
Generate trading signals (stocks or crypto) and agentic-execution prompts
optimized specifically for Perplexity Comet browser automation.
Also supports ChatGPT Atlas on popular broker platforms.

COMET OPTIMIZATION FEATURES:
- Voice-command friendly prompts
- Autonomous workflow execution
- Real-time Coinbase integration (via Perplexity partnership)
- Context-aware trading instructions
- Streamlined no-click execution patterns

Usage
-----
python agentic_trader.py \
  --portfolio-type stocks \
  --horizon short \
  --platform coinbase \
  --file my_portfolio.csv

Inputs
------
Portfolio file (.csv or .xlsx) with columns (case-insensitive):
- ticker or symbol (e.g., AAPL, MSFT, BTC, ETH)  [required]
- quantity or shares                              [optional; defaults to 0]
- purchase_date or date                           [optional]
- purchase_price or price                         [optional]
- notes                                           [optional]

Outputs
-------
- Console report (signals + reasoning)
- prompts_{timestamp}.txt  -> copy/paste-able agentic prompts per action
- signals_{timestamp}.csv  -> tabular summary of scores and decisions

Dependencies (install as needed)
--------------------------------
pip install pandas numpy yfinance feedparser vaderSentiment snscrape

Notes
-----
- News sentiment pulls from Yahoo Finance via yfinance.get_news() when available,
  and can optionally blend in RSS (feedparser) and X posts (snscrape) if installed.
- For crypto tickers, symbols like 'BTC'/'ETH' are mapped to 'BTC-USD'/'ETH-USD' for prices.
- This is NOT financial advice; it's a research tool to aid your process.
"""

from __future__ import annotations
import argparse, datetime as dt, json, os, re, sys, textwrap
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Soft deps; handled at runtime if missing
try:
    import yfinance as yf
except Exception as e:
    print("yfinance is required: pip install yfinance", file=sys.stderr); raise

try:
    import feedparser  # optional
except Exception:
    feedparser = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

# snscrape for X (optional). Do not hard-fail if missing.
try:
    import subprocess, shutil
    _SNSCRAPE = shutil.which("snscrape") is not None
except Exception:
    _SNSCRAPE = False


# ----------------------------- Config -------------------------------- #

DEFAULT_STOCK_SOURCES = [
    # High-impact publications (not exhaustive). These serve both as human references
    # and for optional RSS lookups if you add/point to RSS feeds you prefer.
    {"name": "Reuters Markets", "url": "https://www.reuters.com/markets/"},
    {"name": "Bloomberg Markets", "url": "https://www.bloomberg.com/markets"},
    {"name": "WSJ Finance & Markets", "url": "https://www.wsj.com/finance"},
    {"name": "Financial Times Markets", "url": "https://www.ft.com/markets"},
    {"name": "CNBC Markets", "url": "https://www.cnbc.com/markets/"},
    {"name": "MarketWatch", "url": "https://www.marketwatch.com/"},
    {"name": "Morningstar News", "url": "https://www.morningstar.com/news"},
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/"},
    {"name": "Seeking Alpha", "url": "https://seekingalpha.com/"},
]

DEFAULT_CRYPTO_SOURCES = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/markets"},
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/"},
    {"name": "The Block", "url": "https://www.theblock.co/"},
    {"name": "Decrypt", "url": "https://decrypt.co/"},
    {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/"},
    {"name": "Kaiko (research)", "url": "https://www.kaiko.com/"},
    {"name": "Glassnode (research)", "url": "https://glassnode.com/"},
    {"name": "CryptoQuant (research)", "url": "https://cryptoquant.com/"},
    {"name": "Santiment (research)", "url": "https://santiment.net/"},
    {"name": "DefiLlama (dashboards)", "url": "https://defillama.com/"},
]

# Trustworthy X (Twitter) accounts to scan if snscrape is present.
DEFAULT_STOCK_X_HANDLES = [
    "markets",        # Bloomberg Markets
    "WSJmarkets",     # WSJ Markets
    "ReutersBiz",     # Reuters Business
    "CNBCnow",        # CNBC breaking
    "bespokeinvest",  # Bespoke
    "elerianm",       # Mohamed El-Erian
    "TheStalwart",    # Joe Weisenthal
    "LizAnnSonders",  # Schwab strategist
    "lisaabramowicz1" # Bloomberg Radio
]

DEFAULT_CRYPTO_X_HANDLES = [
    "CoinDesk", "Cointelegraph", "TheBlock__", "decryptmedia",
    "KaikoData", "glassnode", "cryptoquant_com", "santimentfeed",
    "DefiLlama", "MessariCrypto"
]


# --------------------------- Data classes ---------------------------- #

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    purchase_price: Optional[float] = None
    purchase_date: Optional[pd.Timestamp] = None
    notes: str = ""

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

@dataclass
class Scores:
    ta: float
    news: float
    x: float
    total: float

@dataclass
class Decision:
    action: str   # "Strong Buy", "Buy", "Hold", "Trim", "Sell"
    target_pct_delta: float  # +3.0 means increase target weight by +3%
    rationale: str


# ------------------------- Utility functions ------------------------- #

def _coerce_symbol(portfolio_type: str, s: str) -> str:
    s = s.strip().upper()
    if portfolio_type == "crypto":
        # yfinance uses -USD for most major pairs
        if "-" not in s and "/" not in s:
            return f"{s}-USD"
    return s

def _read_portfolio(path: str, portfolio_type: str) -> List[Position]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("File must be .csv or .xlsx")

    cols = {c.lower(): c for c in df.columns}
    def pick(*options):
        for o in options:
            if o in cols: return cols[o]
        return None

    sym_col = pick("ticker", "symbol")
    if not sym_col: raise ValueError("Missing column: ticker/symbol")
    qty_col = pick("quantity", "shares", "qty")
    pp_col = pick("purchase_price", "price", "cost_basis")
    pd_col = pick("purchase_date", "date")
    notes_col = pick("notes",)

    positions: List[Position] = []
    for _, row in df.iterrows():
        sym = _coerce_symbol(portfolio_type, str(row[sym_col]))
        qty = float(row[qty_col]) if qty_col and not pd.isna(row.get(qty_col, np.nan)) else 0.0
        pp = float(row[pp_col]) if pp_col and not pd.isna(row.get(pp_col, np.nan)) else None
        date = None
        if pd_col and not pd.isna(row.get(pd_col, np.nan)):
            date = pd.to_datetime(row[pd_col], errors="coerce")
        notes = str(row[notes_col]) if notes_col and not pd.isna(row.get(notes_col, np.nan)) else ""
        positions.append(Position(symbol=sym, qty=qty, purchase_price=pp, purchase_date=date, notes=notes))
    return positions

def _yf_period_for_horizon(horizon: str) -> str:
    return {"short":"3mo","medium":"6mo","long":"2y"}.get(horizon, "6mo")

def _download_history(symbol: str, horizon: str) -> pd.DataFrame:
    # daily bars are sufficient; intraday would require interval='1h' and different logic
    period = _yf_period_for_horizon(horizon)
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        # one more try: sometimes crypto pairs are weird
        if symbol.endswith("-USD"):
            df = yf.download(symbol.replace("-USD","-USDT"), period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No price data for {symbol}")
    return df

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()

def _indicators(df: pd.DataFrame) -> Indicators:
    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi14 = _rsi(close, 14)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up = bb_mid + 2*bb_std
    bb_dn = bb_mid - 2*bb_std
    atr14 = _atr(df, 14)
    last = -1
    return Indicators(
        close=float(close.iloc[last]),
        sma50=float(sma50.iloc[last]),
        sma200=float(sma200.iloc[last]) if not np.isnan(sma200.iloc[last]) else float("nan"),
        ema12=float(ema12.iloc[last]),
        ema26=float(ema26.iloc[last]),
        macd=float(macd.iloc[last]),
        macd_signal=float(macd_signal.iloc[last]),
        rsi14=float(rsi14.iloc[last]),
        bb_mid=float(bb_mid.iloc[last]),
        bb_up=float(bb_up.iloc[last]),
        bb_dn=float(bb_dn.iloc[last]),
        atr14=float(atr14.iloc[last])
    )

def _score_ta(ind: Indicators, horizon: str) -> Tuple[float,str]:
    parts = []
    score = 0.0

    # Trend bias
    if not np.isnan(ind.sma200):
        if ind.close > ind.sma200:
            score += 0.5; parts.append("Uptrend (close > SMA200)")
        else:
            score -= 0.5; parts.append("Downtrend (close < SMA200)")
    else:
        # for short periods where SMA200 is NaN
        if ind.close > ind.sma50:
            score += 0.2; parts.append("Above SMA50")
        else:
            score -= 0.2; parts.append("Below SMA50")

    # Momentum via MACD
    if ind.macd > ind.macd_signal:
        score += 0.25; parts.append("Momentum positive (MACD>signal)")
    else:
        score -= 0.25; parts.append("Momentum negative (MACD<signal)")

    # RSI contribution: lower RSI -> more upside potential; very high RSI -> downside
    rsi_term = ((50.0 - ind.rsi14) / 50.0) * 0.2
    score += rsi_term
    parts.append(f"RSI14={ind.rsi14:.1f}")

    # Bollinger extremes
    if ind.close < ind.bb_dn:
        score += 0.1; parts.append("Below lower Bollinger (mean-revert +)")
    elif ind.close > ind.bb_up:
        score -= 0.1; parts.append("Above upper Bollinger (mean-revert -)")

    # Horizon weighting (short puts more weight on momentum; long on trend)
    if horizon == "short":
        score *= 1.00
    elif horizon == "medium":
        score *= 0.95
    else: # long
        score *= 0.90

    return max(min(score, 1.0), -1.0), "; ".join(parts)

def _sentiment_vader(texts: List[str]) -> float:
    if not texts: return 0.0
    if _VADER is None: return 0.0
    vals = [_VADER.polarity_scores(t)["compound"] for t in texts if t and isinstance(t, str)]
    if not vals: return 0.0
    # clamp to [-1,1] and average
    return float(np.mean(vals))

def _news_titles_for_symbol(symbol: str, portfolio_type: str, limit: int = 20) -> List[str]:
    titles = []
    # Try Yahoo Finance news via yfinance
    try:
        news_items = yf.Ticker(symbol).get_news()  # returns list of dicts
        for it in news_items[:limit]:
            t = it.get("title") or it.get("content","")
            if t:
                # Filter: mention symbol or synonym ($AAPL, BTC, etc.) or include anyway for breadth
                if re.search(rf'\b{re.escape(symbol.split("-")[0])}\b', t, re.I) or \
                   re.search(rf'\${re.escape(symbol.split("-")[0])}\b', t, re.I):
                    titles.append(t.strip())
                else:
                    titles.append(t.strip())
    except Exception:
        pass

    # Optional: RSS blending if feedparser is present (user can swap in preferred feeds)
    if feedparser is not None:
        RSS_CANDIDATES = [
            # you can add ticker-specific feeds if you prefer
            "https://feeds.cnbc.com/rss/market.rss",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
        ]
        for url in RSS_CANDIDATES:
            try:
                d = feedparser.parse(url)
                for e in d.entries[:max(0, limit//2)]:
                    titles.append(getattr(e, "title", "").strip())
            except Exception:
                continue

    # Deduplicate, keep most recent chunk
    uniq = []
    seen = set()
    for t in titles:
        if t and t not in seen:
            uniq.append(t); seen.add(t)
    return uniq[:limit]

def _x_posts_for_symbol(symbol: str, handles: List[str], limit_per_handle: int = 5) -> List[str]:
    if not _SNSCRAPE: 
        return []
    texts = []
    base = symbol.split("-")[0]
    q_any = [base, f"${base}"]
    for handle in handles:
        for q in q_any:
            # snscrape returns JSON lines with 'content' fields if we use --jsonl
            cmd = f"snscrape --max-results {limit_per_handle} twitter-search 'from:{handle} {q}'"
            try:
                out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                for line in out.strip().splitlines():
                    try:
                        obj = json.loads(line)
                        texts.append(obj.get("content",""))
                    except Exception:
                        # older snscrape may not return jsonl; fall back to raw line
                        if line and not line.startswith("{"):
                            texts.append(line)
            except Exception:
                continue
    return texts

def _score_news_and_x(symbol: str, portfolio_type: str, horizon: str) -> Tuple[float,float,str]:
    # News titles sentiment
    news_titles = _news_titles_for_symbol(symbol, portfolio_type, limit=30)
    news_score = _sentiment_vader(news_titles)

    # X posts sentiment (optional)
    handles = DEFAULT_STOCK_X_HANDLES if portfolio_type=="stocks" else DEFAULT_CRYPTO_X_HANDLES
    x_posts = _x_posts_for_symbol(symbol, handles, limit_per_handle=5)
    x_score = _sentiment_vader(x_posts)

    # Weighting by horizon (long gives more weight to news; short balances)
    if horizon == "short":
        wn, wx = 0.65, 0.35
    elif horizon == "medium":
        wn, wx = 0.7, 0.3
    else:
        wn, wx = 0.8, 0.2

    blended = wn*news_score + wx*x_score
    detail = f"NewsSent={news_score:+.2f} from {len(news_titles)} headlines; XSent={x_score:+.2f} from {len(x_posts)} posts."
    return float(blended), float(x_score), detail

def _combine_scores(ta: float, news_blended: float, x: float, horizon: str) -> float:
    # Blend: short -> TA heavier; long -> news heavier
    if horizon == "short":
        w_ta, w_news, w_x = 0.60, 0.25, 0.15
    elif horizon == "medium":
        w_ta, w_news, w_x = 0.50, 0.35, 0.15
    else:
        w_ta, w_news, w_x = 0.40, 0.45, 0.15
    return float(np.clip(w_ta*ta + w_news*news_blended + w_x*x, -1.0, 1.0))

def _decide(total_score: float, ind: Indicators) -> Decision:
    # Map score to discrete action and target allocation delta (suggested)
    if total_score >= 0.60:
        return Decision("Strong Buy", +3.0, f"Score={total_score:+.2f} (high conviction)")
    if total_score >= 0.30:
        return Decision("Buy", +1.5, f"Score={total_score:+.2f}")
    if total_score <= -0.60:
        return Decision("Sell", -3.0, f"Score={total_score:+.2f} (high conviction)")
    if total_score <= -0.30:
        return Decision("Trim", -1.5, f"Score={total_score:+.2f}")
    return Decision("Hold", 0.0, f"Score={total_score:+.2f}")

# ---------------------- Agentic prompt generation --------------------- #

PLATFORM_ALIASES = {
    "coinbase": ["coinbase", "coin base", "cb"],
    "fidelity": ["fidelity", "fido"],
    "schwab": ["schwab", "charles schwab", "td ameritrade", "tda"],
    "etrade": ["etrade", "e*trade", "e-trade"],
    "robinhood": ["robinhood", "rh"],
    "ibkr": ["ibkr", "interactive brokers", "ib"]
}

def _normalize_platform(p: str) -> str:
    p = p.strip().lower()
    for k, vals in PLATFORM_ALIASES.items():
        if p == k or p in vals:
            return k
    return p

def _prompt_for_platform(platform: str, broker_action: str, symbol: str,
                         target_pct_delta: float, horizon: str,
                         order_type: str = "market",
                         limit_hint: Optional[float] = None) -> Dict[str,str]:
    """
    Return dict with 'comet' and 'atlas' prompts tailored to the platform.
    Comet prompts are optimized for voice commands and autonomous execution.
    Atlas prompts remain instruction-based for step-by-step execution.
    """
    # Comet-optimized base directive - action-oriented and conversational
    comet_base = textwrap.dedent(f"""
    Execute {broker_action.lower()} order for {symbol} on {platform.title()}.

    Trading parameters:
    • Action: {broker_action.upper()}
    • Symbol: {symbol}
    • Order type: {order_type.upper()}
    • Portfolio allocation: {abs(target_pct_delta):.1f}% of total account value
    • Time-in-force: DAY order

    Execution requirements:
    • Use live market price from platform
    • Calculate position size from account balance
    • Pause for 2FA if prompted - ask user for confirmation
    • Verify order details before submission
    • Confirm execution and provide order summary
    """).strip()

    # Atlas base directive - detailed instructions
    atlas_base = textwrap.dedent(f"""
    You are an agentic browser with keyboard/mouse control. Assume I'm already logged in.
    Task: Execute a {broker_action.upper()} for {symbol}.
    Constraints:
    - Order type: {order_type.upper()}
    - Target allocation change: {target_pct_delta:+.1f}% of total account value (compute from account summary).
    - Time-in-force: DAY unless exchange/platform defaults require otherwise.
    - Price source: use the live quote shown on the platform. If LIMIT, set limit {"≈ "+str(limit_hint) if limit_hint else "near the mid-price"}.
    - Risk prompts: if two-factor auth is required, pause and ask me to confirm or provide code; do NOT attempt to bypass.
    - Confirmation: after placing the order, open the Orders/History page and read back the filled/queued order details (symbol, action, qty, price, timestamp).
    """).strip()

    # Platform-specific instructions optimized for Comet vs Atlas
    comet_steps = {
        "coinbase": f"""
🚀 COMET-OPTIMIZED for Coinbase Partnership:
Access live {symbol} data through Perplexity integration. Execute {broker_action.lower()} order:
→ Use built-in Coinbase market data for {symbol} price analysis
→ Navigate to Coinbase trading interface
→ Search and select "{symbol}"
→ Choose {"Buy" if broker_action.lower() in ["buy","strong buy"] else "Sell"} with real-time pricing
→ Set order type: {order_type.title()}
→ Calculate position: account_balance × {abs(target_pct_delta)/100:.3f} ÷ live_price
→ Leverage Comet's autonomous execution for seamless order placement
→ Confirm trade with integrated verification system
        """,
        "fidelity": f"""
Go to Fidelity trading page for {symbol}. Execute {broker_action.lower()}:
→ Navigate to Trade > Stocks/ETFs
→ Enter symbol: {symbol}
→ Select {"BUY" if broker_action.lower().startswith("buy") else "SELL"} action
→ Set order type: {order_type.title()}
→ Calculate quantity: (account_value × {abs(target_pct_delta)/100:.3f}) ÷ live_price
→ Set time-in-force: DAY
→ Preview and place order
        """,
        "schwab": f"""
Access Schwab trading for {symbol}. Place {broker_action.lower()} order:
→ Go to Trade > Stocks & ETFs
→ Symbol: {symbol}, Action: {"Buy" if broker_action.lower().startswith("buy") else "Sell"}
→ Order type: {order_type.title()}
→ Shares from {abs(target_pct_delta):.1f}% portfolio allocation
→ Time-in-force: DAY
→ Review and place order
        """,
        "etrade": f"""
Navigate to E*TRADE for {symbol} trading. Execute {broker_action.lower()}:
→ Access Trading > Stocks/ETFs
→ Symbol: {symbol}
→ Action: {"Buy" if broker_action.lower().startswith("buy") else "Sell"}
→ Order: {order_type.title()}
→ Calculate quantity from {abs(target_pct_delta):.1f}% allocation
→ Set TIF: DAY, Preview and place
        """,
        "robinhood": f"""
Open {symbol} on Robinhood. Place {broker_action.lower()} order:
→ Search and select {symbol}
→ Tap Trade > {"Buy" if broker_action.lower().startswith("buy") else "Sell"}
→ Switch to {order_type.title()} order
→ Calculate: portfolio_value × {abs(target_pct_delta)/100:.3f} = order_amount
→ Review order details and submit
        """,
        "ibkr": f"""
Access IBKR order ticket for {symbol}. Execute {broker_action.lower()}:
→ Client Portal > Trade > Order Ticket
→ Symbol: {symbol}
→ Side: {"BUY" if broker_action.lower().startswith("buy") else "SELL"}
→ Type: {"MKT" if order_type=='market' else "LMT"}
→ Calculate quantity from target allocation
→ TIF: DAY, Submit and transmit
        """,
    }

    atlas_steps = {
        "coinbase": f"""
            1) Navigate to Trade → Search, find "{symbol}".
            2) Click {"'Buy'" if broker_action.lower() in ["buy","strong buy"] else "'Sell'"}.
            3) Choose {"'Market'" if order_type=='market' else "'Limit'"} order.
            4) Calculate notional = ({abs(target_pct_delta):.1f}% of account value); divide by live price to get quantity.
            5) Enter amount, click Preview, then Confirm {"Buy" if broker_action.lower().startswith("buy") else "Sell"}.
        """,
        "fidelity": f"""
            1) Go to Trade → Stocks/ETFs (web).
            2) Symbol: {symbol}. Action: {"BUY" if broker_action.lower().startswith("buy") else "SELL"}.
            3) Order type: {"Market" if order_type=='market' else "Limit"}; {"set Limit ≈ mid" if order_type!='market' else "no price field needed"}.
            4) Quantity: compute shares = round((account_value*{abs(target_pct_delta)/100:.4f})/live_price, 0).
            5) Time-in-force: DAY. Click Preview → Place Order.
        """,
        "schwab": f"""
            1) Trade → Stocks & ETFs.
            2) Symbol {symbol}; Action {"Buy" if broker_action.lower().startswith("buy") else "Sell"}; Order: {"Market" if order_type=='market' else "Limit"}.
            3) Qty from {abs(target_pct_delta):.1f}% allocation; TIF DAY → Review → Place.
        """,
        "etrade": f"""
            1) Trading → Stocks/ETFs.
            2) Enter {symbol}, {"Buy" if broker_action.lower().startswith("buy") else "Sell"}, {"Market" if order_type=='market' else "Limit"}.
            3) Compute qty from target allocation, set TIF DAY → Preview → Place.
        """,
        "robinhood": f"""
            1) Search {symbol} → Trade → {"Buy" if broker_action.lower().startswith("buy") else "Sell"}.
            2) Switch to {"'Market'" if order_type=='market' else "'Limit'"} order.
            3) Use notional = {abs(target_pct_delta):.1f}% of portfolio value to compute shares; Review → Submit.
        """,
        "ibkr": f"""
            1) Client Portal → Trade → Order Ticket.
            2) Symbol {symbol}; Side {"BUY" if broker_action.lower().startswith("buy") else "SELL"}; Type {"MKT" if order_type=='market' else "LMT"}.
            3) Quantity from target allocation; TIF DAY → Submit → Transmit.
        """,
    }

    plat = _normalize_platform(platform)

    # Get platform-specific steps for each browser
    comet_step_text = comet_steps.get(plat, f"""
Access {platform} trading for {symbol}. Execute {broker_action.lower()}:
→ Open order ticket for {symbol}
→ Select {"BUY" if broker_action.lower().startswith("buy") else "SELL"} action
→ Set order type: {order_type.title()}
→ Calculate quantity from {abs(target_pct_delta):.1f}% of account value
→ Set TIF: DAY, Preview and submit
    """)

    atlas_step_text = atlas_steps.get(plat, f"""
        1) Open the order ticket for {symbol}.
        2) Choose {"BUY" if broker_action.lower().startswith("buy") else "SELL"}; {"Market" if order_type=='market' else "Limit"}.
        3) Compute quantity from {abs(target_pct_delta):.1f}% of account value; TIF DAY → Preview → Submit.
    """)

    # Comet prompt - optimized for voice commands and autonomous execution
    comet = f"""COMET TRADING COMMAND
{comet_base}

EXECUTION WORKFLOW:
{textwrap.dedent(comet_step_text).strip()}

COMPLETION CHECKLIST:
✓ Verify symbol matches: {symbol}
✓ Confirm order type: {order_type.upper()}
✓ Validate allocation: {abs(target_pct_delta):.1f}% of portfolio
✓ Check order status and provide execution summary
✓ Report: symbol, action, quantity, price, order ID, timestamp

VOICE COMMAND: "Execute {broker_action.lower()} order for {symbol} using {abs(target_pct_delta):.1f}% portfolio allocation on {platform}"
"""

    # Atlas prompt - detailed step-by-step instructions
    atlas = f"""ATLAS RUNBOOK
Goal: {broker_action} {symbol} using {platform.title()} with {order_type.upper()} order for ≈{abs(target_pct_delta):.1f}% of account value.

Steps (be explicit, wait for UI loads, and confirm each field):
{textwrap.dedent(atlas_step_text).strip()}

After submission: navigate to Orders/History, extract and state: symbol, side, qty, order type, limit/exec price, timestamp, status.
"""

    return {"comet": comet.strip(), "atlas": atlas.strip()}


# ------------------------------ Main --------------------------------- #

def analyze_positions(portfolio_type: str, horizon: str, platform: str, positions: List[Position]) -> pd.DataFrame:
    rows = []
    for pos in positions:
        try:
            df = _download_history(pos.symbol, horizon)
            ind = _indicators(df)
            ta_score, ta_detail = _score_ta(ind, horizon)
            news_blended, x_only, nx_detail = _score_news_and_x(pos.symbol, portfolio_type, horizon)
            total = _combine_scores(ta_score, news_blended, x_only, horizon)
            decision = _decide(total, ind)
            rows.append({
                "symbol": pos.symbol,
                "close": ind.close,
                "rsi14": ind.rsi14,
                "sma50": ind.sma50,
                "sma200": ind.sma200,
                "macd": ind.macd,
                "macd_signal": ind.macd_signal,
                "atr14": ind.atr14,
                "ta_score": round(ta_score,3),
                "news_x_score": round(news_blended,3),
                "x_only_score": round(x_only,3),
                "total_score": round(total,3),
                "decision": decision.action,
                "target_pct_delta": decision.target_pct_delta,
                "ta_reason": ta_detail,
                "nx_detail": nx_detail
            })
        except Exception as e:
            rows.append({
                "symbol": pos.symbol,
                "error": str(e),
                "decision": "No Data",
                "target_pct_delta": 0.0
            })
    return pd.DataFrame(rows)

def generate_prompts(df_signals: pd.DataFrame, platform: str, horizon: str) -> Dict[str, List[Dict[str,str]]]:
    prompts = {"comet": [], "atlas": []}
    for _, r in df_signals.iterrows():
        if r.get("decision") in ["Buy","Strong Buy","Trim","Sell"]:
            side = r["decision"]
            # For simplicity, trims map to SELL, but the browser prompt explains "reduce by X%".
            broker_action = "Buy" if side in ["Buy","Strong Buy"] else "Sell"
            p = _prompt_for_platform(platform, broker_action, r["symbol"], float(r["target_pct_delta"]), horizon)
            prompts["comet"].append({"symbol": r["symbol"], "decision": side, "prompt": p["comet"]})
            prompts["atlas"].append({"symbol": r["symbol"], "decision": side, "prompt": p["atlas"]})
    return prompts

def _write_outputs(df: pd.DataFrame, prompts: Dict[str, List[Dict[str,str]]]) -> Tuple[str,str]:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sig_path = f"signals_{stamp}.csv"
    pr_path  = f"prompts_{stamp}.txt"
    df.to_csv(sig_path, index=False)
    with open(pr_path, "w", encoding="utf-8") as f:
        for mode in ["comet","atlas"]:
            f.write(f"\n===== {mode.upper()} PROMPTS =====\n\n")
            if not prompts[mode]:
                f.write("(No actionable prompts — all Hold or no data)\n\n")
            for item in prompts[mode]:
                f.write(f"### {item['symbol']} — {item['decision']}\n")
                f.write(item["prompt"])
                f.write("\n\n---\n\n")
    return sig_path, pr_path

def main():
    parser = argparse.ArgumentParser(description="Agentic Trading App (signals + executable prompts)")
    parser.add_argument("--portfolio-type", required=True, choices=["stocks","crypto"])
    parser.add_argument("--horizon", required=True, choices=["short","medium","long"])
    parser.add_argument("--platform", required=True, help="e.g., Coinbase, Fidelity, Schwab, E*TRADE, Robinhood, IBKR")
    parser.add_argument("--file", required=True, help="CSV or XLSX portfolio file")
    args = parser.parse_args()

    positions = _read_portfolio(args.file, args.portfolio_type)
    df_signals = analyze_positions(args.portfolio_type, args.horizon, args.platform, positions)
    prompts = generate_prompts(df_signals, args.platform, args.horizon)
    sig_path, pr_path = _write_outputs(df_signals, prompts)

    # Console summary
    print("\n=== Agentic Trading Report ===")
    print(f"Portfolio type: {args.portfolio_type} | Horizon: {args.horizon} | Platform: {args.platform}")
    print(df_signals[["symbol","close","total_score","decision","target_pct_delta"]].to_string(index=False))
    print(f"\nSaved signals to: {sig_path}")
    print(f"Saved agentic prompts to: {pr_path}")
    print("\nDISCLAIMER: Educational research tool. Not financial advice. Review orders before submitting.")

if __name__ == "__main__":
    main()