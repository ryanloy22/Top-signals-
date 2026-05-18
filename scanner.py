"""
TradeSignal Scanner v5.2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New in v5.2:
  - HC Telegram alerts show % move + dollar P&L at each target
  - Daily HC accuracy summary sent at 01:00 UTC via Telegram
  - Updated account balances: crypto $23k, stocks $2.1k

New in v5.1:
  - High Conviction tier (score ≥ 10, R/R ≥ 3) — instant Telegram alert
  - Telegram alerts replace Twilio SMS

New in v5.0:
  - Weighted penalty system (no hard blocks)
  - BTC correlation penalty for altcoin longs/shorts
  - Funding rate penalty/bonus
  - Session timing badges (prime vs off-hours)
  - Daily trend filter for 15m scalps
  - Minimum liquidity filter ($10M daily volume)
  - ATR-adjusted position sizing
  - Economic event awareness (Fed/CPI/NFP calendar)
  - Key S/R levels from daily chart on each signal
  - Score thresholds: stocks=5, crypto=6, high-leverage=7
  - Runs every hour 6am-6pm Denver (MST/MDT) Mon-Sun

Accounts:
  Crypto (BloFin) : $23,000 | 5x leverage | 5% risk = $1,150
  Stocks (Webull) : $2,100  | 1x          | 5% risk = $105
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, gc, json, time, math, datetime, urllib.request
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    raise ImportError("Run: pip3 install numpy pandas yfinance ta") from e

try:
    from ta.momentum import StochRSIIndicator, RSIIndicator
    from ta.trend import MACD, EMAIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import VolumeWeightedAveragePrice
except ImportError:
    raise ImportError("Run: pip3 install ta")

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG = {
    "CRYPTO_ACCOUNT":       23000,
    "STOCK_ACCOUNT":        2100,
    "RISK_PCT":             0.05,
    "CRYPTO_LEVERAGE":      5,
    "MIN_SCORE_STOCK":            5,
    "MIN_SCORE_CRYPTO":           5,
    "MIN_SCORE_HIGH_LEV":         5,    # for 5x+ leverage crypto
    "MIN_SCORE_HIGH_CONVICTION":  10,   # score ≥ 10 + R/R ≥ 3 → Telegram alert
    "MIN_RR":               2.5,
    "VOL_SPIKE_PCT":        15.0,
    "VOL_SURGE_RATIO":      1.5,
    "MIN_DAILY_VOLUME_USD": 10_000_000,  # $10M minimum liquidity
    "TOP_N":                5,

    # Penalty weights
    "PENALTY_BTC_BEAR":     2,    # penalty when BTC bearish vs altcoin long
    "PENALTY_FUNDING_HIGH": 2,    # penalty when funding > 0.05%
    "PENALTY_LOW_VOLUME":   1,    # penalty when RVOL < 0.3x
    "BONUS_PRIME_SESSION":  1,    # bonus for signals in prime trading hours
    "BONUS_FUNDING_NEG":    1,    # bonus when funding negative (short squeeze setup)

    # Alerts
    "TELEGRAM_BOT_TOKEN":   os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID":     os.getenv("TELEGRAM_CHAT_ID",   ""),
    "SENDGRID_API_KEY":     os.getenv("SENDGRID_API_KEY", ""),
    "ALERT_EMAIL_FROM":     os.getenv("ALERT_EMAIL_FROM", ""),
    "ALERT_EMAIL_TO":       os.getenv("ALERT_EMAIL_TO",   ""),

    # Alpaca paper trading
    "ALPACA_API_KEY":          os.getenv("ALPACA_API_KEY",    ""),
    "ALPACA_SECRET_KEY":       os.getenv("ALPACA_SECRET_KEY", ""),
    "ALPACA_DAILY_LOSS_LIMIT": 3.0,   # stop trading if paper account down 3% today
    "ALPACA_MAX_POSITIONS":    3,      # max concurrent open positions
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WATCHLISTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STOCKS = [
    "AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","AMD","NFLX","CRM",
    "COIN","MSTR","HOOD","MARA","RIOT","CLSK","HUT","CORZ","CIFR","BTBT",
    "PLTR","SMCI","IONQ","RKLB","SOFI","UPST","AFRM","SHOP","SNOW","DDOG",
    "SPY","QQQ","IWM","SOXL","TQQQ","ARKK","GLD","SLV","USO","TLT",
    "PYPL","SQ","V","MA","JPM","GS",
    "XOM","CVX","OXY","SLB","HAL",
]

CRYPTO = [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD",
    "TAO-USD","SUI20947-USD","HYPE-USD",
    "ARB11841-USD","OP-USD","MATIC-USD","AVAX-USD","LINK-USD","DOT-USD",
    "FET-USD","AGIX-USD","RNDR-USD","OCEAN-USD","VIRTUAL-USD",
    "UNI7083-USD","AAVE-USD","CRV-USD","MKR-USD","INJ-USD",
    "PEPE24478-USD","WIF-USD","BONK-USD","TIA-USD","SEI-USD",
    "ONDO-USD","POLYX-USD",
]

ALL_TICKERS = STOCKS + CRYPTO
HIGH_BETA   = {"TSLA","NVDA","AMD","COIN","MSTR","PLTR","SMCI","SOXL","TQQQ"}

# High leverage crypto — stricter scoring
HIGH_LEV_CRYPTO = {"ETH-USD","BTC-USD","SOL-USD","BNB-USD","AVAX-USD"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION TIMING (Denver = MST/MDT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_session_info() -> dict:
    """
    Denver time = UTC-7 (MDT summer) or UTC-6 (MST winter)
    Prime sessions:
      - US Market Open:  9:30am-11:30am ET = 7:30-9:30am Denver MDT
      - US Power Hour:   3:00pm-4:00pm ET  = 1:00-2:00pm Denver MDT
      - Asian Session:   8:00pm-11:00pm ET = 6:00-9:00pm Denver MDT
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # Approximate Denver offset (MDT = UTC-6, MST = UTC-7)
    month = now_utc.month
    denver_offset = -6 if 3 <= month <= 11 else -7
    now_denver = now_utc + datetime.timedelta(hours=denver_offset)
    hour = now_denver.hour + now_denver.minute / 60

    prime = (
        (7.5 <= hour <= 9.5) or    # US open
        (13.0 <= hour <= 14.0) or  # US power hour
        (18.0 <= hour <= 21.0)     # Asian session
    )
    in_range = 6.0 <= hour <= 18.0

    session = "PRIME 🔥" if prime else "STANDARD"
    return {
        "prime":     prime,
        "in_range":  in_range,
        "session":   session,
        "hour_denver": round(hour, 1),
        "time_denver": now_denver.strftime("%I:%M %p MT"),
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BTC TREND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_btc_trend_cache = {"trend": "neutral", "ts": 0}

def get_btc_trend() -> str:
    """Cache BTC trend for 5 minutes to avoid repeated downloads."""
    now = time.time()
    if now - _btc_trend_cache["ts"] < 300:
        return _btc_trend_cache["trend"]
    try:
        df = yf.download("BTC-USD", period="1d", interval="15m",
                        progress=False, auto_adjust=True)
        if df is None or len(df) < 20:
            return "neutral"
        close = df["Close"].squeeze()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        trend = "bullish" if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) else "bearish"
        _btc_trend_cache["trend"] = trend
        _btc_trend_cache["ts"]    = now
        return trend
    except Exception:
        return "neutral"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNDING RATES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_funding_cache = {}

def get_funding_rate(symbol: str) -> Optional[float]:
    """Get funding rate from Binance. symbol = ETHUSDT etc."""
    cache_key = symbol
    now = time.time()
    if cache_key in _funding_cache and now - _funding_cache[cache_key]["ts"] < 300:
        return _funding_cache[cache_key]["rate"]
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
            if data and len(data) > 0:
                rate = float(data[0]["fundingRate"]) * 100
                _funding_cache[cache_key] = {"rate": rate, "ts": now}
                return rate
    except Exception:
        pass
    return None

CRYPTO_FUNDING_MAP = {
    "ETH-USD": "ETHUSDT", "BTC-USD": "BTCUSDT", "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT", "AVAX-USD": "AVAXUSDT", "LINK-USD": "LINKUSDT",
    "MATIC-USD": "MATICUSDT", "INJ-USD": "INJUSDT", "AAVE-USD": "AAVEUSDT",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ECONOMIC CALENDAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_high_impact_events() -> dict:
    """
    Simple heuristic: flag known high-impact days.
    First Friday = NFP, mid-month = CPI, FOMC dates vary.
    Returns a flag and note if today is a high-impact day.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    day_of_week = now.weekday()  # 0=Monday
    day_of_month = now.day
    hour_utc = now.hour

    flags = []

    # First Friday of month = NFP
    if day_of_week == 4 and day_of_month <= 7:
        flags.append("⚠️ NFP Day — Nonfarm Payrolls release. Expect volatility.")

    # ~10th-15th of month = CPI
    if 10 <= day_of_month <= 15 and day_of_week < 5:
        flags.append("⚠️ Potential CPI week — check calendar before trading.")

    # Pre-market hours (before 8am ET = 13:00 UTC)
    if day_of_week < 5 and hour_utc < 13:
        flags.append("📋 Pre-market — lower liquidity, wider spreads.")

    return {
        "has_event": len(flags) > 0,
        "flags": flags,
        "note": " | ".join(flags) if flags else None,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MARKET SENTIMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_fear_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.loads(r.read().decode())["data"][0]
            return {"score": int(d["value"]), "label": d["value_classification"]}
    except Exception:
        return {"score": 50, "label": "Neutral"}

def get_vix():
    try:
        hist = yf.Ticker("^VIX").history(period="2d")
        if hist.empty: return {"value": None, "level": "unknown"}
        val = round(float(hist["Close"].iloc[-1]), 2)
        level = "low" if val < 15 else "moderate" if val < 20 else "elevated" if val < 30 else "extreme"
        return {"value": val, "level": level}
    except Exception:
        return {"value": None, "level": "unknown"}

def get_sentiment():
    fg, vix = get_fear_greed(), get_vix()
    s, v = fg["score"], vix["value"]
    if s >= 60 and (v is None or v < 20):   bias, icon = "RISK-ON",  "🟢"
    elif s <= 40 or (v is not None and v >= 25): bias, icon = "RISK-OFF", "🔴"
    else:                                        bias, icon = "NEUTRAL",  "🟡"
    return {
        "bias": bias, "icon": icon,
        "fear_greed_score": s, "fear_greed_label": fg["label"],
        "vix": v, "vix_level": vix["level"],
        "macro_note": "⚠️ US-Iran tensions active. Oil elevated. Verify news before entering.",
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JEDI GREEN LIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gaussian_weight(i, j, bw):
    return math.exp(-((i - j) ** 2) / (bw * bw * 2))

def jedi_green_lights(prices: pd.Series, bandwidth: float = 19.0) -> dict:
    prices = prices.reset_index(drop=True)
    n = len(prices)
    results = []
    for i in [n - 2, n - 1]:
        total, wsum = 0.0, 0.0
        for j in range(max(0, i - 499), i + 1):
            w = gaussian_weight(i, j, bandwidth)
            total += prices.iloc[j] * w
            wsum  += w
        results.append(total / wsum if wsum > 0 else float("nan"))
    slope = results[1] - results[0]
    return {
        "signal":   "bullish" if slope > 0 else "bearish",
        "smoothed": round(results[1], 4) if not math.isnan(results[1]) else None,
        "slope":    round(slope, 6) if not math.isnan(slope) else None,
        "strength": round(abs(slope), 6) if not math.isnan(slope) else None,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calc_atr(high, low, close, period=14):
    try:
        return float(AverageTrueRange(high, low, close, window=period).average_true_range().iloc[-1])
    except Exception:
        return float((high - low).tail(period).mean())

def get_daily_trend(ticker: str) -> str:
    """Get daily trend for scalp filter."""
    try:
        df = yf.download(ticker, period="3mo", interval="1d",
                        progress=False, auto_adjust=True)
        if df is None or len(df) < 20:
            return "neutral"
        close = df["Close"].squeeze()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        if float(ema20.iloc[-1]) > float(ema50.iloc[-1]):
            return "bullish"
        return "bearish"
    except Exception:
        return "neutral"

def get_daily_sr_levels(ticker: str) -> dict:
    """Get previous day high/low and weekly open as S/R levels."""
    try:
        df = yf.download(ticker, period="5d", interval="1d",
                        progress=False, auto_adjust=True)
        if df is None or len(df) < 2:
            return {}
        prev = df.iloc[-2]
        return {
            "prev_day_high": round(float(prev["High"]), 4),
            "prev_day_low":  round(float(prev["Low"]),  4),
            "prev_day_close": round(float(prev["Close"]), 4),
        }
    except Exception:
        return {}

def check_min_liquidity(ticker: str, avg_vol: float, price: float) -> bool:
    """Check if ticker has minimum $10M daily volume."""
    if ticker.endswith("-USD"):
        # Crypto — use volume * price
        daily_vol_usd = avg_vol * price
        return daily_vol_usd >= CONFIG["MIN_DAILY_VOLUME_USD"]
    else:
        # Stocks — avg volume * price
        daily_vol_usd = avg_vol * price
        return daily_vol_usd >= CONFIG["MIN_DAILY_VOLUME_USD"]

def calc_atr_position_size(ticker, price, stop_loss, atr_val, is_crypto, direction):
    """
    ATR-adjusted position sizing.
    When ATR is high (volatile) = smaller position.
    When ATR is low (calm) = larger position (up to max).
    """
    try:
        if is_crypto:
            account  = CONFIG["CRYPTO_ACCOUNT"]
            leverage = CONFIG["CRYPTO_LEVERAGE"]
        else:
            account  = CONFIG["STOCK_ACCOUNT"]
            leverage = 1

        max_risk    = account * CONFIG["RISK_PCT"]
        risk_per    = abs(price - stop_loss)
        if risk_per <= 0:
            return None

        # ATR adjustment: normalize ATR as % of price
        atr_pct = atr_val / price if price > 0 else 0
        # Scale factor: if ATR > 3% reduce size, if ATR < 1% increase size
        if atr_pct > 0.03:    scale = 0.7
        elif atr_pct > 0.02:  scale = 0.85
        elif atr_pct < 0.005: scale = 1.2
        elif atr_pct < 0.01:  scale = 1.1
        else:                  scale = 1.0

        adjusted_risk = max_risk * scale
        raw_units     = adjusted_risk / risk_per
        units         = raw_units * leverage if is_crypto else raw_units
        pos_value     = units * price / leverage if is_crypto else units * price

        # Cap at 30% of account
        max_pos = account * 0.30 * leverage if is_crypto else account * 0.30
        if pos_value > max_pos:
            units     = max_pos / price
            pos_value = max_pos

        return {
            "units":       round(units, 4),
            "dollar_risk": round(units * risk_per / leverage if is_crypto else units * risk_per, 2),
            "pos_value":   round(pos_value, 2),
            "account":     "crypto" if is_crypto else "stocks",
            "leverage":    leverage,
            "atr_scale":   round(scale, 2),
            "atr_pct":     round(atr_pct * 100, 3),
        }
    except Exception:
        return None

def get_higher_tf_bias(ticker, primary_tf):
    try:
        if primary_tf == "15m":
            higher_tfs = [("1h", "1mo"), ("4h", "3mo")]
        elif primary_tf == "1d":
            higher_tfs = [("1wk", "1y")]
        else:
            return "neutral"
        signals = []
        for htf, period in higher_tfs:
            df = yf.download(ticker, period=period, interval=htf,
                           progress=False, auto_adjust=True)
            if df is None or len(df) < 20:
                continue
            close = df["Close"].squeeze()
            ema5  = close.ewm(span=5,  adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            signals.append("bullish" if float(ema5.iloc[-1]) > float(ema20.iloc[-1]) else "bearish")
            time.sleep(0.1)
        if not signals: return "neutral"
        return "bullish" if signals.count("bullish") > signals.count("bearish") else "bearish"
    except Exception:
        return "neutral"

def check_earnings(ticker):
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty: return False
        if hasattr(cal, 'columns'):
            for col in cal.columns:
                val = cal[col].iloc[0] if len(cal) > 0 else None
                if val and hasattr(val, 'date'):
                    days = (val.date() - datetime.date.today()).days
                    if 0 <= days <= 7: return True
        return False
    except Exception:
        return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE SIGNAL ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_ticker(ticker: str, btc_trend: str, session: dict) -> Optional[dict]:
    try:
        is_crypto  = ticker.endswith("-USD")
        is_hbeta   = ticker in HIGH_BETA
        is_high_lev = ticker in HIGH_LEV_CRYPTO
        tf         = "15m" if (is_crypto or is_hbeta) else "1d"
        period     = "5d"  if tf == "15m" else "6mo"

        df = yf.download(ticker, period=period, interval=tf,
                        progress=False, auto_adjust=True)
        if df is None or len(df) < 55:
            return None

        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        vol   = df["Volume"].squeeze()

        price     = float(close.iloc[-1])
        avg_vol   = float(vol.tail(20).mean())
        cur_vol   = float(vol.iloc[-1])

        # ── Liquidity filter ─────────────────────────────────────────────
        if not check_min_liquidity(ticker, avg_vol, price):
            return None

        # ── EMAs ─────────────────────────────────────────────────────────
        ema5  = close.ewm(span=5,  adjust=False).mean()
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        e5, e9, e50 = float(ema5.iloc[-1]), float(ema9.iloc[-1]), float(ema50.iloc[-1])
        e5p, e9p    = float(ema5.iloc[-2]), float(ema9.iloc[-2])

        cross_bull = e5p < e9p and e5 > e9
        cross_bear = e5p > e9p and e5 < e9
        trend_up   = e5 > e9 > e50
        trend_down = e5 < e9 < e50

        # ── MACD ─────────────────────────────────────────────────────────
        macd_i = MACD(close)
        ml = float(macd_i.macd().iloc[-1])
        ms = float(macd_i.macd_signal().iloc[-1])
        mh = float(macd_i.macd_diff().iloc[-1])
        macd_bull = ml > ms and mh > 0
        macd_bear = ml < ms and mh < 0

        # ── StochRSI ──────────────────────────────────────────────────────
        srsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        sk = float(srsi.stochrsi_k().iloc[-1])
        sd = float(srsi.stochrsi_d().iloc[-1])
        srsi_bull = sk > sd and sk < 0.8
        srsi_bear = sk < sd and sk > 0.2

        # ── RSI + Divergence ──────────────────────────────────────────────
        rsi_ind    = RSIIndicator(close, window=14)
        rsi_series = rsi_ind.rsi()
        rsi_val    = float(rsi_series.iloc[-1])

        # RSI divergence
        try:
            pr = close.values[-14:]
            rs = rsi_series.values[-14:]
            bull_div = min(pr[-5:]) < min(pr[:5]) and min(rs[-5:]) > min(rs[:5])
            bear_div = max(pr[-5:]) > max(pr[:5]) and max(rs[-5:]) < max(rs[:5])
        except Exception:
            bull_div = bear_div = False
        rsi_div = {"bullish": bool(bull_div), "bearish": bool(bear_div)}

        # ── VWAP ─────────────────────────────────────────────────────────
        try:
            vwap_ind = VolumeWeightedAveragePrice(
                high=high, low=low, close=close, volume=vol)
            vwap_val  = float(vwap_ind.volume_weighted_average_price().iloc[-1])
            vwap_bull = price > vwap_val
            vwap_bear = price < vwap_val
        except Exception:
            vwap_val  = None
            vwap_bull = vwap_bear = False

        # ── Volume / RVOL ─────────────────────────────────────────────────
        vol_ratio  = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0
        vol_surge  = vol_ratio >= CONFIG["VOL_SURGE_RATIO"]

        # ── Bollinger ─────────────────────────────────────────────────────
        bb     = BollingerBands(close, window=20, window_dev=2)
        bw     = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1])
        bw_avg = float((bb.bollinger_hband() - bb.bollinger_lband()).tail(20).mean())
        bb_squeeze = bw < bw_avg * 0.75

        # ── Volatility spike ──────────────────────────────────────────────
        prev       = float(close.iloc[-2])
        candle_pct = round(abs(price - prev) / prev * 100, 2) if prev > 0 else 0
        vol_spike  = candle_pct >= CONFIG["VOL_SPIKE_PCT"]

        # ── Jedi Green Lights ─────────────────────────────────────────────
        gl = jedi_green_lights(close)

        # ── Base Score ────────────────────────────────────────────────────
        bull_score = sum([
            cross_bull * 3, trend_up * 2, macd_bull * 2,
            (gl["signal"] == "bullish") * 2, vwap_bull * 2,
            bull_div * 2, srsi_bull * 1, vol_surge * 1,
            bb_squeeze * 1, vol_spike * 1,
        ])
        bear_score = sum([
            cross_bear * 3, trend_down * 2, macd_bear * 2,
            (gl["signal"] == "bearish") * 2, vwap_bear * 2,
            bear_div * 2, srsi_bear * 1, vol_surge * 1,
            bb_squeeze * 1, vol_spike * 1,
        ])

        direction = "LONG" if bull_score >= bear_score else "SHORT"
        score     = bull_score if direction == "LONG" else bear_score

        # ── Bonuses ───────────────────────────────────────────────────────
        penalty_notes = []

        # Funding rate (still fetched for display, no penalty)
        if is_crypto and ticker != "BTC-USD":
            binance_sym  = CRYPTO_FUNDING_MAP.get(ticker)
            funding_rate = get_funding_rate(binance_sym) if binance_sym else None
        else:
            funding_rate = None

        # Session bonus
        if session["prime"]:
            score += CONFIG["BONUS_PRIME_SESSION"]
            penalty_notes.append(f"Prime session +{CONFIG['BONUS_PRIME_SESSION']}")

        # ── Daily trend (info only, no penalty) ───────────────────────────
        daily_trend = "neutral"
        if tf == "15m":
            daily_trend = get_daily_trend(ticker)

        # ── MTF Confirmation (info only, no penalty) ──────────────────────
        htf_bias = get_higher_tf_bias(ticker, tf)
        mtf_confirmed = (
            (direction == "LONG"  and htf_bias == "bullish") or
            (direction == "SHORT" and htf_bias == "bearish") or
            htf_bias == "neutral"
        )

        # ── Score threshold check ─────────────────────────────────────────
        if is_high_lev:
            min_score = CONFIG["MIN_SCORE_HIGH_LEV"]
        elif is_crypto:
            min_score = CONFIG["MIN_SCORE_CRYPTO"]
        else:
            min_score = CONFIG["MIN_SCORE_STOCK"]

        if score < min_score:
            return None

        # ── ATR-adjusted position sizing ──────────────────────────────────
        atr_val   = calc_atr(high, low, close)
        if direction == "LONG":
            stop_loss = max(e9 * 0.995, price - 1.5 * atr_val)
            risk      = price - stop_loss
            if risk <= 0: return None
            t1 = round(price + risk * 2.0, 4)
            t2 = round(price + risk * 3.5, 4)
            t3 = round(price + risk * 5.0, 4)
        else:
            stop_loss = min(e9 * 1.005, price + 1.5 * atr_val)
            risk      = stop_loss - price
            if risk <= 0: return None
            t1 = round(price - risk * 2.0, 4)
            t2 = round(price - risk * 3.5, 4)
            t3 = round(price - risk * 5.0, 4)

        rr = round(abs(t3 - price) / risk, 2)
        if rr < CONFIG["MIN_RR"]: return None

        pos = calc_atr_position_size(ticker, price, stop_loss, atr_val, is_crypto, direction)

        # ── Key S/R levels ────────────────────────────────────────────────
        sr_levels = get_daily_sr_levels(ticker)

        # ── Earnings ─────────────────────────────────────────────────────
        has_earnings = check_earnings(ticker) if not is_crypto else False

        # ── Signal type ───────────────────────────────────────────────────
        if vol_spike and vol_surge:      stype = "VOLATILITY SPIKE 🚨"
        elif cross_bull or cross_bear:   stype = "EMA CROSSOVER ⚡"
        elif bull_div or bear_div:       stype = "RSI DIVERGENCE 🔀"
        elif bb_squeeze and vol_surge:   stype = "SQUEEZE BREAKOUT 💥"
        else:                            stype = "MOMENTUM 📈"

        def safe(v):
            if v is None: return None
            try:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
                return v
            except Exception:
                return None

        return {
            "ticker":         ticker,
            "type":           "crypto" if is_crypto else "stock",
            "signal_type":    stype,
            "trade_type":     "⚡ Scalp (mins-hours)" if tf == "15m" else "📅 Swing (days-weeks)",
            "price":          round(price, 6),
            "direction":      direction,
            "score":          score,
            "rr_ratio":       rr,
            "entry":          round(price, 6),
            "stop_loss":      round(stop_loss, 6),
            "target1":        t1,
            "target2":        t2,
            "target3":        t3,
            "risk_per_unit":  round(risk, 6),
            "atr":            round(atr_val, 6),
            "vwap":           safe(vwap_val),
            "vwap_bias":      "above" if vwap_bull else "below",
            "rsi":            round(rsi_val, 2),
            "rsi_divergence": rsi_div,
            "rvol":           vol_ratio,
            "vol_ratio":      vol_ratio,
            "candle_pct":     candle_pct,
            "vol_spike":      vol_spike,
            "bb_squeeze":     bb_squeeze,
            "ema_cross":      cross_bull or cross_bear,
            "macd_signal":    "bull" if macd_bull else "bear",
            "stochrsi_k":     round(sk, 3),
            "mtf_bias":       htf_bias,
            "mtf_confirmed":  mtf_confirmed,
            "daily_trend":    daily_trend,
            "funding_rate":   safe(funding_rate),
            "btc_trend":      btc_trend if is_crypto else None,
            "session":        session["session"],
            "penalty_notes":  penalty_notes,
            "sr_levels":      sr_levels,
            "has_earnings":   has_earnings,
            "green_lights":   gl,
            "position":         pos,
            "timeframe":        tf,
            "high_conviction":  score >= CONFIG["MIN_SCORE_HIGH_CONVICTION"] and rr >= 3.0,
            "scanned_at":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    except Exception as e:
        print(f"  ⚠ {ticker}: {e}")
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALERTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_digest(scalps, swings, sentiment, session, events):
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        "="*50,
        f"TRADESIGNAL v5.0 | {session['time_denver']} | {session['session']}",
        f"Time   : {now}",
        f"Market : {sentiment['icon']} {sentiment['bias']} | F&G:{sentiment['fear_greed_score']} | VIX:{sentiment['vix']}",
        sentiment['macro_note'],
    ]
    if events["has_event"]:
        for flag in events["flags"]:
            lines.append(flag)
    lines += ["="*50, "", f"⚡ TOP SCALPS — {len(scalps)} signals", "-"*45]
    for i, r in enumerate(scalps, 1):
        gl  = r["green_lights"]
        pos = r.get("position") or {}
        arr = "▲" if r["direction"] == "LONG" else "▼"
        sr  = r.get("sr_levels", {})
        earn = " ⚠️EARNINGS" if r.get("has_earnings") else ""
        pen  = " [" + ", ".join(r.get("penalty_notes",[])) + "]" if r.get("penalty_notes") else ""
        lines += [
            f"{i}. {r['ticker']} {arr}{r['direction']} | {r['signal_type']}{earn}",
            f"   Entry:{r['entry']}  Stop:{r['stop_loss']}  T1:{r['target1']}  T3:{r['target3']}",
            f"   R/R:{r['rr_ratio']}  Score:{r['score']}  MTF:{r.get('mtf_bias','?').upper()}  Session:{r.get('session','?')}",
            f"   GL:{'BULL' if gl['signal']=='bullish' else 'BEAR'}  RVOL:{r['rvol']}x  Daily:{r.get('daily_trend','?')}",
        ]
        if sr:
            lines.append(f"   S/R: PrevH:{sr.get('prev_day_high','?')} PrevL:{sr.get('prev_day_low','?')}")
        if r.get("funding_rate") is not None:
            lines.append(f"   Funding:{r['funding_rate']:.4f}%  BTC:{r.get('btc_trend','?')}")
        if pen:
            lines.append(f"   Adjustments:{pen}")
        if pos:
            lines.append(f"   Pos:{pos.get('units','?')} units | Risk:${pos.get('dollar_risk','?')} | ATR scale:{pos.get('atr_scale','?')}x")
        lines.append("")
    lines += [f"📅 TOP SWINGS — {len(swings)} signals", "-"*45]
    for i, r in enumerate(swings, 1):
        gl  = r["green_lights"]
        pos = r.get("position") or {}
        arr = "▲" if r["direction"] == "LONG" else "▼"
        earn = " ⚠️EARNINGS" if r.get("has_earnings") else ""
        lines += [
            f"{i}. {r['ticker']} {arr}{r['direction']} | {r['signal_type']}{earn}",
            f"   Entry:{r['entry']}  Stop:{r['stop_loss']}  T1:{r['target1']}  T3:{r['target3']}",
            f"   R/R:{r['rr_ratio']}  Score:{r['score']}  MTF:{r.get('mtf_bias','?').upper()}",
            f"   GL:{'BULL' if gl['signal']=='bullish' else 'BEAR'}  RVOL:{r['rvol']}x",
        ]
        if pos:
            lines.append(f"   Pos:{pos.get('units','?')} units | Risk:${pos.get('dollar_risk','?')} | ATR scale:{pos.get('atr_scale','?')}x")
        lines.append("")
    lines.append("="*50)
    return "\n".join(lines)

def send_email(subject, body):
    if not SENDGRID_AVAILABLE:
        print("  ⚠ Run: pip3 install sendgrid"); return
    c = CONFIG
    if not all([c["SENDGRID_API_KEY"], c["ALERT_EMAIL_FROM"], c["ALERT_EMAIL_TO"]]):
        print("  ⚠ SendGrid credentials missing"); return
    try:
        sg = sendgrid.SendGridAPIClient(api_key=c["SENDGRID_API_KEY"])
        sg.send(Mail(from_email=c["ALERT_EMAIL_FROM"], to_emails=c["ALERT_EMAIL_TO"],
                     subject=subject, plain_text_content=body))
        print("  ✅ Email sent")
    except Exception as e:
        print(f"  ⚠ Email failed: {e}")

def send_telegram(message: str):
    token   = CONFIG.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = CONFIG.get("TELEGRAM_CHAT_ID",   "")
    if not token or not chat_id:
        print("  ⚠ Telegram credentials missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return
    try:
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
        req     = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read().decode())
            if resp.get("ok"):
                print("  ✅ Telegram sent")
            else:
                print(f"  ⚠ Telegram error: {resp}")
    except Exception as e:
        print(f"  ⚠ Telegram failed: {e}")

def format_hc_telegram(r: dict) -> str:
    arr   = "▲" if r["direction"] == "LONG" else "▼"
    earn  = "  ⚠️ EARNINGS WEEK" if r.get("has_earnings") else ""
    pos   = r.get("position") or {}
    dr    = pos.get("dollar_risk", 0) or 0
    acct  = CONFIG["CRYPTO_ACCOUNT"] if pos.get("account") == "crypto" else CONFIG["STOCK_ACCOUNT"]
    entry = r["entry"]
    direction = r["direction"]

    def pct(target):
        if direction == "LONG":
            return (target - entry) / entry * 100
        return (entry - target) / entry * 100

    def fmt_target(label, target, rr):
        profit   = dr * rr
        acct_pct = profit / acct * 100 if acct else 0
        move     = pct(target)
        return f"{label}:     {target}  (+{move:.1f}% | +${profit:.0f} | +{acct_pct:.2f}% acct)"

    stop_move    = abs(pct(r["stop_loss"]))
    stop_acct    = dr / acct * 100 if acct else 0

    lines = [
        f"🔥 <b>HIGH CONVICTION TRADE</b>",
        f"<b>{r['ticker']}</b>  {arr}{r['direction']}  |  {r['signal_type']}{earn}",
        f"Score: {r['score']}  |  R/R: {r['rr_ratio']}:1  |  {r.get('session','?')}",
        f"",
        f"Entry:   {entry}",
        f"Stop:    {r['stop_loss']}  (-{stop_move:.1f}% | -${dr:.0f} | -{stop_acct:.2f}% acct)",
        fmt_target("T1", r["target1"], 2.0),
        fmt_target("T2", r["target2"], 3.5),
        fmt_target("T3", r["target3"], 5.0),
    ]
    if pos:
        lines.append(f"Size:    {pos.get('units','?')} units  |  Risk: ${dr:.0f}")
    if r.get("funding_rate") is not None:
        lines.append(f"Funding: {r['funding_rate']:.4f}%  |  BTC: {r.get('btc_trend','?')}")
    return "\n".join(lines)

def save_hc_alert(signal: dict, output_dir: str):
    try:
        path   = os.path.join(output_dir, "hc_alerts.json")
        alerts = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    alerts = json.load(f)
            except Exception:
                alerts = []
        alerts.append(clean_nan({
            "alerted_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ticker":      signal["ticker"],
            "direction":   signal["direction"],
            "entry":       signal["entry"],
            "stop_loss":   signal["stop_loss"],
            "target1":     signal["target1"],
            "target2":     signal["target2"],
            "target3":     signal["target3"],
            "score":       signal["score"],
            "rr_ratio":    signal["rr_ratio"],
            "signal_type": signal["signal_type"],
            "position":    signal.get("position"),
        }))
        alerts = alerts[-300:]
        with open(path, "w") as f:
            json.dump(alerts, f, indent=2, cls=SafeEncoder)
    except Exception as e:
        print(f"  ⚠ save_hc_alert: {e}")

def check_outcome(alert: dict) -> dict:
    ticker     = alert["ticker"]
    direction  = alert["direction"]
    entry      = float(alert["entry"])
    stop       = float(alert["stop_loss"])
    t1, t2, t3 = float(alert["target1"]), float(alert["target2"]), float(alert["target3"])
    alerted_at = datetime.datetime.fromisoformat(alert["alerted_at"])
    try:
        df = yf.download(ticker, start=alerted_at.strftime("%Y-%m-%d"),
                         interval="15m", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {"outcome": "no data"}
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index >= alerted_at]
        if df.empty:
            return {"outcome": "no data"}
        hi  = float(df["High"].max())
        lo  = float(df["Low"].min())
        cur = float(df["Close"].squeeze().iloc[-1])
        if direction == "LONG":
            hit_t3, hit_t2, hit_t1 = hi >= t3, hi >= t2, hi >= t1
            hit_stop = lo <= stop
        else:
            hit_t3, hit_t2, hit_t1 = lo <= t3, lo <= t2, lo <= t1
            hit_stop = hi >= stop
        if hit_t3:        label = "T3 ✅✅✅"
        elif hit_t2:      label = "T2 ✅✅"
        elif hit_t1:      label = "T1 ✅"
        elif hit_stop:    label = "Stop ❌"
        else:             label = "Open ⏳"
        return {"outcome": label, "current": round(cur, 4), "high": round(hi, 4), "low": round(lo, 4)}
    except Exception:
        return {"outcome": "error"}

def send_daily_hc_summary(output_dir: str):
    path = os.path.join(output_dir, "hc_alerts.json")
    if not os.path.exists(path):
        print("  No hc_alerts.json found.")
        return
    with open(path) as f:
        alerts = json.load(f)
    cutoff      = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
    day_alerts  = [a for a in alerts
                   if datetime.datetime.fromisoformat(a["alerted_at"]) >= cutoff]
    if not day_alerts:
        send_telegram("📊 <b>Daily HC Summary</b>\nNo High Conviction alerts in the last 24 hours.")
        return
    print(f"\n  📊 Checking outcomes for {len(day_alerts)} HC alert(s)...")
    wins = losses = opens = 0
    lines = [
        f"📊 <b>DAILY HC SUMMARY</b>",
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')}  —  {len(day_alerts)} alert(s)",
        "",
    ]
    for i, alert in enumerate(day_alerts, 1):
        result = check_outcome(alert)
        outcome = result.get("outcome", "?")
        pos  = alert.get("position") or {}
        dr   = (pos.get("dollar_risk") or 0)
        acct = CONFIG["CRYPTO_ACCOUNT"] if (pos.get("account") == "crypto") else CONFIG["STOCK_ACCOUNT"]
        if "T3" in outcome:
            profit = dr * 5.0;  pnl = f"+${profit:.0f} (+{profit/acct*100:.2f}% acct)"; wins += 1
        elif "T2" in outcome:
            profit = dr * 3.5;  pnl = f"+${profit:.0f} (+{profit/acct*100:.2f}% acct)"; wins += 1
        elif "T1" in outcome:
            profit = dr * 2.0;  pnl = f"+${profit:.0f} (+{profit/acct*100:.2f}% acct)"; wins += 1
        elif "Stop" in outcome:
            pnl = f"-${dr:.0f} (-{dr/acct*100:.2f}% acct)"; losses += 1
        else:
            pnl = "—"; opens += 1
        arr  = "▲" if alert["direction"] == "LONG" else "▼"
        time_str = datetime.datetime.fromisoformat(alert["alerted_at"]).strftime("%H:%M UTC")
        lines += [
            f"{i}. <b>{alert['ticker']}</b> {arr}{alert['direction']}  Score:{alert['score']}  {time_str}",
            f"   {outcome}  |  P&L: {pnl}",
            f"   Entry:{alert['entry']}  Now:{result.get('current','?')}  H:{result.get('high','?')}  L:{result.get('low','?')}",
            "",
        ]
    resolved = wins + losses
    if resolved:
        lines.append(f"Win rate: {wins}/{resolved} ({round(wins/resolved*100)}%)  |  {opens} still open")
    else:
        lines.append(f"All {opens} alert(s) still open / no data yet")
    send_telegram("\n".join(lines))
    print(f"  ✅ Summary sent — {wins}W / {losses}L / {opens} open")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HISTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_history(results, output_dir):
    try:
        path = os.path.join(output_dir, "scan_history.json")
        history = []
        if os.path.exists(path):
            with open(path, "r") as f:
                history = json.load(f)
        entry = {
            "scanned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "signals": [{
                "ticker": r["ticker"], "direction": r["direction"],
                "entry": r["entry"], "stop_loss": r["stop_loss"],
                "target1": r["target1"], "target3": r["target3"],
                "score": r["score"], "timeframe": r["timeframe"],
            } for r in results]
        }
        history.append(entry)
        history = history[-500:]
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  ⚠ History: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_setup(r):
    gl  = r["green_lights"]
    pos = r.get("position") or {}
    arr = "▲ LONG" if r["direction"] == "LONG" else "▼ SHORT"
    sl_note = "← above entry" if r["direction"] == "SHORT" else "← below entry"
    sr  = r.get("sr_levels", {})
    pen = r.get("penalty_notes", [])

    print(f"\n  {r['ticker']} ({r['type'].upper()}) | {arr} | {r['signal_type']}")
    print(f"  {r['trade_type']} | Session: {r.get('session','?')}")
    print(f"  Score:{r['score']}  R/R:{r['rr_ratio']}:1  RSI:{r.get('rsi','?')}  RVOL:{r.get('rvol','?')}x")
    if r.get("funding_rate") is not None:
        print(f"  Funding:{r['funding_rate']:.4f}%  BTC Trend:{r.get('btc_trend','?')}  Daily:{r.get('daily_trend','?')}")
    print(f"  VWAP:{r.get('vwap','N/A')} ({'above' if r.get('vwap_bias')=='above' else 'below'})  MTF:{r.get('mtf_bias','?').upper()} {'✅' if r.get('mtf_confirmed') else '⚠️'}")
    print(f"  Entry    : {r['entry']}")
    print(f"  Stop Loss: {r['stop_loss']}  {sl_note}")
    if sr:
        print(f"  Key S/R  : PrevH:{sr.get('prev_day_high','?')}  PrevL:{sr.get('prev_day_low','?')}")
    print(f"  Target 1 : {r['target1']}  (2:1)")
    print(f"  Target 2 : {r['target2']}  (3.5:1)")
    print(f"  Target 3 : {r['target3']}  (5:1)")
    print(f"  GL: {'🟢 Bull' if gl['signal']=='bullish' else '🟣 Bear'}  slope:{gl.get('slope','?')}")
    if pen:
        print(f"  Adjustments: {' | '.join(pen)}")
    if pos:
        print(f"  Position : {pos.get('units','?')} units | Risk:${pos.get('dollar_risk','?')} | Value:${pos.get('pos_value','?')} | ATR scale:{pos.get('atr_scale','?')}x ({pos.get('atr_pct','?')}% ATR)")
    if r.get("has_earnings"):
        print(f"  ⚠️  EARNINGS WITHIN 7 DAYS — trade carefully")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON CLEANER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):  # numpy scalar
            v = obj.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        return super().default(obj)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALPACA PAPER TRADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# yfinance → Alpaca symbol map for special crypto tickers
_CRYPTO_SYMBOL_MAP = {
    "SUI20947-USD": "SUI/USD", "UNI7083-USD": "UNI/USD",
    "PEPE24478-USD": "PEPE/USD", "ARB11841-USD": "ARB/USD",
}

def alpaca_symbol(ticker: str) -> str:
    if ticker in _CRYPTO_SYMBOL_MAP:
        return _CRYPTO_SYMBOL_MAP[ticker]
    if ticker.endswith("-USD"):
        return ticker.replace("-USD", "/USD")
    return ticker

def get_alpaca_client():
    key    = CONFIG.get("ALPACA_API_KEY", "")
    secret = CONFIG.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        return None
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(key, secret, paper=True)
    except Exception as e:
        print(f"  ⚠ Alpaca client error: {e}")
        return None

def get_alpaca_account_state(client) -> dict:
    try:
        acct       = client.get_account()
        equity     = float(acct.equity)
        last_eq    = float(acct.last_equity)
        daily_pnl  = (equity - last_eq) / last_eq * 100 if last_eq else 0
        positions  = {p.symbol for p in client.get_all_positions()}
        return {"equity": equity, "daily_pnl": daily_pnl, "open_symbols": positions}
    except Exception as e:
        print(f"  ⚠ Alpaca account fetch error: {e}")
        return {"equity": 0, "daily_pnl": 0, "open_symbols": set()}

def place_alpaca_trade(client, signal: dict) -> bool:
    try:
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        ticker    = signal["ticker"]
        is_crypto = ticker.endswith("-USD")
        direction = signal["direction"]
        symbol    = alpaca_symbol(ticker)

        # Alpaca spot doesn't support crypto shorts
        if is_crypto and direction == "SHORT":
            print(f"  ⚠ Alpaca: skipping crypto short ({symbol}) — spot only")
            return False

        side             = OrderSide.BUY if direction == "LONG" else OrderSide.SELL
        entry            = float(signal["entry"])
        stop_price       = round(float(signal["stop_loss"]), 6)
        take_profit_price = round(float(signal["target1"]), 6)
        pos              = signal.get("position") or {}
        dollar_risk      = float(pos.get("dollar_risk") or 0)

        # Calculate unlevered qty from dollar risk and per-unit risk
        risk_per = abs(entry - stop_price)
        if risk_per <= 0 or dollar_risk <= 0:
            print(f"  ⚠ Alpaca: invalid sizing for {symbol}")
            return False
        qty = round(dollar_risk / risk_per, 6 if is_crypto else 2)
        if qty <= 0:
            print(f"  ⚠ Alpaca: qty is 0 for {symbol}")
            return False

        req = MarketOrderRequest(
            symbol         = symbol,
            qty            = qty,
            side           = side,
            time_in_force  = TimeInForce.GTC,
            order_class    = "bracket",
            take_profit    = TakeProfitRequest(limit_price=take_profit_price),
            stop_loss      = StopLossRequest(stop_price=stop_price),
        )
        order = client.submit_order(req)
        print(f"  ✅ Alpaca: {symbol} {direction} {qty} | TP:{take_profit_price} SL:{stop_price} | #{order.id}")

        # Notify via Telegram
        send_telegram(
            f"📈 <b>PAPER TRADE PLACED</b>\n"
            f"<b>{symbol}</b>  {'▲' if direction=='LONG' else '▼'}{direction}\n"
            f"Qty: {qty}  |  Entry: ~{entry}\n"
            f"Stop: {stop_price}  |  T1: {take_profit_price}\n"
            f"Risk: ${dollar_risk:.0f}  |  Score: {signal['score']}"
        )
        return True

    except Exception as e:
        print(f"  ⚠ Alpaca order failed: {e}")
        return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN SCANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_scanner(send_alerts=True):
    print("\n" + "━"*60)
    print("  TradeSignal Scanner v5.0")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stocks min:{CONFIG['MIN_SCORE_STOCK']} | Crypto min:{CONFIG['MIN_SCORE_CRYPTO']} | High-lev min:{CONFIG['MIN_SCORE_HIGH_LEV']} | HC min:{CONFIG['MIN_SCORE_HIGH_CONVICTION']}")
    print("━"*60)

    # Context
    session  = get_session_info()
    events   = check_high_impact_events()
    btc_trend = get_btc_trend()

    print(f"\n⏰ Denver time: {session['time_denver']} | Session: {session['session']}")
    print(f"₿  BTC trend: {btc_trend.upper()}")
    if events["has_event"]:
        for flag in events["flags"]:
            print(f"  {flag}")

    print("\n📡 Fetching sentiment...")
    sentiment = get_sentiment()
    print(f"  {sentiment['icon']} {sentiment['bias']}  F&G:{sentiment['fear_greed_score']}  VIX:{sentiment['vix']}")

    total = len(ALL_TICKERS)
    print(f"\n🔍 Scanning {total} assets...\n")

    results, vol_spikes, ema_crosses = [], [], []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{i:02d}/{total}] {ticker:<18}", end=" ", flush=True)
        r = analyze_ticker(ticker, btc_trend, session)
        if r:
            results.append(r)
            gl_icon  = "🟢" if r["green_lights"]["signal"] == "bullish" else "🟣"
            flag     = "🚨" if r["vol_spike"] else "⚡" if r["ema_cross"] else "🔀" if r["rsi_divergence"]["bullish"] or r["rsi_divergence"]["bearish"] else "💥" if r["bb_squeeze"] else "📈"
            tf_tag   = "15m" if r["timeframe"] == "15m" else " 1d"
            arrow    = "▲" if r["direction"] == "LONG" else "▼"
            pen_str  = f" [{','.join(r['penalty_notes'])}]" if r["penalty_notes"] else ""
            print(f"{flag}[{tf_tag}] {arrow}{r['direction']:<5} S:{r['score']} R/R:{r['rr_ratio']} GL:{gl_icon}{pen_str}")
            if r["vol_spike"]:  vol_spikes.append(r)
            if r["ema_cross"]:  ema_crosses.append(r)
        else:
            print("–")
        time.sleep(0.25)

    results.sort(key=lambda x: (x["score"], x["rr_ratio"]), reverse=True)
    scalps     = [r for r in results if r["timeframe"] == "15m"][:CONFIG["TOP_N"]]
    swings     = [r for r in results if r["timeframe"] == "1d"][:CONFIG["TOP_N"]]
    top        = results[:CONFIG["TOP_N"]]
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'━'*60}")
    print(f"  ⚡ TOP {len(scalps)} SCALP SETUPS")
    print(f"{'━'*60}")
    if scalps:
        for r in scalps: print_setup(r)
    else:
        print("  No scalp setups this scan.")

    print(f"\n{'━'*60}")
    print(f"  📅 TOP {len(swings)} SWING SETUPS")
    print(f"{'━'*60}")
    if swings:
        for r in swings: print_setup(r)
    else:
        print("  No swing setups this scan.")

    # High Conviction Telegram alerts
    hc_signals = [r for r in results if r.get("high_conviction")]
    if send_alerts and hc_signals:
        print(f"\n  🔥 {len(hc_signals)} HIGH CONVICTION signal(s) — sending Telegram alerts...")
        for r in hc_signals:
            try:
                send_telegram(format_hc_telegram(r))
            except Exception as e:
                print(f"  ⚠ HC alert format error: {e}")
            save_hc_alert(r, output_dir)

    # Alpaca paper trading
    alpaca = get_alpaca_client()
    if alpaca and hc_signals:
        state = get_alpaca_account_state(alpaca)
        print(f"\n  💰 Alpaca paper | Equity:${state['equity']:,.2f} | Day P&L:{state['daily_pnl']:+.2f}%")
        if state["daily_pnl"] <= -CONFIG["ALPACA_DAILY_LOSS_LIMIT"]:
            print(f"  🛑 Daily loss limit hit ({state['daily_pnl']:.2f}%) — no new trades")
            send_telegram(f"🛑 <b>Daily loss limit hit</b> ({state['daily_pnl']:.2f}%) — trading paused for today")
        elif len(state["open_symbols"]) >= CONFIG["ALPACA_MAX_POSITIONS"]:
            print(f"  ⚠ Max positions ({CONFIG['ALPACA_MAX_POSITIONS']}) reached — skipping")
        else:
            slots = CONFIG["ALPACA_MAX_POSITIONS"] - len(state["open_symbols"])
            placed = 0
            for r in hc_signals:
                if placed >= slots:
                    break
                sym = alpaca_symbol(r["ticker"])
                if sym in state["open_symbols"]:
                    print(f"  ⚠ Already in {sym} — skipping")
                    continue
                if place_alpaca_trade(alpaca, r):
                    state["open_symbols"].add(sym)
                    placed += 1

    # Email digest
    if send_alerts and (scalps or swings):
        body  = format_digest(scalps, swings, sentiment, session, events)
        subj  = f"TradeSignal v5.1 | {len(scalps)+len(swings)} signals | {session['session']} | {sentiment['icon']} {sentiment['bias']} | {session['time_denver']}"
        print(f"\n  📤 Sending digest...")
        send_email(subj, body)

    # Save JSON
    gc.collect()
    output_path = os.path.join(output_dir, "scan_results.json")

    output = {
        "generated_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scanner_version":  "5.0",
        "session":          session,
        "market_sentiment": sentiment,
        "btc_trend":        btc_trend,
        "economic_events":  events,
        "account_info": {
            "crypto": f"${CONFIG['CRYPTO_ACCOUNT']:,} @ {CONFIG['CRYPTO_LEVERAGE']}x",
            "stocks": f"${CONFIG['STOCK_ACCOUNT']:,} @ 1x",
            "risk_pct": f"{int(CONFIG['RISK_PCT']*100)}%",
        },
        "total_signals":    len(results),
        "high_conviction":  len(hc_signals),
        "vol_spikes":       len(vol_spikes),
        "ema_crossovers":   len(ema_crosses),
        "top_scalps":       scalps,
        "top_swings":       swings,
        "top_setups":       top,
        "all_signals":      results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_nan(output), f, indent=2, cls=SafeEncoder)

    save_history(results, output_dir)

    print(f"\n✅ {len(results)} signals | ⚡ Scalps:{len(scalps)}  📅 Swings:{len(swings)}  🔥 HC:{len(hc_signals)}")
    print(f"   🚨 Spikes:{len(vol_spikes)}  ⚡ Crosses:{len(ema_crosses)}")
    print(f"   📁 {output_path}")
    print("━"*60 + "\n")
    return output


if __name__ == "__main__":
    import sys
    if "--summary" in sys.argv:
        send_daily_hc_summary(os.path.dirname(os.path.abspath(__file__)))
    else:
        run_scanner(send_alerts=True)
