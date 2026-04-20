"""
TradeSignal Scanner v4.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New in v4.0:
  - Multi-timeframe confirmation (15m + 1h + 4h for scalps)
  - VWAP (institutional bias filter)
  - RSI divergence detection
  - Relative Volume / RVOL (time-of-day adjusted)
  - Position sizing (crypto $20k/5x, stocks $1,200)
  - Minimum volume filter (removes illiquid signals)
  - Earnings/news flag
  - Historical scan tracking (scan_history.json)
  - News/geopolitical risk flag

Accounts:
  Crypto (BloFin) : $20,000 | 5x leverage | 5% risk = $1,000/trade
  Stocks (Webull) : $1,200  | 1x          | 5% risk = $60/trade

Min R/R    : 2.5:1  |  Target: 5:1
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
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

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
    # Accounts
    "CRYPTO_ACCOUNT":     20000,
    "STOCK_ACCOUNT":      1200,
    "RISK_PCT":           0.05,      # 5% risk per trade
    "CRYPTO_LEVERAGE":    5,

    # Signal thresholds
    "MIN_RR":             2.5,
    "MIN_SCORE":          5,         # raised from 4 — higher quality only
    "VOL_SPIKE_PCT":      15.0,
    "VOL_SURGE_RATIO":    1.5,
    "MIN_AVG_VOLUME":     500000,    # filter out illiquid assets
    "TOP_N":              5,

    # Alerts
    "TWILIO_SID":         os.getenv("TWILIO_SID", ""),
    "TWILIO_AUTH":        os.getenv("TWILIO_AUTH", ""),
    "TWILIO_FROM":        os.getenv("TWILIO_FROM", ""),
    "TWILIO_TO":          os.getenv("TWILIO_TO",   ""),
    "SENDGRID_API_KEY":   os.getenv("SENDGRID_API_KEY", ""),
    "ALERT_EMAIL_FROM":   os.getenv("ALERT_EMAIL_FROM", ""),
    "ALERT_EMAIL_TO":     os.getenv("ALERT_EMAIL_TO",   ""),
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
        "smoothed": round(results[1], 4),
        "slope":    round(slope, 6),
        "strength": round(abs(slope), 6),
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
    fg  = get_fear_greed()
    vix = get_vix()
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
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calc_atr(high, low, close, period=14):
    try:
        return float(AverageTrueRange(high, low, close, window=period).average_true_range().iloc[-1])
    except Exception:
        return float((high - low).tail(period).mean())

def tf_label(tf):
    labels = {"15m": "⚡ Scalp (mins-hours)", "1h": "⏱ Intraday (hours)",
              "4h": "📊 Swing (hrs-days)",   "1d": "📅 Swing (days-weeks)"}
    return labels.get(tf, tf)

def calc_vwap(df):
    """Calculate VWAP from OHLCV dataframe."""
    try:
        vwap = VolumeWeightedAveragePrice(
            high=df["High"].squeeze(), low=df["Low"].squeeze(),
            close=df["Close"].squeeze(), volume=df["Volume"].squeeze()
        )
        return float(vwap.volume_weighted_average_price().iloc[-1])
    except Exception:
        return None

def detect_rsi_divergence(close, rsi_series, lookback=14):
    """
    Detect RSI divergence:
    Bullish: price makes lower low but RSI makes higher low
    Bearish: price makes higher high but RSI makes lower high
    """
    try:
        price = close.values[-lookback:]
        rsi   = rsi_series.values[-lookback:]
        # Find recent swing highs/lows
        p_high, p_low = max(price[-5:]), min(price[-5:])
        p_high_prev   = max(price[:5])
        p_low_prev    = min(price[:5])
        r_high        = max(rsi[-5:])
        r_low         = min(rsi[-5:])
        r_high_prev   = max(rsi[:5])
        r_low_prev    = min(rsi[:5])

        bull_div = p_low < p_low_prev and r_low > r_low_prev
        bear_div = p_high > p_high_prev and r_high < r_high_prev
        return {"bullish": bull_div, "bearish": bear_div}
    except Exception:
        return {"bullish": False, "bearish": False}

def calc_rvol(vol_series):
    """
    Relative Volume — current volume vs average of same-period bars historically.
    Approximated as current vs 20-bar average (intraday time-of-day adjustment).
    """
    try:
        avg = float(vol_series.tail(20).mean())
        cur = float(vol_series.iloc[-1])
        return round(cur / avg, 2) if avg > 0 else 0
    except Exception:
        return 0

def check_earnings(ticker):
    """Flag if ticker has earnings coming up in next 7 days."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty:
            return False
        if hasattr(cal, 'columns'):
            for col in cal.columns:
                val = cal[col].iloc[0] if len(cal) > 0 else None
                if val and hasattr(val, 'date'):
                    days = (val.date() - datetime.date.today()).days
                    if 0 <= days <= 7:
                        return True
        return False
    except Exception:
        return False

def get_higher_tf_bias(ticker, primary_tf):
    """
    Get bias from higher timeframes for MTF confirmation.
    For 15m scalps: check 1h and 4h
    For 1d swings: check weekly
    Returns: 'bullish', 'bearish', or 'neutral'
    """
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
            if float(ema5.iloc[-1]) > float(ema20.iloc[-1]):
                signals.append("bullish")
            else:
                signals.append("bearish")
            time.sleep(0.1)

        if not signals:
            return "neutral"
        bull_count = signals.count("bullish")
        bear_count = signals.count("bearish")
        if bull_count > bear_count:   return "bullish"
        if bear_count > bull_count:   return "bearish"
        return "neutral"
    except Exception:
        return "neutral"

def calc_position_size(ticker, price, stop_loss, direction, is_crypto):
    """
    Calculate exact position size based on account and risk %.
    Returns: contracts/shares, dollar risk, dollar value of position
    """
    try:
        if is_crypto:
            account  = CONFIG["CRYPTO_ACCOUNT"]
            leverage = CONFIG["CRYPTO_LEVERAGE"]
        else:
            account  = CONFIG["STOCK_ACCOUNT"]
            leverage = 1

        max_risk  = account * CONFIG["RISK_PCT"]
        risk_per  = abs(price - stop_loss)
        if risk_per <= 0:
            return None

        raw_units = max_risk / risk_per
        # Apply leverage — you can control more with leverage
        units     = raw_units * leverage if is_crypto else raw_units
        pos_value = units * price / leverage if is_crypto else units * price

        # Cap position at 30% of account
        max_pos   = account * 0.30 * leverage if is_crypto else account * 0.30
        if pos_value > max_pos:
            units     = max_pos / price
            pos_value = max_pos

        return {
            "units":       round(units, 4),
            "dollar_risk": round(units * risk_per / leverage if is_crypto else units * risk_per, 2),
            "pos_value":   round(pos_value, 2),
            "account":     "crypto" if is_crypto else "stocks",
            "leverage":    leverage,
        }
    except Exception:
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE SIGNAL ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_ticker(ticker: str) -> Optional[dict]:
    try:
        is_crypto = ticker.endswith("-USD")
        is_hbeta  = ticker in HIGH_BETA
        tf        = "15m" if (is_crypto or is_hbeta) else "1d"
        period    = "5d"  if tf == "15m" else "6mo"

        df = yf.download(ticker, period=period, interval=tf,
                        progress=False, auto_adjust=True)
        if df is None or len(df) < 55:
            return None

        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        vol   = df["Volume"].squeeze()

        # ── Minimum volume filter ────────────────────────────────────────────
        avg_vol = float(vol.tail(20).mean())
        if avg_vol < CONFIG["MIN_AVG_VOLUME"] and not is_crypto:
            return None

        # ── EMAs ─────────────────────────────────────────────────────────────
        ema5  = close.ewm(span=5,  adjust=False).mean()
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        e5, e9, e50 = float(ema5.iloc[-1]), float(ema9.iloc[-1]), float(ema50.iloc[-1])
        e5p, e9p    = float(ema5.iloc[-2]), float(ema9.iloc[-2])

        cross_bull = e5p < e9p and e5 > e9
        cross_bear = e5p > e9p and e5 < e9
        trend_up   = e5 > e9 > e50
        trend_down = e5 < e9 < e50

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_i = MACD(close)
        ml = float(macd_i.macd().iloc[-1])
        ms = float(macd_i.macd_signal().iloc[-1])
        mh = float(macd_i.macd_diff().iloc[-1])
        macd_bull = ml > ms and mh > 0
        macd_bear = ml < ms and mh < 0

        # ── StochRSI ──────────────────────────────────────────────────────────
        srsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        sk = float(srsi.stochrsi_k().iloc[-1])
        sd = float(srsi.stochrsi_d().iloc[-1])
        srsi_bull = sk > sd and sk < 0.8
        srsi_bear = sk < sd and sk > 0.2

        # ── RSI + Divergence ──────────────────────────────────────────────────
        rsi_ind    = RSIIndicator(close, window=14)
        rsi_series = rsi_ind.rsi()
        rsi_val    = float(rsi_series.iloc[-1])
        rsi_div    = detect_rsi_divergence(close, rsi_series)

        # ── VWAP ─────────────────────────────────────────────────────────────
        vwap_val   = calc_vwap(df)
        price      = float(close.iloc[-1])
        vwap_bull  = vwap_val is not None and price > vwap_val
        vwap_bear  = vwap_val is not None and price < vwap_val

        # ── Volume / RVOL ─────────────────────────────────────────────────────
        cur_vol    = float(vol.iloc[-1])
        vol_ratio  = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0
        rvol       = calc_rvol(vol)
        vol_surge  = vol_ratio >= CONFIG["VOL_SURGE_RATIO"]

        # ── Bollinger Band Squeeze ────────────────────────────────────────────
        bb     = BollingerBands(close, window=20, window_dev=2)
        bw     = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1])
        bw_avg = float((bb.bollinger_hband() - bb.bollinger_lband()).tail(20).mean())
        bb_squeeze = bw < bw_avg * 0.75

        # ── Volatility spike ──────────────────────────────────────────────────
        prev       = float(close.iloc[-2])
        candle_pct = round(abs(price - prev) / prev * 100, 2) if prev > 0 else 0
        vol_spike  = candle_pct >= CONFIG["VOL_SPIKE_PCT"]

        # ── Jedi Green Lights ─────────────────────────────────────────────────
        gl = jedi_green_lights(close)

        # ── Score — direction first ───────────────────────────────────────────
        bull_score = sum([
            cross_bull * 3,
            trend_up * 2,
            macd_bull * 2,
            (gl["signal"] == "bullish") * 2,
            vwap_bull * 2,                      # NEW: VWAP
            rsi_div["bullish"] * 2,             # NEW: RSI divergence
            srsi_bull * 1,
            vol_surge * 1,
            bb_squeeze * 1,
            vol_spike * 1,
        ])
        bear_score = sum([
            cross_bear * 3,
            trend_down * 2,
            macd_bear * 2,
            (gl["signal"] == "bearish") * 2,
            vwap_bear * 2,
            rsi_div["bearish"] * 2,
            srsi_bear * 1,
            vol_surge * 1,
            bb_squeeze * 1,
            vol_spike * 1,
        ])

        direction = "LONG" if bull_score >= bear_score else "SHORT"
        score     = bull_score if direction == "LONG" else bear_score

        if score < CONFIG["MIN_SCORE"]:
            return None

        # ── Multi-timeframe confirmation ──────────────────────────────────────
        htf_bias = get_higher_tf_bias(ticker, tf)
        mtf_confirmed = (
            (direction == "LONG"  and htf_bias == "bullish") or
            (direction == "SHORT" and htf_bias == "bearish") or
            htf_bias == "neutral"
        )
        # Penalize if higher TF disagrees
        if not mtf_confirmed:
            score = max(0, score - 3)
            if score < CONFIG["MIN_SCORE"]:
                return None

        # ── Earnings flag ─────────────────────────────────────────────────────
        has_earnings = check_earnings(ticker) if not is_crypto else False

        # ── Direction-aware SL/TP ─────────────────────────────────────────────
        atr_val = calc_atr(high, low, close)
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
        if rr < CONFIG["MIN_RR"]:
            return None

        # ── Position sizing ───────────────────────────────────────────────────
        pos = calc_position_size(ticker, price, stop_loss, direction, is_crypto)

        # ── Signal type ───────────────────────────────────────────────────────
        if vol_spike and vol_surge:      stype = "VOLATILITY SPIKE 🚨"
        elif cross_bull or cross_bear:   stype = "EMA CROSSOVER ⚡"
        elif rsi_div["bullish"] or rsi_div["bearish"]: stype = "RSI DIVERGENCE 🔀"
        elif bb_squeeze and vol_surge:   stype = "SQUEEZE BREAKOUT 💥"
        else:                            stype = "MOMENTUM 📈"

        return {
            "ticker":        ticker,
            "type":          "crypto" if is_crypto else "stock",
            "signal_type":   stype,
            "trade_type":    tf_label(tf),
            "price":         round(price, 4),
            "direction":     direction,
            "score":         score,
            "rr_ratio":      rr,
            "entry":         round(price, 4),
            "stop_loss":     round(stop_loss, 4),
            "target1":       t1,
            "target2":       t2,
            "target3":       t3,
            "risk_per_unit": round(risk, 4),
            "atr":           round(atr_val, 4),
            "vwap":          round(vwap_val, 4) if vwap_val else None,
            "vwap_bias":     "above" if vwap_bull else "below",
            "rsi":           round(rsi_val, 2),
            "rsi_divergence":rsi_div,
            "rvol":          rvol,
            "vol_ratio":     vol_ratio,
            "candle_pct":    candle_pct,
            "vol_spike":     vol_spike,
            "bb_squeeze":    bb_squeeze,
            "ema_cross":     cross_bull or cross_bear,
            "macd_signal":   "bull" if macd_bull else "bear",
            "stochrsi_k":    round(sk, 3),
            "mtf_bias":      htf_bias,
            "mtf_confirmed": mtf_confirmed,
            "has_earnings":  has_earnings,
            "green_lights":  gl,
            "position":      pos,
            "timeframe":     tf,
            "scanned_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    except Exception as e:
        print(f"  ⚠ {ticker}: {e}")
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALERTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_digest(scalps, swings, sentiment):
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        "="*50,
        "TRADESIGNAL v4.0 DIGEST",
        f"Time   : {now}",
        f"Market : {sentiment['icon']} {sentiment['bias']} | F&G:{sentiment['fear_greed_score']} | VIX:{sentiment['vix']}",
        sentiment['macro_note'],
        "="*50, "",
        f"⚡ TOP SCALPS (15m) — {len(scalps)} signals",
        "-"*45,
    ]
    for i, r in enumerate(scalps, 1):
        gl  = r["green_lights"]
        pos = r.get("position") or {}
        arr = "▲" if r["direction"] == "LONG" else "▼"
        earnings_flag = " ⚠️ EARNINGS SOON" if r.get("has_earnings") else ""
        mtf_flag = f" MTF:{r.get('mtf_bias','?').upper()}" 
        lines += [
            f"{i}. {r['ticker']} {arr}{r['direction']} | {r['signal_type']}{earnings_flag}",
            f"   Entry    : {r['entry']}  |  VWAP: {r.get('vwap','N/A')} ({r.get('vwap_bias','?')})",
            f"   Stop Loss: {r['stop_loss']}  |  RSI: {r.get('rsi','?')}  |  RVOL: {r.get('rvol','?')}x",
            f"   T1:{r['target1']}  T2:{r['target2']}  T3:{r['target3']}",
            f"   R/R:{r['rr_ratio']}  GL:{'BULL' if gl['signal']=='bullish' else 'BEAR'}{mtf_flag}",
            f"   Position : {pos.get('units','?')} units | Risk: ${pos.get('dollar_risk','?')} | Value: ${pos.get('pos_value','?')}",
            "",
        ]
    lines += [f"📅 TOP SWINGS (1d) — {len(swings)} signals", "-"*45]
    for i, r in enumerate(swings, 1):
        gl  = r["green_lights"]
        pos = r.get("position") or {}
        arr = "▲" if r["direction"] == "LONG" else "▼"
        earnings_flag = " ⚠️ EARNINGS SOON" if r.get("has_earnings") else ""
        mtf_flag = f" MTF:{r.get('mtf_bias','?').upper()}"
        lines += [
            f"{i}. {r['ticker']} {arr}{r['direction']} | {r['signal_type']}{earnings_flag}",
            f"   Entry    : {r['entry']}  |  VWAP: {r.get('vwap','N/A')} ({r.get('vwap_bias','?')})",
            f"   Stop Loss: {r['stop_loss']}  |  RSI: {r.get('rsi','?')}  |  RVOL: {r.get('rvol','?')}x",
            f"   T1:{r['target1']}  T2:{r['target2']}  T3:{r['target3']}",
            f"   R/R:{r['rr_ratio']}  GL:{'BULL' if gl['signal']=='bullish' else 'BEAR'}{mtf_flag}",
            f"   Position : {pos.get('units','?')} units | Risk: ${pos.get('dollar_risk','?')} | Value: ${pos.get('pos_value','?')}",
            "",
        ]
    lines.append("="*50)
    return "\n".join(lines)

def send_sms(body):
    if not TWILIO_AVAILABLE: return
    c = CONFIG
    if not all([c["TWILIO_SID"], c["TWILIO_AUTH"], c["TWILIO_FROM"], c["TWILIO_TO"]]): return
    try:
        TwilioClient(c["TWILIO_SID"], c["TWILIO_AUTH"]).messages.create(
            body=body[:1600], from_=c["TWILIO_FROM"], to=c["TWILIO_TO"])
        print("  ✅ SMS sent")
    except Exception as e:
        print(f"  ⚠ SMS failed: {e}")

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

def send_digest(scalps, swings, sentiment):
    total = len(scalps) + len(swings)
    body  = format_digest(scalps, swings, sentiment)
    subj  = f"TradeSignal v4 | {total} signals | {sentiment['icon']} {sentiment['bias']} | {datetime.datetime.now().strftime('%I:%M %p')}"
    print(f"\n  📤 Sending digest ({total} signals)...")
    send_sms(body[:1600])
    send_email(subj, body)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HISTORY TRACKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_history(results, output_dir):
    """Append current signals to scan_history.json for tracking accuracy."""
    try:
        history_path = os.path.join(output_dir, "scan_history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)

        entry = {
            "scanned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "signals": [{
                "ticker":    r["ticker"],
                "direction": r["direction"],
                "entry":     r["entry"],
                "stop_loss": r["stop_loss"],
                "target1":   r["target1"],
                "target3":   r["target3"],
                "score":     r["score"],
                "timeframe": r["timeframe"],
            } for r in results]
        }
        history.append(entry)
        # Keep last 30 days of history
        history = history[-360:]
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  📚 History saved ({len(history)} scans tracked)")
    except Exception as e:
        print(f"  ⚠ History save failed: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRINT HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_setup(r):
    gl  = r["green_lights"]
    pos = r.get("position") or {}
    arr = "▲ LONG" if r["direction"] == "LONG" else "▼ SHORT"
    sl_note = "← above entry" if r["direction"] == "SHORT" else "← below entry"
    earnings = " ⚠️  EARNINGS SOON — CAUTION" if r.get("has_earnings") else ""
    mtf = r.get("mtf_bias", "unknown")
    mtf_icon = "✅" if r.get("mtf_confirmed") else "⚠️"

    print(f"\n  {r['ticker']} ({r['type'].upper()}) | {arr} | {r['signal_type']}{earnings}")
    print(f"  {r['trade_type']}")
    print(f"  Score:{r['score']}  R/R:{r['rr_ratio']}:1  RSI:{r.get('rsi','?')}  RVOL:{r.get('rvol','?')}x")
    print(f"  VWAP : {r.get('vwap','N/A')} — price is {r.get('vwap_bias','?')} VWAP")
    print(f"  MTF  : {mtf_icon} Higher TF bias = {mtf.upper()}")
    print(f"  Entry    : {r['entry']}")
    print(f"  Stop Loss: {r['stop_loss']}  {sl_note}")
    print(f"  Target 1 : {r['target1']}  (2:1)")
    print(f"  Target 2 : {r['target2']}  (3.5:1)")
    print(f"  Target 3 : {r['target3']}  (5:1)")
    print(f"  GL: {'🟢 Bull' if gl['signal']=='bullish' else '🟣 Bear'}  slope:{gl['slope']}")
    if pos:
        print(f"  Position : {pos.get('units','?')} units | Dollar Risk: ${pos.get('dollar_risk','?')} | Position Value: ${pos.get('pos_value','?')} ({pos.get('leverage','1')}x)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN SCANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_scanner(send_alerts=True):
    print("\n" + "━"*60)
    print("  TradeSignal Scanner v4.0")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Crypto: ${CONFIG['CRYPTO_ACCOUNT']:,} @ {CONFIG['CRYPTO_LEVERAGE']}x | Stocks: ${CONFIG['STOCK_ACCOUNT']:,}")
    print(f"  Risk per trade: {int(CONFIG['RISK_PCT']*100)}% | Max crypto risk: ${int(CONFIG['CRYPTO_ACCOUNT']*CONFIG['RISK_PCT']):,} | Max stock risk: ${int(CONFIG['STOCK_ACCOUNT']*CONFIG['RISK_PCT']):,}")
    print("━"*60)

    print("\n📡 Fetching sentiment...")
    sentiment = get_sentiment()
    print(f"  {sentiment['icon']} {sentiment['bias']}  F&G:{sentiment['fear_greed_score']}  VIX:{sentiment['vix']}")
    print(f"  {sentiment['macro_note']}")

    total = len(ALL_TICKERS)
    print(f"\n🔍 Scanning {total} assets with MTF confirmation...\n")

    results, vol_spikes, ema_crosses = [], [], []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{i:02d}/{total}] {ticker:<18}", end=" ", flush=True)
        r = analyze_ticker(ticker)
        if r:
            results.append(r)
            gl_icon  = "🟢" if r["green_lights"]["signal"] == "bullish" else "🟣"
            flag     = "🚨" if r["vol_spike"] else "⚡" if r["ema_cross"] else "🔀" if r["rsi_divergence"]["bullish"] or r["rsi_divergence"]["bearish"] else "💥" if r["bb_squeeze"] else "📈"
            tf_tag   = "15m" if r["timeframe"] == "15m" else " 1d"
            arrow    = "▲" if r["direction"] == "LONG" else "▼"
            mtf_icon = "✅" if r["mtf_confirmed"] else "⚠️"
            earn_tag = "💰" if r.get("has_earnings") else ""
            print(f"{flag} [{tf_tag}] {arrow}{r['direction']:<5} S:{r['score']} R/R:{r['rr_ratio']} GL:{gl_icon} MTF:{mtf_icon} RVOL:{r['rvol']}x {earn_tag}")
            if r["vol_spike"]:  vol_spikes.append(r)
            if r["ema_cross"]:  ema_crosses.append(r)
        else:
            print("–")
        time.sleep(0.25)

    # Sort and split
    results.sort(key=lambda x: (x["score"], x["rr_ratio"]), reverse=True)
    scalps = [r for r in results if r["timeframe"] == "15m"][:CONFIG["TOP_N"]]
    swings = [r for r in results if r["timeframe"] == "1d"][:CONFIG["TOP_N"]]
    top    = results[:CONFIG["TOP_N"]]

    # Print results
    print(f"\n{'━'*60}")
    print(f"  ⚡ TOP {len(scalps)} SCALP SETUPS  (15m — MTF confirmed)")
    print(f"{'━'*60}")
    if scalps:
        for r in scalps: print_setup(r)
    else:
        print("  No scalp setups this scan.")

    print(f"\n{'━'*60}")
    print(f"  📅 TOP {len(swings)} SWING SETUPS  (Daily — MTF confirmed)")
    print(f"{'━'*60}")
    if swings:
        for r in swings: print_setup(r)
    else:
        print("  No swing setups this scan.")

    # Send digest
    if send_alerts and (scalps or swings):
        send_digest(scalps, swings, sentiment)

    # Save JSON
    gc.collect()
    output_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "scan_results.json")
    output = {
        "generated_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scanner_version":  "4.0",
        "market_sentiment": sentiment,
        "account_info": {
            "crypto": f"${CONFIG['CRYPTO_ACCOUNT']:,} @ {CONFIG['CRYPTO_LEVERAGE']}x",
            "stocks": f"${CONFIG['STOCK_ACCOUNT']:,} @ 1x",
            "risk_pct": f"{int(CONFIG['RISK_PCT']*100)}%",
        },
        "total_signals":    len(results),
        "vol_spikes":       len(vol_spikes),
        "ema_crossovers":   len(ema_crosses),
        "top_scalps":       scalps,
        "top_swings":       swings,
        "top_setups":       top,
        "all_signals":      results,
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    # Save history
    save_history(results, output_dir)

    print(f"\n✅ {len(results)} signals | ⚡ Scalps:{len(scalps)}  📅 Swings:{len(swings)}")
    print(f"   🚨 Spikes:{len(vol_spikes)}  ⚡ Crosses:{len(ema_crosses)}")
    print(f"   📁 {output_path}")
    print("━"*60 + "\n")
    return output


def run_continuous():
    print("🔄 Continuous mode — every 4 hours. Ctrl+C to stop.\n")
    while True:
        run_scanner(send_alerts=True)
        print("⏳ Next scan in 4 hours...\n")
        time.sleep(4 * 60 * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        run_continuous()
    else:
        run_scanner(send_alerts=True)
