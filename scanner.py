"""
TradeSignal Scanner v3.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Indicators : EMA 5/9/50, MACD, StochRSI, Volume Surge,
             Bollinger Band Squeeze, Jedi Green Lights
Sentiment  : VIX + CNN Fear & Greed
Alerts     : Email (SendGrid) + SMS (Twilio)
Outputs    : Top 5 Scalps (15m) + Top 5 Swings (1d)
Min R/R    : 2.5:1  |  Target R/R: 5:1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import gc
import json
import time
import math
import datetime
import urllib.request
from typing import Optional

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    raise ImportError(f"Run: pip3 install numpy pandas yfinance ta") from e

try:
    from ta.momentum import StochRSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands, AverageTrueRange
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG = {
    "TWILIO_SID":         os.getenv("TWILIO_SID", ""),
    "TWILIO_AUTH":        os.getenv("TWILIO_AUTH", ""),
    "TWILIO_FROM":        os.getenv("TWILIO_FROM", ""),
    "TWILIO_TO":          os.getenv("TWILIO_TO",   ""),
    "SENDGRID_API_KEY":   os.getenv("SENDGRID_API_KEY", ""),
    "ALERT_EMAIL_FROM":   os.getenv("ALERT_EMAIL_FROM", ""),
    "ALERT_EMAIL_TO":     os.getenv("ALERT_EMAIL_TO",   ""),
    "MIN_RR":             2.5,
    "MIN_SCORE":          4,
    "VOL_SPIKE_PCT":      15.0,
    "VOL_SURGE_RATIO":    1.5,
    "TOP_N":              5,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WATCHLISTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# High beta tickers that get 15m timeframe
HIGH_BETA = {"TSLA","NVDA","AMD","COIN","MSTR","PLTR","SMCI","SOXL","TQQQ"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JEDI GREEN LIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MARKET SENTIMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        if hist.empty:
            return {"value": None, "level": "unknown"}
        val = round(float(hist["Close"].iloc[-1]), 2)
        level = "low" if val < 15 else "moderate" if val < 20 else "elevated" if val < 30 else "extreme"
        return {"value": val, "level": level}
    except Exception:
        return {"value": None, "level": "unknown"}

def get_sentiment():
    fg  = get_fear_greed()
    vix = get_vix()
    s   = fg["score"]
    v   = vix["value"]
    if s >= 60 and (v is None or v < 20):
        bias, icon = "RISK-ON",  "🟢"
    elif s <= 40 or (v is not None and v >= 25):
        bias, icon = "RISK-OFF", "🔴"
    else:
        bias, icon = "NEUTRAL",  "🟡"
    return {
        "bias": bias, "icon": icon,
        "fear_greed_score": s, "fear_greed_label": fg["label"],
        "vix": v, "vix_level": vix["level"],
        "macro_note": "⚠️ US-Iran tensions active. Oil above $100. Check news before entering.",
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calc_atr(high, low, close, period=14):
    try:
        return float(AverageTrueRange(high, low, close, window=period).average_true_range().iloc[-1])
    except Exception:
        return float((high - low).tail(period).mean())

def tf_label(tf):
    return "⚡ Scalp (mins-hours)" if tf == "15m" else "📅 Swing (days-weeks)"

def analyze_ticker(ticker: str) -> Optional[dict]:
    try:
        is_crypto = ticker.endswith("-USD")
        is_hbeta  = ticker in HIGH_BETA
        tf     = "15m" if (is_crypto or is_hbeta) else "1d"
        period = "5d"  if tf == "15m" else "6mo"

        df = yf.download(ticker, period=period, interval=tf, progress=False, auto_adjust=True)
        if df is None or len(df) < 55:
            return None

        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        vol   = df["Volume"].squeeze()

        # EMAs
        ema5  = close.ewm(span=5,  adjust=False).mean()
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        e5, e9, e50 = float(ema5.iloc[-1]), float(ema9.iloc[-1]), float(ema50.iloc[-1])
        e5p, e9p    = float(ema5.iloc[-2]), float(ema9.iloc[-2])

        cross_bull = e5p < e9p and e5 > e9
        cross_bear = e5p > e9p and e5 < e9
        trend_up   = e5 > e9 > e50
        trend_down = e5 < e9 < e50

        # MACD
        macd_i = MACD(close)
        ml = float(macd_i.macd().iloc[-1])
        ms = float(macd_i.macd_signal().iloc[-1])
        mh = float(macd_i.macd_diff().iloc[-1])
        macd_bull = ml > ms and mh > 0
        macd_bear = ml < ms and mh < 0

        # StochRSI
        srsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        sk = float(srsi.stochrsi_k().iloc[-1])
        sd = float(srsi.stochrsi_d().iloc[-1])
        srsi_bull = sk > sd and sk < 0.8
        srsi_bear = sk < sd and sk > 0.2

        # Volume
        avg_vol   = float(vol.tail(20).mean())
        cur_vol   = float(vol.iloc[-1])
        vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0
        vol_surge = vol_ratio >= CONFIG["VOL_SURGE_RATIO"]

        # Bollinger Band Squeeze
        bb     = BollingerBands(close, window=20, window_dev=2)
        bw     = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1])
        bw_avg = float((bb.bollinger_hband() - bb.bollinger_lband()).tail(20).mean())
        bb_squeeze = bw < bw_avg * 0.75

        # Volatility spike
        price      = float(close.iloc[-1])
        prev       = float(close.iloc[-2])
        candle_pct = round(abs(price - prev) / prev * 100, 2) if prev > 0 else 0
        vol_spike  = candle_pct >= CONFIG["VOL_SPIKE_PCT"]

        # Jedi Green Lights
        gl = jedi_green_lights(close)

        # ATR / R/R
        atr_val   = calc_atr(high, low, close)
        stop_loss = max(e9 * 0.995, price - 1.5 * atr_val)
        risk      = price - stop_loss
        if risk <= 0:
            return None
        t1  = round(price + risk * 2.0, 4)
        t2  = round(price + risk * 3.5, 4)
        t3  = round(price + risk * 5.0, 4)
        rr  = round((t3 - price) / risk, 2)

        # Score
        bull_score = sum([
            cross_bull * 3, trend_up * 2, macd_bull * 2, srsi_bull * 1,
            vol_surge * 1, bb_squeeze * 1, (gl["signal"] == "bullish") * 2, vol_spike * 1,
        ])
        bear_score = sum([
            cross_bear * 3, trend_down * 2, macd_bear * 2, srsi_bear * 1,
            vol_surge * 1, bb_squeeze * 1, (gl["signal"] == "bearish") * 2, vol_spike * 1,
        ])

        direction = "LONG" if bull_score >= bear_score else "SHORT"
        score     = bull_score if direction == "LONG" else bear_score

        if score < CONFIG["MIN_SCORE"] or rr < CONFIG["MIN_RR"]:
            return None

        if vol_spike and vol_surge:    stype = "VOLATILITY SPIKE 🚨"
        elif cross_bull or cross_bear: stype = "EMA CROSSOVER ⚡"
        elif bb_squeeze and vol_surge: stype = "SQUEEZE BREAKOUT 💥"
        else:                          stype = "MOMENTUM 📈"

        return {
            "ticker":       ticker,
            "type":         "crypto" if is_crypto else "stock",
            "signal_type":  stype,
            "trade_type":   tf_label(tf),
            "price":        round(price, 4),
            "direction":    direction,
            "score":        score,
            "rr_ratio":     rr,
            "entry":        round(price, 4),
            "stop_loss":    round(stop_loss, 4),
            "target1":      t1,
            "target2":      t2,
            "target3":      t3,
            "risk_per_unit":round(risk, 4),
            "atr":          round(atr_val, 4),
            "vol_ratio":    vol_ratio,
            "candle_pct":   candle_pct,
            "vol_spike":    vol_spike,
            "bb_squeeze":   bb_squeeze,
            "ema_cross":    cross_bull or cross_bear,
            "macd_signal":  "bull" if macd_bull else "bear",
            "stochrsi_k":   round(sk, 3),
            "green_lights": gl,
            "timeframe":    tf,
            "scanned_at":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    except Exception as e:
        print(f"  ⚠ {ticker}: {e}")
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALERTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_alert(r, sentiment):
    gl = r["green_lights"]
    return f"""
{'='*45}
TRADESIGNAL ALERT
{'='*45}
{r['signal_type']}
{r['trade_type']}
Ticker    : {r['ticker']} ({r['type'].upper()})
Direction : {r['direction']}
Score     : {r['score']}  R/R: {r['rr_ratio']}:1

ENTRY     : {r['entry']}
STOP LOSS : {r['stop_loss']}
TARGET 1  : {r['target1']}  (2:1)
TARGET 2  : {r['target2']}  (3.5:1)
TARGET 3  : {r['target3']}  (5:1)

Green Lights : {'BULL' if gl['signal']=='bullish' else 'BEAR'}  slope={gl['slope']}
Vol Ratio    : {r['vol_ratio']}x  |  Candle: {r['candle_pct']}%
BB Squeeze   : {'Yes' if r['bb_squeeze'] else 'No'}

Market: {sentiment['icon']} {sentiment['bias']}
F&G: {sentiment['fear_greed_score']} ({sentiment['fear_greed_label']})  VIX: {sentiment['vix']}
{sentiment['macro_note']}
{'='*45}"""

def format_digest(scalps, swings, sentiment):
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        "="*45,
        "TRADESIGNAL DIGEST",
        f"Time  : {now}",
        f"Market: {sentiment['icon']} {sentiment['bias']} | F&G:{sentiment['fear_greed_score']} | VIX:{sentiment['vix']}",
        sentiment['macro_note'],
        "="*45,
        "",
        f"⚡ TOP SCALPS (15m) — {len(scalps)} found",
        "-"*40,
    ]
    for i, r in enumerate(scalps, 1):
        gl = r["green_lights"]
        lines += [
            f"{i}. {r['ticker']} | {r['direction']} | {r['signal_type']}",
            f"   Entry:{r['entry']}  Stop:{r['stop_loss']}  T3:{r['target3']}  R/R:{r['rr_ratio']}",
            f"   GL:{'BULL' if gl['signal']=='bullish' else 'BEAR'}  Vol:{r['vol_ratio']}x  Score:{r['score']}",
            "",
        ]
    lines += [
        f"📅 TOP SWINGS (1d) — {len(swings)} found",
        "-"*40,
    ]
    for i, r in enumerate(swings, 1):
        gl = r["green_lights"]
        lines += [
            f"{i}. {r['ticker']} | {r['direction']} | {r['signal_type']}",
            f"   Entry:{r['entry']}  Stop:{r['stop_loss']}  T3:{r['target3']}  R/R:{r['rr_ratio']}",
            f"   GL:{'BULL' if gl['signal']=='bullish' else 'BEAR'}  Vol:{r['vol_ratio']}x  Score:{r['score']}",
            "",
        ]
    lines.append("="*45)
    return "\n".join(lines)

def send_sms(body):
    if not TWILIO_AVAILABLE:
        print("  ⚠ Run: pip3 install twilio"); return
    c = CONFIG
    if not all([c["TWILIO_SID"], c["TWILIO_AUTH"], c["TWILIO_FROM"], c["TWILIO_TO"]]):
        print("  ⚠ Twilio credentials missing"); return
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
        sg.send(Mail(
            from_email=c["ALERT_EMAIL_FROM"],
            to_emails=c["ALERT_EMAIL_TO"],
            subject=subject,
            plain_text_content=body,
        ))
        print("  ✅ Email sent")
    except Exception as e:
        print(f"  ⚠ Email failed: {e}")

def send_alert(r, sentiment):
    body = format_alert(r, sentiment)
    subj = f"ALERT: {r['signal_type']} | {r['ticker']} {r['direction']} | {r['trade_type']}"
    print(f"\n  Sending alert: {r['ticker']}...")
    send_sms(body[:1600])
    send_email(subj, body)

def send_digest(scalps, swings, sentiment):
    total = len(scalps) + len(swings)
    body  = format_digest(scalps, swings, sentiment)
    subj  = f"TradeSignal Digest | {total} signals | {sentiment['icon']} {sentiment['bias']}"
    print(f"\n  Sending digest ({total} signals)...")
    send_sms(body[:1600])
    send_email(subj, body)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRINT SETUP HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_setup(r):
    gl = r["green_lights"]
    print(f"\n  {r['ticker']} ({r['type'].upper()}) | {r['direction']} | {r['signal_type']}")
    print(f"  {r['trade_type']}")
    print(f"  Score:{r['score']}  R/R:{r['rr_ratio']}:1")
    print(f"  Entry    : {r['entry']}")
    print(f"  Stop Loss: {r['stop_loss']}")
    print(f"  Target 1 : {r['target1']}  (2:1)")
    print(f"  Target 2 : {r['target2']}  (3.5:1)")
    print(f"  Target 3 : {r['target3']}  (5:1)")
    print(f"  GL: {'🟢 Bull' if gl['signal']=='bullish' else '🟣 Bear'}  slope:{gl['slope']}  Vol:{r['vol_ratio']}x  Candle:{r['candle_pct']}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN SCANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_scanner(send_alerts=True):
    print("\n" + "━"*55)
    print("  TradeSignal Scanner v3.1")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("━"*55)

    print("\n📡 Fetching sentiment...")
    sentiment = get_sentiment()
    print(f"  {sentiment['icon']} {sentiment['bias']}  F&G:{sentiment['fear_greed_score']}  VIX:{sentiment['vix']}")
    print(f"  {sentiment['macro_note']}")

    total = len(ALL_TICKERS)
    print(f"\n🔍 Scanning {total} assets ({len(STOCKS)} stocks + {len(CRYPTO)} crypto)...\n")

    results, vol_spikes, ema_crosses = [], [], []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{i:02d}/{total}] {ticker:<18}", end=" ", flush=True)
        r = analyze_ticker(ticker)
        if r:
            results.append(r)
            gl_icon = "🟢" if r["green_lights"]["signal"] == "bullish" else "🟣"
            flag    = "🚨" if r["vol_spike"] else "⚡" if r["ema_cross"] else "💥" if r["bb_squeeze"] else "📈"
            tf_tag  = "15m" if r["timeframe"] == "15m" else " 1d"
            print(f"{flag} [{tf_tag}] {r['direction']:<5} Score:{r['score']} R/R:{r['rr_ratio']} GL:{gl_icon} Vol:{r['vol_ratio']}x")
            if send_alerts and (r["vol_spike"] or r["ema_cross"]):
                if r["vol_spike"]:  vol_spikes.append(r)
                if r["ema_cross"]:  ema_crosses.append(r)
                send_alert(r, sentiment)
        else:
            print("–")
        time.sleep(0.25)

    # Sort and split
    results.sort(key=lambda x: (x["score"], x["rr_ratio"]), reverse=True)
    scalps = [r for r in results if r["timeframe"] == "15m"][:CONFIG["TOP_N"]]
    swings = [r for r in results if r["timeframe"] == "1d"][:CONFIG["TOP_N"]]
    top    = results[:CONFIG["TOP_N"]]

    # Print scalps
    print(f"\n{'━'*55}")
    print(f"  ⚡ TOP {len(scalps)} SCALP SETUPS  (15 min — trade today)")
    print(f"{'━'*55}")
    if scalps:
        for r in scalps: print_setup(r)
    else:
        print("  No scalp setups this scan.")

    # Print swings
    print(f"\n{'━'*55}")
    print(f"  📅 TOP {len(swings)} SWING SETUPS  (Daily — hold days/weeks)")
    print(f"{'━'*55}")
    if swings:
        for r in swings: print_setup(r)
    else:
        print("  No swing setups this scan.")

    # Send digest
    if send_alerts and (scalps or swings):
        send_digest(scalps, swings, sentiment)

    # Save JSON
    gc.collect()
    output = {
        "generated_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "market_sentiment": sentiment,
        "total_signals":    len(results),
        "vol_spikes":       len(vol_spikes),
        "ema_crossovers":   len(ema_crosses),
        "top_scalps":       scalps,
        "top_swings":       swings,
        "top_setups":       top,
        "all_signals":      results,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ {len(results)} signals saved | ⚡ Scalps:{len(scalps)}  📅 Swings:{len(swings)}")
    print(f"   🚨 Vol spikes:{len(vol_spikes)}  EMA crosses:{len(ema_crosses)}")
    print(f"   📁 {output_path}")
    print("━"*55 + "\n")
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
