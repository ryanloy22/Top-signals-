"""
Crypto Signal Engine Backtest
Compares old logic (EMA 5/9, 15m) vs new logic (ADX gate, EMA 21/55, 1H)
over the past 7 days of real price data.
"""

import datetime, math, warnings, logging
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend   import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume     import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

# ── Config ────────────────────────────────────────────────────────────────────
MIN_SCORE = 5
MIN_RR    = 2.5
OUTCOME_WINDOW = 48   # candles forward to check outcome
VOL_SURGE_RATIO = 1.5

CRYPTO = [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD",
    "AVAX-USD","LINK-USD","DOT-USD","MATIC-USD","AAVE-USD",
    "UNI7083-USD","MKR-USD","INJ-USD","CRV-USD",
    "FET-USD","RNDR-USD","OCEAN-USD",
    "PEPE24478-USD","WIF-USD","BONK-USD",
    "TIA-USD","SEI-USD","ONDO-USD",
    "ARB11841-USD","OP-USD",
    "TAO-USD","SOL-USD","SUI20947-USD",
]
CRYPTO = list(dict.fromkeys(CRYPTO))  # deduplicate

# ── Helpers ───────────────────────────────────────────────────────────────────
def col(df, name):
    c = df[name]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.squeeze()

def check_outcome(df_full, signal_idx, entry, stop, t1, direction):
    """Look forward from signal_idx+1, return WIN/LOSS/ACTIVE."""
    end = min(signal_idx + 1 + OUTCOME_WINDOW, len(df_full))
    for i in range(signal_idx + 1, end):
        hi = float(df_full["High"].iloc[i]) if not isinstance(df_full["High"].iloc[i], pd.Series) else float(df_full["High"].iloc[i].iloc[0])
        lo = float(df_full["Low"].iloc[i])  if not isinstance(df_full["Low"].iloc[i],  pd.Series) else float(df_full["Low"].iloc[i].iloc[0])
        if direction == "LONG":
            if lo <= stop:  return "LOSS"
            if hi >= t1:    return "WIN"
        else:
            if hi >= stop:  return "LOSS"
            if lo <= t1:    return "WIN"
    return "ACTIVE"

def calc_atr(high, low, close, period=14):
    try:
        return float(AverageTrueRange(high, low, close, window=period).average_true_range().iloc[-1])
    except Exception:
        return float((high - low).tail(period).mean())

# ── OLD logic (EMA 5/9/50, no ADX, uses same 1H data for fair comparison) ─────
def old_signal(close, high, low, vol, price):
    try:
        ema5  = close.ewm(span=5,  adjust=False).mean()
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        e5,  e9,  e50 = float(ema5.iloc[-1]),  float(ema9.iloc[-1]),  float(ema50.iloc[-1])
        e5p, e9p      = float(ema5.iloc[-2]),  float(ema9.iloc[-2])

        cross_bull = e5p < e9p and e5 > e9
        cross_bear = e5p > e9p and e5 < e9
        trend_up   = e5 > e9 > e50
        trend_down = e5 < e9 < e50

        rsi_s   = RSIIndicator(close, window=14).rsi()
        rsi_val = float(rsi_s.iloc[-1])

        mi   = MACD(close)
        ml, ms, mh = float(mi.macd().iloc[-1]), float(mi.macd_signal().iloc[-1]), float(mi.macd_diff().iloc[-1])
        macd_bull = ml > ms and mh > 0
        macd_bear = ml < ms and mh < 0

        srsi = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        sk, sd = float(srsi.stochrsi_k().iloc[-1]), float(srsi.stochrsi_d().iloc[-1])
        srsi_bull = sk > sd and sk < 0.8
        srsi_bear = sk < sd and sk > 0.2

        avg_vol   = float(vol.tail(20).mean())
        cur_vol   = float(vol.iloc[-1])
        vol_surge = (cur_vol / avg_vol >= VOL_SURGE_RATIO) if avg_vol > 0 else False

        bb    = BollingerBands(close, window=20, window_dev=2)
        bw    = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1])
        bw_avg= float((bb.bollinger_hband() - bb.bollinger_lband()).tail(20).mean())
        bb_sq = bw < bw_avg * 0.75

        try:
            pr = close.values[-14:]
            rs = rsi_s.values[-14:]
            bull_div = min(pr[-5:]) < min(pr[:5]) and min(rs[-5:]) > min(rs[:5])
            bear_div = max(pr[-5:]) > max(pr[:5]) and max(rs[-5:]) < max(rs[:5])
        except Exception:
            bull_div = bear_div = False

        try:
            vwap_val  = float(VolumeWeightedAveragePrice(high=high,low=low,close=close,volume=vol).volume_weighted_average_price().iloc[-1])
            vwap_bull = price > vwap_val
            vwap_bear = price < vwap_val
        except Exception:
            vwap_bull = vwap_bear = False

        prev = float(close.iloc[-2])
        vol_spike = abs(price - prev) / prev * 100 >= 15.0 if prev > 0 else False

        bull_sc = sum([cross_bull*3, trend_up*2, macd_bull*2, vwap_bull*2, bull_div*2,
                       srsi_bull*1, vol_surge*1, bb_sq*1, vol_spike*1])
        bear_sc = sum([cross_bear*3, trend_down*2, macd_bear*2, vwap_bear*2, bear_div*2,
                       srsi_bear*1, vol_surge*1, bb_sq*1, vol_spike*1])

        direction = "LONG" if bull_sc >= bear_sc else "SHORT"
        score     = bull_sc if direction == "LONG" else bear_sc
        if score < MIN_SCORE:
            return None

        atr = calc_atr(high, low, close)
        sl  = max(e9 * 0.995, price - 1.5*atr) if direction == "LONG" else min(e9 * 1.005, price + 1.5*atr)
        risk = abs(price - sl)
        if risk <= 0: return None
        t1 = price + risk*2.0 if direction == "LONG" else price - risk*2.0
        t3 = price + risk*5.0 if direction == "LONG" else price - risk*5.0
        if abs(t3-price)/risk < MIN_RR: return None

        return {"direction": direction, "score": score, "entry": price, "stop": sl, "t1": t1}
    except Exception:
        return None

# ── NEW logic (ADX gate, EMA 21/55, OBV, RSI range) ──────────────────────────
def new_signal(close, high, low, vol, price):
    try:
        adx_val = float(ADXIndicator(high, low, close, window=14).adx().iloc[-1])
        if adx_val < 25:
            return None

        ema21 = close.ewm(span=21, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()
        e21, e55   = float(ema21.iloc[-1]), float(ema55.iloc[-1])
        e21p, e55p = float(ema21.iloc[-2]), float(ema55.iloc[-2])

        cross_bull = e21p < e55p and e21 > e55
        cross_bear = e21p > e55p and e21 < e55
        trend_up   = e21 > e55
        trend_down = e21 < e55

        rsi_val     = float(RSIIndicator(close, window=14).rsi().iloc[-1])
        rsi_long_ok  = 45 <= rsi_val <= 68
        rsi_short_ok = rsi_val <= 55

        mi   = MACD(close)
        ml, ms, mh = float(mi.macd().iloc[-1]), float(mi.macd_signal().iloc[-1]), float(mi.macd_diff().iloc[-1])
        macd_bull = ml > ms and mh > 0
        macd_bear = ml < ms and mh < 0

        obv      = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        obv_bull = float(obv.iloc[-1]) > float(obv.iloc[-3])
        obv_bear = float(obv.iloc[-1]) < float(obv.iloc[-3])

        avg_vol   = float(vol.tail(20).mean())
        cur_vol   = float(vol.iloc[-1])
        vol_surge = (cur_vol / avg_vol >= VOL_SURGE_RATIO) if avg_vol > 0 else False

        bb    = BollingerBands(close, window=20, window_dev=2)
        bw    = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1])
        bw_avg= float((bb.bollinger_hband() - bb.bollinger_lband()).tail(20).mean())
        bb_sq = bw < bw_avg * 0.75

        try:
            vwap_val  = float(VolumeWeightedAveragePrice(high=high,low=low,close=close,volume=vol).volume_weighted_average_price().iloc[-1])
            vwap_bull = price > vwap_val
            vwap_bear = price < vwap_val
        except Exception:
            vwap_bull = vwap_bear = False

        bull_sc = sum([cross_bull*3, trend_up*2, macd_bull*2, rsi_long_ok*2,
                       obv_bull*2, vwap_bull*1, vol_surge*1, bb_sq*1])
        bear_sc = sum([cross_bear*3, trend_down*2, macd_bear*2, rsi_short_ok*2,
                       obv_bear*2, vwap_bear*1, vol_surge*1, bb_sq*1])

        direction = "LONG" if bull_sc >= bear_sc else "SHORT"
        score     = bull_sc if direction == "LONG" else bear_sc
        if score < MIN_SCORE:
            return None

        atr = calc_atr(high, low, close)
        sl  = max(e55 * 0.99, price - 1.5*atr) if direction == "LONG" else min(e55 * 1.01, price + 1.5*atr)
        risk = abs(price - sl)
        if risk <= 0: return None
        t1 = price + risk*2.0 if direction == "LONG" else price - risk*2.0
        t3 = price + risk*5.0 if direction == "LONG" else price - risk*5.0
        if abs(t3-price)/risk < MIN_RR: return None

        return {"direction": direction, "score": score, "entry": price,
                "stop": sl, "t1": t1, "adx": round(adx_val,1)}
    except Exception:
        return None

# ── Main backtest ─────────────────────────────────────────────────────────────
def backtest_ticker(ticker):
    results = {"old": [], "new": []}
    try:
        # Download 1H data — 60 days for warmup + test window
        df_raw = yf.download(ticker, period="60d", interval="1h",
                             progress=False, auto_adjust=True)
        if df_raw is None or len(df_raw) < 100:
            return results

        # Flatten MultiIndex if present
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        df_raw.index = pd.to_datetime(df_raw.index, utc=True)

        # Test window: past 7 days — all timestamps kept tz-aware UTC
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
        test_indices = [i for i, ts in enumerate(df_raw.index)
                        if ts >= cutoff and i >= 55]

        if not test_indices:
            return results

        for i in test_indices:
            window = df_raw.iloc[:i+1]
            if len(window) < 55:
                continue

            close = window["Close"]
            high  = window["High"]
            low   = window["Low"]
            vol   = window["Volume"]
            price = float(close.iloc[-1])
            if price <= 0:
                continue

            ts = df_raw.index[i]

            # Old signal
            sig_old = old_signal(close, high, low, vol, price)
            if sig_old:
                outcome = check_outcome(df_raw, i, sig_old["entry"],
                                        sig_old["stop"], sig_old["t1"],
                                        sig_old["direction"])
                results["old"].append({
                    "ticker": ticker, "ts": str(ts)[:16],
                    "direction": sig_old["direction"],
                    "score": sig_old["score"],
                    "outcome": outcome,
                })

            # New signal
            sig_new = new_signal(close, high, low, vol, price)
            if sig_new:
                outcome = check_outcome(df_raw, i, sig_new["entry"],
                                        sig_new["stop"], sig_new["t1"],
                                        sig_new["direction"])
                results["new"].append({
                    "ticker": ticker, "ts": str(ts)[:16],
                    "direction": sig_new["direction"],
                    "score": sig_new["score"],
                    "adx": sig_new.get("adx"),
                    "outcome": outcome,
                })

    except Exception as e:
        print(f"  ⚠ {ticker}: {e}")
    return results

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CRYPTO BACKTEST — Old vs New Signal Engine")
print("  Test window: past 7 days | 1H chart")
print("="*60)

all_old, all_new = [], []

for i, ticker in enumerate(CRYPTO, 1):
    print(f"  [{i:02d}/{len(CRYPTO)}] {ticker:<20}", end=" ", flush=True)
    r = backtest_ticker(ticker)
    all_old.extend(r["old"])
    all_new.extend(r["new"])
    print(f"old:{len(r['old'])} new:{len(r['new'])}")

# ── Deduplicate: 1 signal per ticker per 12h window (no consecutive pile-ons) ─
def dedup(signals, hours=12):
    seen = {}
    out  = []
    for s in signals:
        key = (s["ticker"], s["direction"])
        ts  = s["ts"]
        if key not in seen or ts > seen[key]:
            out.append(s)
            seen[key] = ts  # naive dedup: skip until ts > last+12h
    return out

# ── Results ───────────────────────────────────────────────────────────────────
def summarize(signals, label, score_min=None):
    if not signals:
        print(f"\n{label}: No signals generated")
        return
    total  = len(signals)
    wins   = sum(1 for s in signals if s["outcome"] == "WIN")
    losses = sum(1 for s in signals if s["outcome"] == "LOSS")
    active = sum(1 for s in signals if s["outcome"] == "ACTIVE")
    resolved = wins + losses
    wr = wins / resolved * 100 if resolved > 0 else 0

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Total signals  : {total}")
    print(f"  Resolved       : {resolved}  (WIN:{wins}  LOSS:{losses})")
    print(f"  Still active   : {active}")
    print(f"  Win rate       : {wr:.1f}%  (resolved trades only)")

    # By ticker
    from collections import defaultdict
    by_ticker = defaultdict(lambda: {"W":0,"L":0,"A":0})
    for s in signals:
        by_ticker[s["ticker"]][s["outcome"][0]] += 1
    print(f"\n  Ticker breakdown:")
    for t, v in sorted(by_ticker.items(), key=lambda x: -(x[1]["W"]+x[1]["L"])):
        res = v["W"] + v["L"]
        pct = v["W"]/res*100 if res > 0 else 0
        print(f"    {t:<20} W:{v['W']} L:{v['L']} A:{v['A']}  ({pct:.0f}%)")

    # Sample signals
    resolved_sigs = [s for s in signals if s["outcome"] != "ACTIVE"]
    if resolved_sigs:
        print(f"\n  Sample resolved signals:")
        for s in resolved_sigs[:10]:
            adx_str = f" ADX:{s.get('adx','')}" if s.get("adx") else ""
            icon = "✅" if s["outcome"] == "WIN" else "❌"
            print(f"    {icon} {s['ticker']:<18} {s['direction']:<5} S:{s['score']}{adx_str}  ({s['ts']})")

def wr_pct(sigs):
    res = [s for s in sigs if s["outcome"] != "ACTIVE"]
    if not res: return 0.0, 0, len(res)
    w = sum(1 for s in res if s["outcome"] == "WIN")
    return w / len(res) * 100, w, len(res)

# ── Print results at multiple score thresholds ────────────────────────────────
print("\n" + "="*60)
print("  RESULTS BY SCORE THRESHOLD  (deduplicated signals)")
print("="*60)

thresholds = [
    ("All signals  (score ≥ 5)",  5),
    ("Good signals (score ≥ 8)",  8),
    ("HC signals   (score ≥ 10)", 10),
    ("HC signals   (score ≥ 12)", 12),
]

print(f"\n  {'Threshold':<30} {'OLD':>20}  {'NEW':>20}  {'Δ':>8}")
print(f"  {'-'*28} {'-'*20}  {'-'*20}  {'-'*8}")

for label, sc in thresholds:
    o = [s for s in all_old if s["score"] >= sc]
    n = [s for s in all_new if s["score"] >= sc]
    o_pct, o_w, o_r = wr_pct(o)
    n_pct, n_w, n_r = wr_pct(n)
    delta = n_pct - o_pct
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<30} {o_pct:>5.1f}% ({o_w}W/{o_r-o_w}L n={len(o):>4})  "
          f"{n_pct:>5.1f}% ({n_w}W/{n_r-n_w}L n={len(n):>4})  "
          f"{sign}{delta:.1f}%")

# ── Detailed breakdown at score >= 8 ─────────────────────────────────────────
print("\n" + "="*60)
print("  TICKER BREAKDOWN AT SCORE ≥ 8")
print("="*60)

from collections import defaultdict

for logic_label, signals in [("OLD", all_old), ("NEW", all_new)]:
    sigs = [s for s in signals if s["score"] >= 8]
    print(f"\n  {logic_label}  ({len(sigs)} signals)")
    print(f"  {'Ticker':<22} {'W':>4} {'L':>4} {'A':>4}  {'WR%':>5}")
    print(f"  {'-'*44}")
    by_t = defaultdict(lambda: {"W":0,"L":0,"A":0})
    for s in sigs:
        by_t[s["ticker"]][s["outcome"][0]] += 1
    for t, v in sorted(by_t.items(), key=lambda x: -(x[1]["W"]+x[1]["L"])):
        res = v["W"] + v["L"]
        pct = v["W"]/res*100 if res > 0 else 0
        print(f"  {t:<22} {v['W']:>4} {v['L']:>4} {v['A']:>4}  {pct:>4.0f}%")

print()

