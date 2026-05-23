"""
Meme Coin Monitor v1.0
Scans Dexscreener, Pump.fun, and Reddit every 30 minutes.
Alerts via Telegram when a coin shows early cross-platform momentum.
"""

import os, json, re, math, datetime, time, urllib.request, urllib.parse
from typing import Optional

# ── Config ──────────────────���─────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

MAX_MCAP          = 10_000_000   # $10M — above this it's not "early"
MIN_VOL_RATIO     = 0.20         # Volume/mcap must be ≥ 20% (active trading)
ALERT_SCORE_MIN   = 4            # Min combined score to send alert
RESCAN_HOURS      = 24           # Don't re-alert same coin within 24h

ALERT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meme_alerts.json")

SUBREDDITS = [
    "CryptoMoonShots",
    "SatoshiStreetBets",
    "memecoins",
    "solana",
    "lowcapcrypto",
]

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"  [Telegram] {msg[:100]}")
        return
    try:
        payload = json.dumps({
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       msg[:4096],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        print(f"  ⚠ Telegram error: {e}")

# ── Seen alerts (dedup) ─────────────────��────────────────────────��────────────
def load_seen() -> dict:
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_seen(seen: dict):
    with open(ALERT_FILE, "w") as f:
        json.dump(seen, f, indent=2)

def already_alerted(seen: dict, key: str) -> bool:
    if key not in seen:
        return False
    alerted_at = datetime.datetime.fromisoformat(seen[key])
    hours_ago  = (datetime.datetime.utcnow() - alerted_at).total_seconds() / 3600
    return hours_ago < RESCAN_HOURS

def mark_alerted(seen: dict, key: str):
    seen[key] = datetime.datetime.utcnow().isoformat()

# ── HTTP helper ───────────────────────────────────────────��───────────────────
def get_json(url: str, headers: dict = None, timeout: int = 12) -> Optional[dict]:
    try:
        h = {"User-Agent": "MemeMonitor/1.0 (crypto research bot)"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  ⚠ fetch {url[:60]}: {e}")
        return None

# ── Dexscreener ───────────��───────────────────────────────────────────────────
def fetch_dexscreener() -> list:
    """Fetch trending/boosted Solana coins from Dexscreener."""
    found = {}

    # Top boosted tokens — people paying to promote = lots of eyeballs
    data = get_json("https://api.dexscreener.com/token-boosts/top/v1")
    if data and isinstance(data, list):
        for item in data[:100]:
            if item.get("chainId") != "solana":
                continue
            addr = item.get("tokenAddress", "")
            if addr and addr not in found:
                found[addr] = {"boost_amount": item.get("totalAmount", 0), "chain": "solana"}

    # Latest token profiles
    data = get_json("https://api.dexscreener.com/token-profiles/latest/v1")
    if data and isinstance(data, list):
        for item in data[:50]:
            if item.get("chainId") != "solana":
                continue
            addr = item.get("tokenAddress", "")
            if addr and addr not in found:
                found[addr] = {"chain": "solana"}

    # Now fetch pair data for each address to get mcap/volume
    results = []
    for addr in list(found.keys())[:30]:  # cap at 30 to avoid rate limits
        time.sleep(0.2)
        pair_data = get_json(f"https://api.dexscreener.com/latest/dex/tokens/{addr}")
        if not pair_data:
            continue
        pairs = pair_data.get("pairs") or []
        if not pairs:
            continue
        # Use the highest volume pair
        pairs = sorted(pairs, key=lambda p: (p.get("volume") or {}).get("h24", 0) or 0, reverse=True)
        p = pairs[0]

        mcap   = p.get("marketCap", 0) or 0
        vol24  = (p.get("volume") or {}).get("h24", 0) or 0
        price  = float(p.get("priceUsd") or 0)
        name   = (p.get("baseToken") or {}).get("name", "Unknown")
        symbol = (p.get("baseToken") or {}).get("symbol", "???")
        url    = p.get("url", f"https://dexscreener.com/solana/{addr}")

        # Age from pair creation
        created_at = p.get("pairCreatedAt")  # unix ms
        age_hours  = None
        if created_at:
            age_hours = (time.time() - created_at / 1000) / 3600

        if mcap <= 0 or mcap > MAX_MCAP:
            continue
        if vol24 <= 0:
            continue
        vol_ratio = vol24 / mcap if mcap > 0 else 0
        if vol_ratio < MIN_VOL_RATIO:
            continue

        results.append({
            "source":     "dexscreener",
            "address":    addr,
            "name":       name,
            "symbol":     symbol.upper(),
            "chain":      "Solana",
            "mcap":       mcap,
            "vol24":      vol24,
            "vol_ratio":  round(vol_ratio * 100, 1),
            "price":      price,
            "age_hours":  age_hours,
            "url":        url,
        })

    print(f"  Dexscreener: {len(results)} qualifying coins")
    return results

# ── Pump.fun ──────────────���─────────────────────────────��─────────────────────
def fetch_pumpfun() -> list:
    """Fetch trending new coins from Pump.fun."""
    results = []
    # Sort by recently created + active trading
    data = get_json(
        "https://client-api-2-74b1891ee9f9.herokuapp.com/coins"
        "?limit=50&sort=last_reply&order=DESC&includeNsfw=false&offset=0"
    )
    if not data or not isinstance(data, list):
        # Fallback endpoint
        data = get_json(
            "https://frontend-api.pump.fun/coins"
            "?limit=50&sort=last_reply&order=DESC&includeNsfw=false&offset=0"
        )
    if not data or not isinstance(data, list):
        print("  Pump.fun: API unavailable")
        return results

    for coin in data:
        mcap   = float(coin.get("usd_market_cap", 0) or 0)
        symbol = str(coin.get("symbol", "???")).upper()
        name   = coin.get("name", "Unknown")
        addr   = coin.get("mint", "")
        reply_count = int(coin.get("reply_count", 0) or 0)
        created_ts  = coin.get("created_timestamp")

        age_hours = None
        if created_ts:
            age_hours = (time.time() - created_ts / 1000) / 3600

        if mcap <= 0 or mcap > MAX_MCAP:
            continue
        if reply_count < 5:  # some community activity
            continue

        results.append({
            "source":       "pumpfun",
            "address":      addr,
            "name":         name,
            "symbol":       symbol,
            "chain":        "Solana",
            "mcap":         mcap,
            "vol24":        0,
            "vol_ratio":    0,
            "price":        0,
            "age_hours":    age_hours,
            "reply_count":  reply_count,
            "url":          f"https://pump.fun/{addr}",
        })

    print(f"  Pump.fun: {len(results)} qualifying coins")
    return results

# ── Reddit ────────────────���───────────────────────────────────────────────────
def fetch_reddit() -> dict:
    """Scrape Reddit hot/new posts for ticker mentions. Returns {TICKER: {count, score, posts}}"""
    mentions = {}

    for sub in SUBREDDITS:
        for sort in ["hot", "new"]:
            time.sleep(0.5)  # be polite
            data = get_json(
                f"https://www.reddit.com/r/{sub}/{sort}.json?limit=50",
                headers={"User-Agent": "MemeMonitor/1.0 crypto research"},
            )
            if not data:
                continue
            posts = (data.get("data") or {}).get("children", [])
            for post in posts:
                p     = post.get("data", {})
                title = p.get("title", "")
                text  = p.get("selftext", "")
                upvotes = int(p.get("score", 0) or 0)
                combined = f"{title} {text}"

                # Extract $TICKER patterns
                tickers = re.findall(r'\$([A-Za-z]{2,10})', combined)
                # Also catch plain uppercase words that look like tickers in meme coin context
                plain   = re.findall(r'\b([A-Z]{3,8})\b', title)
                all_t   = [t.upper() for t in tickers] + plain

                # Filter obvious non-coins
                skip = {"USD","BTC","ETH","SOL","THE","FOR","NOT","ALL",
                        "AND","ARE","YOU","HAS","HOW","NEW","TOP","GET",
                        "NOW","OUT","DAY","ITS","CAN","USE","HIT","WIN",
                        "MAX","ATH","JUST","LIKE","THIS","THAT","WITH",
                        "PUMP","MOON","HOLD","SELL","BUY","DYOR","FOMO",
                        "APR","MAY","JUN","UTC","PDT","CDT","EDT"}
                for t in all_t:
                    if t in skip or len(t) < 2:
                        continue
                    if t not in mentions:
                        mentions[t] = {"count": 0, "score": 0, "posts": [], "subs": set()}
                    mentions[t]["count"]  += 1
                    mentions[t]["score"]  += upvotes
                    mentions[t]["subs"].add(sub)
                    if len(mentions[t]["posts"]) < 3:
                        mentions[t]["posts"].append(title[:80])

    # Filter: only tickers mentioned 2+ times or with score > 50
    filtered = {k: v for k, v in mentions.items()
                if v["count"] >= 2 or v["score"] >= 50}
    print(f"  Reddit: {len(filtered)} tickers with meaningful mentions")
    return filtered

# ── Scoring ────────────────��──────────────────────────────────────────────────
def score_coin(coin: dict, reddit_mentions: dict) -> int:
    s = 0

    # Market cap tier
    mcap = coin.get("mcap", 0)
    if mcap < 500_000:    s += 3   # < $500k — very early
    elif mcap < 2_000_000: s += 2  # < $2M
    elif mcap < 5_000_000: s += 1  # < $5M

    # Volume/mcap ratio (activity)
    vr = coin.get("vol_ratio", 0)
    if vr >= 100: s += 3    # trading volume > market cap — very hot
    elif vr >= 50: s += 2
    elif vr >= 20: s += 1

    # Age
    age = coin.get("age_hours")
    if age is not None:
        if age < 12:  s += 3   # brand new
        elif age < 48: s += 2
        elif age < 168: s += 1  # < 1 week

    # Pump.fun community activity
    if coin.get("reply_count", 0) >= 50: s += 2
    elif coin.get("reply_count", 0) >= 20: s += 1

    # Reddit cross-reference (cross-platform = stronger signal)
    sym = coin.get("symbol", "").upper()
    if sym in reddit_mentions:
        rm = reddit_mentions[sym]
        s += 2
        if rm["score"] >= 100: s += 1  # high upvotes

    return s

# ── Format alert ─────────────���────────────────────────────────────────────────
def format_alert(coin: dict, score: int, reddit_mentions: dict) -> str:
    sym   = coin.get("symbol", "???")
    name  = coin.get("name", "Unknown")
    chain = coin.get("chain", "?")
    mcap  = coin.get("mcap", 0)
    vol   = coin.get("vol24", 0)
    vr    = coin.get("vol_ratio", 0)
    age   = coin.get("age_hours")
    url   = coin.get("url", "")
    src   = coin.get("source", "")

    age_str = f"{age:.1f}h old" if age is not None else "age unknown"

    mcap_str = f"${mcap/1_000_000:.2f}M" if mcap >= 1_000_000 else f"${mcap/1_000:.0f}K"
    vol_str  = f"${vol/1_000_000:.2f}M"  if vol  >= 1_000_000 else f"${vol/1_000:.0f}K"

    source_icons = {"dexscreener": "📈 Dexscreener", "pumpfun": "🚀 Pump.fun"}
    sources = [source_icons.get(src, src)]

    reddit_info = ""
    if sym in reddit_mentions:
        rm = reddit_mentions[sym]
        sub_list = ", ".join(list(rm["subs"])[:3])
        sources.append(f"💬 Reddit ({sub_list})")
        if rm["posts"]:
            reddit_info = f'\n💬 <i>"{rm["posts"][0]}"</i>'

    stars = "⭐" * min(score, 5)

    lines = [
        f"🚨 <b>MEME GEM SIGNAL {stars}</b>",
        f"",
        f"<b>${sym}</b> — {name}",
        f"⛓ {chain}  |  Score: {score}/10",
        f"",
        f"💰 Market Cap: {mcap_str}",
    ]
    if vol > 0:
        lines.append(f"📊 24h Volume: {vol_str}  ({vr:.0f}% of mcap)")
    lines.append(f"⏰ Age: {age_str}")
    if coin.get("reply_count"):
        lines.append(f"💬 Pump.fun replies: {coin['reply_count']}")
    lines.append(f"")
    lines.append(f"📍 Found on: {' | '.join(sources)}")
    if reddit_info:
        lines.append(reddit_info)
    lines.append(f"")
    lines.append(f"���� <a href='{url}'>View on {source_icons.get(src,'chart').split()[-1]}</a>")
    if src == "pumpfun":
        lines.append(f"🔗 <a href='https://dexscreener.com/solana/{coin.get(\"address\",\"\")}'>Dexscreener</a>")
    lines.append(f"")
    lines.append(f"⚠️ <i>DYOR — meme coins are extremely high risk</i>")

    return "\n".join(lines)

# ── Main ─────────────��───────────────────────────���────────────────────────────
def run_monitor():
    print("\n" + "="*50)
    print(f"  Meme Monitor — {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*50)

    seen = load_seen()

    # Fetch data
    print("\n📡 Fetching sources...")
    dex_coins  = fetch_dexscreener()
    pump_coins = fetch_pumpfun()
    reddit     = fetch_reddit()

    all_coins = dex_coins + pump_coins

    # Deduplicate by symbol (dex + pump might overlap)
    seen_symbols = {}
    unique_coins = []
    for c in all_coins:
        sym = c.get("symbol", "").upper()
        addr = c.get("address", "")
        key  = addr or sym
        if key and key not in seen_symbols:
            seen_symbols[key] = True
            unique_coins.append(c)

    print(f"\n🔍 Scoring {len(unique_coins)} coins...")

    alerts_sent = 0
    for coin in unique_coins:
        sym   = coin.get("symbol", "???")
        addr  = coin.get("address", "")
        dedup_key = addr or sym

        if already_alerted(seen, dedup_key):
            continue

        s = score_coin(coin, reddit)
        if s < ALERT_SCORE_MIN:
            continue

        mcap_str = f"${coin['mcap']/1000:.0f}K" if coin['mcap'] < 1_000_000 else f"${coin['mcap']/1_000_000:.1f}M"
        print(f"  🚨 ${sym:<12} score:{s}  mcap:{mcap_str}  src:{coin['source']}")

        msg = format_alert(coin, s, reddit)
        send_telegram(msg)
        mark_alerted(seen, dedup_key)
        alerts_sent += 1
        time.sleep(1)  # don't flood Telegram

    # Also alert on Reddit tickers that aren't on dex yet (very early)
    known_syms = {c.get("symbol","").upper() for c in unique_coins}
    for sym, rm in reddit.items():
        if sym in known_syms:
            continue
        if already_alerted(seen, f"reddit_{sym}"):
            continue
        # Only alert if appearing in multiple subreddits or high score
        if len(rm["subs"]) >= 2 or rm["score"] >= 200:
            print(f"  💬 Reddit-only: ${sym}  score:{rm['score']}  subs:{','.join(rm['subs'])}")
            msg = (
                f"💬 <b>REDDIT BUZZ: ${sym}</b>\n\n"
                f"Not yet on Dexscreener — mentioned across {len(rm['subs'])} subreddit(s)\n"
                f"Upvotes: {rm['score']}\n\n"
                f"📌 Posts:\n" +
                "\n".join(f"• {p}" for p in rm["posts"][:3]) +
                f"\n\n🔍 Search: dexscreener.com + pump.fun for <b>${sym}</b>"
            )
            send_telegram(msg)
            mark_alerted(seen, f"reddit_{sym}")
            alerts_sent += 1
            time.sleep(1)

    save_seen(seen)

    print(f"\n✅ Done — {alerts_sent} alerts sent")
    print(f"   {len(unique_coins)} coins scanned | {len(reddit)} Reddit tickers tracked")

if __name__ == "__main__":
    run_monitor()
