"""
Momentum Monitor v2.0
GME-style momentum detection across crypto and stocks — any size, any age.
Tracks Reddit post velocity, Twitter/X mentions, Dexscreener, Pump.fun.
"""

import os, json, re, datetime, time, urllib.request, urllib.parse
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")  # optional

ALERT_SCORE_MIN = 6
RESCAN_HOURS    = 12   # don't re-alert same ticker within 12h
VELOCITY_HOURS  = 2    # Reddit posts created in last N hours = "velocity"

ALERT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meme_alerts.json")

CRYPTO_SUBS = [
    "CryptoMoonShots", "SatoshiStreetBets", "memecoins",
    "solana", "lowcapcrypto", "CryptoCurrency", "altcoin", "defi",
]
STOCK_SUBS = [
    "wallstreetbets", "Superstonk", "shortsqueeze",
    "pennystocks", "stocks", "options",
]
ALL_SUBS = CRYPTO_SUBS + STOCK_SUBS

SKIP = {
    "USD","BTC","ETH","SOL","THE","FOR","NOT","ALL","AND","ARE","YOU",
    "HAS","HOW","NEW","TOP","GET","NOW","OUT","DAY","ITS","CAN","USE",
    "HIT","WIN","MAX","ATH","JUST","LIKE","THIS","THAT","WITH","PUMP",
    "MOON","HOLD","SELL","BUY","DYOR","FOMO","APR","MAY","JUN","UTC",
    "PDT","CDT","EDT","YOLO","BULL","BEAR","LONG","SHORT","GAIN","LOSS",
    "PRINT","BASED","MEGA","HOLY","HUGE","VERY","MUCH","SUCH","MANY",
    "GOOD","NEXT","WHEN","WHAT","WILL","APES","GANG","LETS","LMAO",
    "EDIT","TLDR","IMO","AMA","EOD","CEO","CFO","CTO","IPO","SEC",
    "FDA","FED","USA","GDP","CPI","ATM","OTC","WSB","DD","TA","FA",
    "CALLS","PUTS","CALL","PUT","CASH","DEBT","RISK","HIGH","BACK",
    "NEED","WANT","MAKE","TAKE","GIVE","COME","LOOK","FEEL","PLAY",
    "OPEN","CLOSE","BREAK","MOVE","PRICE","STOCK","SHARE","MONEY",
    "WEEK","YEAR","TIME","EVEN","ONLY","ALSO","SAME","INTO","FROM",
    "OVER","AFTER","BEEN","THEM","THEY","THEN","THAN","SOME","MORE",
    "MOST","LAST","WELL","REAL","RATE","RISE","FALL","DROP","LOAD",
    "ZERO","FIVE","FOUR","ONCE","DONE","SOON","FAST","SAFE","DEEP",
    "WIDE","FULL","HARD","EASY","NEWS","DATA","PLAN","DEAL","LIVE",
    "LOST","PAID","SOLD","WENT","WENT","BEST","EVER","HERE","SHOW",
}


# ── Utilities ─────────────────────────────────────────────────────────────────
def get_json(url: str, headers: dict = None) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  [warn] {url[:70]}: {e}")
        return None


def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] not configured")
        return
    data = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID, "text": msg,
        "parse_mode": "HTML", "disable_web_page_preview": True,
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data=data, headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  [Telegram error] {e}")


def load_seen() -> dict:
    try:
        with open(ALERT_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_seen(seen: dict):
    with open(ALERT_FILE, "w") as f:
        json.dump(seen, f, indent=2)


# ── Reddit velocity ────────────────────────────────────────────────────────────
def fetch_reddit() -> dict:
    """
    Returns {TICKER: {velocity, upvotes, comments, subs, stock_subs, posts, recent_posts}}
    velocity = number of posts created in the last VELOCITY_HOURS.
    This is the core GME signal — post count exploded hours before price did.
    """
    now_ts = datetime.datetime.utcnow().timestamp()
    cutoff  = now_ts - VELOCITY_HOURS * 3600
    mentions: dict = {}

    def record(ticker, upvotes, comments, sub, title, created_utc):
        t = ticker.upper()
        if t in SKIP or len(t) < 2 or len(t) > 10:
            return
        if t not in mentions:
            mentions[t] = {
                "velocity": 0, "upvotes": 0, "comments": 0,
                "subs": set(), "stock_subs": set(),
                "posts": [], "recent_posts": [],
            }
        m = mentions[t]
        m["upvotes"]  += max(upvotes, 0)
        m["comments"] += max(comments, 0)
        m["subs"].add(sub)
        if sub in STOCK_SUBS:
            m["stock_subs"].add(sub)
        if len(m["posts"]) < 3:
            m["posts"].append(title[:80])
        if created_utc >= cutoff:
            m["velocity"] += 1
            if len(m["recent_posts"]) < 2:
                m["recent_posts"].append(title[:80])

    for sub in ALL_SUBS:
        for sort in ["new", "hot"]:
            time.sleep(0.4)
            data = get_json(
                f"https://www.reddit.com/r/{sub}/{sort}.json?limit=100",
                headers={"User-Agent": "MomentumMonitor/2.0 signal research"},
            )
            if not data:
                continue
            for post in (data.get("data") or {}).get("children", []):
                p        = post.get("data", {})
                title    = p.get("title", "")
                text     = p.get("selftext", "")
                upvotes  = int(p.get("score", 0) or 0)
                comments = int(p.get("num_comments", 0) or 0)
                created  = float(p.get("created_utc", 0) or 0)
                combined = f"{title} {text}"

                for t in re.findall(r'\$([A-Za-z]{2,10})', combined):
                    record(t, upvotes, comments, sub, title, created)
                for t in re.findall(r'\b([A-Z]{2,8})\b', title):
                    record(t, upvotes, comments, sub, title, created)

    # Filter: must have at least velocity>=1, upvotes>=200, or 2+ subs
    filtered = {
        k: v for k, v in mentions.items()
        if v["velocity"] >= 1 or v["upvotes"] >= 200 or len(v["subs"]) >= 2
    }
    print(f"  Reddit: {len(filtered)} tickers with signal")
    return filtered


# ── Twitter / X ────────────────────────────────────────────────────────────────
def fetch_twitter(tickers: list) -> dict:
    """
    Returns {TICKER: {tweet_count, engagement}} for the top Reddit tickers.
    Skips entirely if TWITTER_BEARER_TOKEN is not set.
    Queries at most 10 tickers to stay within free-tier rate limits.
    """
    if not TWITTER_BEARER_TOKEN:
        print("  Twitter: no bearer token — skipping (add TWITTER_BEARER_TOKEN secret to enable)")
        return {}

    results = {}
    start_time = (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

    for ticker in tickers[:10]:
        time.sleep(2)
        query = urllib.parse.quote(f"${ticker} lang:en -is:retweet")
        url = (
            f"https://api.twitter.com/2/tweets/search/recent"
            f"?query={query}&max_results=100"
            f"&tweet.fields=public_metrics"
            f"&start_time={start_time}"
        )
        data = get_json(url, headers=headers)
        if not data:
            continue
        tweets = data.get("data") or []
        tweet_count = len(tweets)
        engagement  = sum(
            (t.get("public_metrics") or {}).get("like_count", 0) +
            (t.get("public_metrics") or {}).get("retweet_count", 0)
            for t in tweets
        )
        if tweet_count > 0:
            results[ticker] = {"tweet_count": tweet_count, "engagement": engagement}
            print(f"    ${ticker}: {tweet_count} tweets, {engagement} engagements")

    print(f"  Twitter: {len(results)} tickers with activity")
    return results


# ── Dexscreener ───────────────────────────────────────────────────────────────
def fetch_dexscreener() -> list:
    """
    Fetch trending/boosted tokens across all chains.
    Captures h1/h24 price change and buy-rate ratio for whale detection.
    No mcap or age filter — scoring handles weighting.
    """
    results  = []
    seen_addrs = set()

    for endpoint in ["/token-boosts/top/v1", "/token-profiles/latest/v1"]:
        data = get_json(f"https://api.dexscreener.com{endpoint}")
        if not data:
            continue
        tokens = data if isinstance(data, list) else []
        for tok in tokens[:40]:
            addr  = tok.get("tokenAddress", "")
            chain = tok.get("chainId", "")
            if not addr or addr in seen_addrs:
                continue
            seen_addrs.add(addr)
            time.sleep(0.2)
            pair_data = get_json(f"https://api.dexscreener.com/latest/dex/tokens/{addr}")
            if not pair_data:
                continue
            pairs = pair_data.get("pairs") or []
            if not pairs:
                continue
            p = pairs[0]

            mcap     = float((p.get("fdv") or p.get("marketCap") or 0))
            vol24    = float((p.get("volume") or {}).get("h24", 0) or 0)
            vol1     = float((p.get("volume") or {}).get("h1",  0) or 0)
            pc_h1    = float((p.get("priceChange") or {}).get("h1",  0) or 0)
            pc_h6    = float((p.get("priceChange") or {}).get("h6",  0) or 0)
            pc_h24   = float((p.get("priceChange") or {}).get("h24", 0) or 0)
            buys_h1  = int((p.get("txns") or {}).get("h1",  {}).get("buys", 0) or 0)
            buys_h24 = int((p.get("txns") or {}).get("h24", {}).get("buys", 0) or 0)

            symbol = ((p.get("baseToken") or {}).get("symbol") or tok.get("description") or "?").upper()
            name   = (p.get("baseToken") or {}).get("name") or symbol
            url    = p.get("url") or f"https://dexscreener.com/{chain}/{addr}"

            # Whale signal: h1 buy rate vs expected (24h / 24)
            normal_h1  = buys_h24 / 24 if buys_h24 > 0 else 0
            whale_ratio = (buys_h1 / normal_h1) if normal_h1 > 0 else 0

            vol_ratio = (vol24 / mcap * 100) if mcap > 0 else 0

            results.append({
                "source":      "dexscreener",
                "address":     addr,
                "symbol":      symbol,
                "name":        name,
                "chain":       chain,
                "mcap":        mcap,
                "vol24":       vol24,
                "vol1":        vol1,
                "vol_ratio":   vol_ratio,
                "pc_h1":       pc_h1,
                "pc_h6":       pc_h6,
                "pc_h24":      pc_h24,
                "whale_ratio": whale_ratio,
                "url":         url,
            })

    print(f"  Dexscreener: {len(results)} tokens")
    return results


# ── Pump.fun ──────────────────────────────────────────────────────────────────
def fetch_pumpfun() -> list:
    results = []
    data = get_json(
        "https://client-api-2-74b1891ee9f9.herokuapp.com/coins"
        "?offset=0&limit=50&sort=last_reply&includeNsfw=false",
        headers={"User-Agent": "MomentumMonitor/2.0"},
    )
    if not data:
        return results

    now_ts = datetime.datetime.utcnow().timestamp()
    for coin in (data if isinstance(data, list) else []):
        addr        = coin.get("mint", "")
        symbol      = (coin.get("symbol") or "").upper()
        name        = coin.get("name") or symbol
        mcap        = float(coin.get("usd_market_cap", 0) or 0)
        reply_count = int(coin.get("reply_count", 0) or 0)
        created_ts  = coin.get("created_timestamp", now_ts * 1000)
        age_hours   = (now_ts - created_ts / 1000) / 3600

        if reply_count < 5 or not addr:
            continue

        results.append({
            "source":      "pumpfun",
            "address":     addr,
            "symbol":      symbol,
            "name":        name,
            "chain":       "solana",
            "mcap":        mcap,
            "vol24":       0,
            "vol1":        0,
            "vol_ratio":   0,
            "pc_h1":       0,
            "pc_h6":       0,
            "pc_h24":      0,
            "whale_ratio": 0,
            "age_hours":   age_hours,
            "reply_count": reply_count,
            "url":         f"https://pump.fun/{addr}",
        })

    print(f"  Pump.fun: {len(results)} coins")
    return results


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_ticker(
    reddit:  Optional[dict],
    twitter: Optional[dict],
    dex:     Optional[dict],
) -> tuple:
    """Returns (score, reasons). Max ~25 pts across all signals."""
    s = 0
    reasons = []

    # Reddit velocity — the GME signal
    if reddit:
        vel = reddit.get("velocity", 0)
        if vel >= 20:   s += 5; reasons.append(f"🔥 Reddit surge: {vel} posts in {VELOCITY_HOURS}h")
        elif vel >= 10: s += 4; reasons.append(f"📈 Reddit velocity: {vel} posts in {VELOCITY_HOURS}h")
        elif vel >= 5:  s += 3; reasons.append(f"📊 Reddit activity: {vel} posts in {VELOCITY_HOURS}h")
        elif vel >= 3:  s += 2; reasons.append(f"💬 Reddit rising: {vel} posts in {VELOCITY_HOURS}h")
        elif vel >= 1:  s += 1; reasons.append(f"👀 Reddit mention: {vel} post in {VELOCITY_HOURS}h")

        up = reddit.get("upvotes", 0)
        if up >= 10000: s += 3; reasons.append(f"⬆️ {up:,} upvotes")
        elif up >= 2000: s += 2; reasons.append(f"⬆️ {up:,} upvotes")
        elif up >= 500:  s += 1; reasons.append(f"⬆️ {up:,} upvotes")

        nsubs = len(reddit.get("subs", set()))
        if nsubs >= 5:   s += 3; reasons.append(f"🌐 Trending in {nsubs} subreddits")
        elif nsubs >= 3: s += 2; reasons.append(f"🌐 Mentioned in {nsubs} subreddits")
        elif nsubs >= 2: s += 1; reasons.append(f"🌐 Mentioned in {nsubs} subreddits")

    # Twitter / X velocity
    if twitter:
        tc = twitter.get("tweet_count", 0)
        if tc >= 80:   s += 5; reasons.append(f"🐦 Twitter: {tc} tweets in 1h")
        elif tc >= 50: s += 4; reasons.append(f"🐦 Twitter: {tc} tweets in 1h")
        elif tc >= 20: s += 3; reasons.append(f"🐦 Twitter: {tc} tweets in 1h")
        elif tc >= 10: s += 2; reasons.append(f"🐦 Twitter: {tc} tweets in 1h")
        elif tc >= 3:  s += 1; reasons.append(f"🐦 Twitter: {tc} tweets in 1h")

        eng = twitter.get("engagement", 0)
        if eng >= 50000: s += 3; reasons.append(f"❤️ {eng:,} likes/RTs")
        elif eng >= 10000: s += 2; reasons.append(f"❤️ {eng:,} likes/RTs")
        elif eng >= 2000:  s += 1; reasons.append(f"❤️ {eng:,} likes/RTs")

    # DEX momentum (crypto only)
    if dex:
        pc1 = dex.get("pc_h1", 0)
        if pc1 >= 100:  s += 4; reasons.append(f"🚀 +{pc1:.0f}% price in 1h")
        elif pc1 >= 50: s += 3; reasons.append(f"🚀 +{pc1:.0f}% price in 1h")
        elif pc1 >= 20: s += 2; reasons.append(f"📈 +{pc1:.0f}% price in 1h")
        elif pc1 >= 10: s += 1; reasons.append(f"📈 +{pc1:.0f}% price in 1h")

        # Whale buy-rate spike
        wr = dex.get("whale_ratio", 0)
        if wr >= 5:   s += 3; reasons.append(f"🐳 {wr:.1f}x normal buy rate (whale activity)")
        elif wr >= 3: s += 2; reasons.append(f"🐳 {wr:.1f}x normal buy rate")
        elif wr >= 2: s += 1; reasons.append(f"🐳 {wr:.1f}x normal buy rate")

        vr = dex.get("vol_ratio", 0)
        if vr >= 200: s += 2; reasons.append(f"💥 Vol/mcap: {vr:.0f}%")
        elif vr >= 50: s += 1; reasons.append(f"📊 Vol/mcap: {vr:.0f}%")

        if dex.get("reply_count", 0) >= 50: s += 2; reasons.append("💬 50+ Pump.fun replies")
        elif dex.get("reply_count", 0) >= 20: s += 1; reasons.append("💬 20+ Pump.fun replies")

    return s, reasons


# ── Where to buy ──────────────────────────────────────────────────────────────
def _where_to_buy(chain: str, src: str, address: str, is_stock: bool) -> list:
    chain = (chain or "").lower()
    lines = []

    if is_stock:
        lines.append("  • <b>Webull</b> — stocks supported")
        lines.append("  • Coinbase / Blofin / Trust Wallet / MetaMask — crypto wallets, not for stocks")
        return lines

    if "solana" in chain or chain == "solana":
        if src == "pumpfun":
            lines.append("  • <b>Pump.fun</b> — buy direct if still on bonding curve")
        lines.append(f"  • <b>Trust Wallet</b> → Browser → <a href='https://jup.ag/swap/SOL-{address}'>Jupiter</a> (paste contract)")
        lines.append("  • <b>MetaMask</b> — ⚠️ Solana not natively supported; use Trust Wallet")
        lines.append("  • Blofin / Coinbase — check if listed; unlikely until after launch")
    elif any(x in chain for x in ["ethereum", "eth", "base", "polygon", "arbitrum", "optimism"]):
        lines.append("  • <b>MetaMask</b> → Uniswap — paste contract address")
        lines.append("  • <b>Trust Wallet</b> → DApp browser → Uniswap")
        if "base" in chain:
            lines.append("  • <b>Coinbase</b> — Base chain; may appear in Coinbase Wallet quickly")
        lines.append("  • Blofin / Webull Pay — CEX; unlikely to list early")
    else:
        lines.append("  • <b>Trust Wallet</b> — supports most chains; paste contract in DEX browser")
        lines.append("  • <b>MetaMask</b> — EVM chains only; check chain compatibility first")

    return lines


# ── Format alert ──────────────────────────────────────────────────────────────
def format_alert(
    ticker:   str,
    reddit:   Optional[dict],
    twitter:  Optional[dict],
    dex:      Optional[dict],
    score:    int,
    reasons:  list,
    is_stock: bool,
) -> str:
    stars      = "⭐" * min(score // 3, 5)
    asset_type = "📈 STOCK MOMENTUM" if is_stock else "🚨 CRYPTO MOMENTUM"
    name       = (dex or {}).get("name") or ticker
    chain      = (dex or {}).get("chain", "")
    address    = (dex or {}).get("address", "")
    src        = (dex or {}).get("source", "")
    url        = (dex or {}).get("url", "")

    lines = [
        f"{asset_type} {stars}",
        f"",
        f"<b>${ticker}</b>  —  {name}",
        f"Score: {score}  |  {'Stock' if is_stock else chain.capitalize() or 'Crypto'}",
        f"",
    ]

    if dex and not is_stock:
        mcap  = dex.get("mcap", 0)
        vol24 = dex.get("vol24", 0)
        pc1   = dex.get("pc_h1", 0)
        pc24  = dex.get("pc_h24", 0)
        mcap_s = f"${mcap/1_000_000:.2f}M" if mcap >= 1_000_000 else (f"${mcap/1_000:.0f}K" if mcap > 0 else "?")
        vol_s  = f"${vol24/1_000_000:.2f}M" if vol24 >= 1_000_000 else (f"${vol24/1_000:.0f}K" if vol24 > 0 else "?")
        lines.append(f"💰 Mcap: {mcap_s}  |  Vol 24h: {vol_s}")
        if pc1:
            lines.append(f"📊 Price: {pc1:+.1f}% (1h)  /  {pc24:+.1f}% (24h)")
        lines.append("")

    if reddit:
        vel   = reddit.get("velocity", 0)
        up    = reddit.get("upvotes", 0)
        nsubs = len(reddit.get("subs", set()))
        subs_list = ", ".join(list(reddit.get("subs", set()))[:4])
        lines.append(f"💬 Reddit: {vel} posts/{VELOCITY_HOURS}h  |  {up:,} upvotes  |  {nsubs} subs")
        lines.append(f"   <i>Subs: {subs_list}</i>")
        if reddit.get("recent_posts"):
            lines.append(f'   <i>"{reddit["recent_posts"][0]}"</i>')
        lines.append("")

    if twitter:
        tc  = twitter.get("tweet_count", 0)
        eng = twitter.get("engagement", 0)
        lines.append(f"🐦 Twitter: {tc} tweets/hr  |  {eng:,} likes+RTs")
        lines.append("")

    if reasons:
        lines.append("⚡ <b>Signals:</b>")
        for r in reasons[:6]:
            lines.append(f"  {r}")
        lines.append("")

    if url:
        lines.append(f"🔗 <a href='{url}'>Chart</a>")
    if src == "pumpfun" and address:
        lines.append(f"🔗 <a href='https://pump.fun/{address}'>Pump.fun</a>")

    buy = _where_to_buy(chain, src, address, is_stock)
    if buy:
        lines.append("")
        lines.append("🛒 <b>Where to buy:</b>")
        lines.extend(buy)

    lines.append("")
    lines.append("⚠️ <i>DYOR — high risk, not financial advice</i>")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_monitor():
    print("\n" + "="*52)
    print(f"  Momentum Monitor v2 — {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*52)

    seen = load_seen()

    print("\n[1/4] Reddit velocity scan...")
    reddit_data = fetch_reddit()

    # Only query Twitter for tickers with the strongest Reddit signal
    print("\n[2/4] Twitter/X scan (top Reddit tickers)...")
    top_tickers = sorted(
        reddit_data, key=lambda t: reddit_data[t].get("velocity", 0), reverse=True
    )[:10]
    twitter_data = fetch_twitter(top_tickers)

    print("\n[3/4] Dexscreener scan...")
    dex_coins = fetch_dexscreener()
    dex_by_sym: dict = {}
    for c in dex_coins:
        sym = c["symbol"]
        if sym not in dex_by_sym or c.get("mcap", 0) > dex_by_sym[sym].get("mcap", 0):
            dex_by_sym[sym] = c

    print("\n[4/4] Pump.fun scan...")
    for c in fetch_pumpfun():
        sym = c["symbol"]
        if sym not in dex_by_sym:
            dex_by_sym[sym] = c

    all_tickers = set(reddit_data.keys()) | set(dex_by_sym.keys())
    print(f"\n{'='*52}")
    print(f"  Scoring {len(all_tickers)} tickers...")

    alerts_sent = 0
    now_iso = datetime.datetime.utcnow().isoformat()

    for ticker in all_tickers:
        reddit  = reddit_data.get(ticker)
        twitter = twitter_data.get(ticker)
        dex     = dex_by_sym.get(ticker)

        score, reasons = score_ticker(reddit, twitter, dex)
        if score < ALERT_SCORE_MIN:
            continue

        entry       = seen.get(ticker, {})
        last_str    = entry.get("alerted_at", "1970-01-01")
        hours_since = (
            datetime.datetime.utcnow() - datetime.datetime.fromisoformat(last_str)
        ).total_seconds() / 3600
        if hours_since < RESCAN_HOURS:
            print(f"  [{ticker}] score={score} — skip (alerted {hours_since:.1f}h ago)")
            continue

        # Stock if it showed up in stock subs but has no DEX data
        is_stock = bool(reddit and reddit.get("stock_subs") and not dex)

        print(f"  [{ticker}] score={score} {'STOCK' if is_stock else ''} — ALERT")
        msg = format_alert(ticker, reddit, twitter, dex, score, reasons, is_stock)
        send_telegram(msg)
        seen[ticker] = {"alerted_at": now_iso, "score": score}
        alerts_sent += 1
        time.sleep(1)

    save_seen(seen)
    print(f"\n  Done. {alerts_sent} alert(s) sent.")


if __name__ == "__main__":
    run_monitor()
