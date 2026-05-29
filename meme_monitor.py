"""
Momentum Monitor v2.0
GME-style momentum detection across crypto and stocks — any size, any age.
Tracks StockTwits velocity, Twitter/X mentions, GeckoTerminal, CoinGecko trending.
"""

import os, json, re, datetime, time, urllib.request, urllib.parse
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")   # optional

ALERT_SCORE_MIN = 10
RESCAN_HOURS    = 12   # don't re-alert same ticker within 12h
VELOCITY_HOURS  = 2    # Reddit posts created in last N hours = "velocity"

ALERT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meme_alerts.json")

STOCK_SUBS = {
    "wallstreetbets", "Superstonk", "shortsqueeze", "pennystocks", "stocks", "options",
}

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


# ── StockTwits ────────────────────────────────────────────────────────────────
def fetch_stocktwits() -> dict:
    """
    StockTwits public API — no auth required, works from any IP.
    Pulls the trending tickers + message volume/sentiment for each.
    Returns {TICKER: {velocity, upvotes, subs, stock_subs, posts, recent_posts}}
    in the same shape the scorer expects, so no downstream changes needed.
    """
    mentions: dict = {}
    now_ts = datetime.datetime.utcnow().timestamp()
    cutoff  = now_ts - VELOCITY_HOURS * 3600

    # 1. Trending tickers (top 30 by message volume right now)
    trending = get_json(
        "https://api.stocktwits.com/api/2/trending/symbols.json"
        "?limit=30",
        headers={"User-Agent": "MomentumMonitor/2.0"},
    )
    trending_syms = []
    if trending:
        for item in (trending.get("symbols") or []):
            sym = (item.get("symbol") or "").upper()
            if sym and sym not in SKIP:
                trending_syms.append(sym)
                wl = int(item.get("watchlist_count") or 0)
                mentions[sym] = {
                    "velocity": 2,   # trending = baseline velocity signal
                    "upvotes": wl, "comments": 0,  # watchlist_count → upvotes for scoring
                    "subs": {"stocktwits_trending"}, "stock_subs": set(),
                    "posts": [], "recent_posts": [],
                    "watchlist_count": wl,
                }

    # 2. Per-ticker stream: message count + sentiment in last window
    for sym in trending_syms[:20]:
        time.sleep(0.3)
        stream = get_json(
            f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
            f"?limit=30",
            headers={"User-Agent": "MomentumMonitor/2.0"},
        )
        if not stream:
            continue
        messages = stream.get("messages") or []
        bull, bear, recent = 0, 0, 0
        for msg in messages:
            created_str = msg.get("created_at", "")
            try:
                created_ts = datetime.datetime.strptime(
                    created_str, "%Y-%m-%dT%H:%M:%SZ"
                ).timestamp()
            except Exception:
                created_ts = 0
            sentiment = (msg.get("entities") or {}).get("sentiment") or {}
            if sentiment.get("basic") == "Bullish":
                bull += 1
            elif sentiment.get("basic") == "Bearish":
                bear += 1
            if created_ts >= cutoff:
                recent += 1
                if len(mentions[sym]["recent_posts"]) < 2:
                    mentions[sym]["recent_posts"].append(msg.get("body", "")[:80])

        mentions[sym]["velocity"]  += recent
        # upvotes stays as watchlist_count (set in step 1) — bull count is too small to score
        mentions[sym]["comments"]   = bear
        if bull > bear:
            mentions[sym]["subs"].add("stocktwits_bullish")
        # Flag as stock context if no crypto suffix pattern
        if not re.search(r'\.(X|USD|BTC|ETH)$', sym):
            mentions[sym]["stock_subs"].add("stocktwits")

        if mentions[sym]["posts"] == []:
            mentions[sym]["posts"] = [f"StockTwits: {bull}B/{bear}Be in last 30 msgs"]

    print(f"  StockTwits: {len(mentions)} tickers with signal")
    return mentions


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


# ── GeckoTerminal ─────────────────────────────────────────────────────────────
def fetch_geckoterminal() -> list:
    """
    GeckoTerminal trending pools — free, no auth, not blocked on datacenter IPs.
    Replaces Dexscreener which returns 403 from GitHub Actions.
    """
    results    = []
    seen_addrs = set()

    for page in range(1, 3):
        data = get_json(
            f"https://api.geckoterminal.com/api/v2/networks/trending_pools?page={page}",
            headers={"Accept": "application/json", "User-Agent": "MomentumMonitor/2.0"},
        )
        if not data:
            break

        for pool in (data.get("data") or []):
            attr = pool.get("attributes") or {}
            rels = pool.get("relationships") or {}

            pool_addr = attr.get("address", "")
            if not pool_addr or pool_addr in seen_addrs:
                continue
            seen_addrs.add(pool_addr)

            pool_name = attr.get("name", "")
            symbol = pool_name.split(" / ")[0].strip().upper() if pool_name else "?"
            if not symbol or symbol in SKIP:
                continue

            network = ((rels.get("network") or {}).get("data") or {}).get("id", "")

            pc     = attr.get("price_change_percentage") or {}
            pc_h1  = float(pc.get("h1",  0) or 0)
            pc_h6  = float(pc.get("h6",  0) or 0)
            pc_h24 = float(pc.get("h24", 0) or 0)

            vol    = attr.get("volume_usd") or {}
            vol24  = float(vol.get("h24", 0) or 0)
            vol1   = float(vol.get("h1",  0) or 0)

            mcap = float(attr.get("market_cap_usd") or attr.get("fdv_usd") or 0)

            txns     = attr.get("transactions") or {}
            buys_h1  = int((txns.get("h1")  or {}).get("buys", 0) or 0)
            buys_h24 = int((txns.get("h24") or {}).get("buys", 0) or 0)

            normal_h1   = buys_h24 / 24 if buys_h24 > 0 else 0
            whale_ratio = (buys_h1 / normal_h1) if normal_h1 > 0 else 0
            vol_ratio   = (vol24 / mcap * 100) if mcap > 0 else 0

            results.append({
                "source":      "geckoterminal",
                "address":     pool_addr,
                "symbol":      symbol,
                "name":        pool_name,
                "chain":       network,
                "mcap":        mcap,
                "vol24":       vol24,
                "vol1":        vol1,
                "vol_ratio":   vol_ratio,
                "pc_h1":       pc_h1,
                "pc_h6":       pc_h6,
                "pc_h24":      pc_h24,
                "whale_ratio": whale_ratio,
                "url":         f"https://www.geckoterminal.com/{network}/pools/{pool_addr}",
            })

        time.sleep(0.5)

    print(f"  GeckoTerminal: {len(results)} tokens")
    return results


# ── CoinGecko Trending ────────────────────────────────────────────────────────
def fetch_coingecko_trending() -> list:
    """
    CoinGecko trending coins — free, no auth, not blocked on datacenter IPs.
    Replaces Pump.fun which is down (530/503 on all known endpoints).
    """
    results = []

    data = get_json(
        "https://api.coingecko.com/api/v3/search/trending",
        headers={"Accept": "application/json", "User-Agent": "MomentumMonitor/2.0"},
    )
    if not data:
        print("  CoinGecko trending: no data")
        return results

    for entry in (data.get("coins") or []):
        item = entry.get("item") or {}
        symbol = (item.get("symbol") or "").upper()
        name   = item.get("name") or symbol

        if not symbol or symbol in SKIP:
            continue

        coin_data = item.get("data") or {}

        pc_h24_map = coin_data.get("price_change_percentage_24h") or {}
        pc_h24 = float(pc_h24_map.get("usd", 0) or 0)

        # market_cap and total_volume may be formatted strings ("$1.23B") or numbers
        mcap_raw = coin_data.get("market_cap", 0)
        mcap = float(mcap_raw) if isinstance(mcap_raw, (int, float)) else 0.0

        vol_raw = coin_data.get("total_volume", 0)
        vol24 = float(vol_raw) if isinstance(vol_raw, (int, float)) else 0.0

        vol_ratio = (vol24 / mcap * 100) if mcap > 0 else 0
        coin_id   = item.get("id", "")

        results.append({
            "source":      "coingecko",
            "address":     coin_id,
            "symbol":      symbol,
            "name":        name,
            "chain":       "various",
            "mcap":        mcap,
            "vol24":       vol24,
            "vol1":        0,
            "vol_ratio":   vol_ratio,
            "pc_h1":       0,
            "pc_h6":       0,
            "pc_h24":      pc_h24,
            "whale_ratio": 0,
            "url":         f"https://www.coingecko.com/en/coins/{coin_id}" if coin_id else "",
        })

    print(f"  CoinGecko trending: {len(results)} coins")
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

    # StockTwits velocity — the GME signal
    if reddit:
        vel = reddit.get("velocity", 0)
        if vel >= 20:   s += 5; reasons.append(f"🔥 StockTwits surge: {vel} msgs in {VELOCITY_HOURS}h")
        elif vel >= 10: s += 4; reasons.append(f"📈 StockTwits velocity: {vel} msgs in {VELOCITY_HOURS}h")
        elif vel >= 5:  s += 3; reasons.append(f"📊 StockTwits activity: {vel} msgs in {VELOCITY_HOURS}h")
        elif vel >= 3:  s += 2; reasons.append(f"💬 StockTwits rising: {vel} msgs in {VELOCITY_HOURS}h")
        elif vel >= 1:  s += 1; reasons.append(f"👀 StockTwits mention: {vel} msg in {VELOCITY_HOURS}h")

        up = reddit.get("upvotes", 0)  # watchlist_count for StockTwits
        if up >= 10000: s += 3; reasons.append(f"👀 {up:,} watchlists")
        elif up >= 2000: s += 2; reasons.append(f"👀 {up:,} watchlists")
        elif up >= 500:  s += 1; reasons.append(f"👀 {up:,} watchlists")

        nsubs = len(reddit.get("subs", set()))
        if nsubs >= 5:   s += 3; reasons.append(f"🌐 Trending in {nsubs} sources")
        elif nsubs >= 3: s += 2; reasons.append(f"🌐 Mentioned in {nsubs} sources")
        elif nsubs >= 2: s += 1; reasons.append(f"🌐 Mentioned in {nsubs} sources")

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

        if dex.get("reply_count", 0) >= 50: s += 2; reasons.append("💬 50+ community replies")
        elif dex.get("reply_count", 0) >= 20: s += 1; reasons.append("💬 20+ community replies")

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
        vel  = reddit.get("velocity", 0)
        bull = reddit.get("upvotes", 0)
        bear = reddit.get("comments", 0)
        wl   = reddit.get("watchlist_count", 0)
        lines.append(f"📊 StockTwits: velocity {vel}  |  🟢{bull} bull / 🔴{bear} bear")
        if wl:
            lines.append(f"   👀 {wl:,} watchlists")
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
    if src == "coingecko" and address:
        lines.append(f"🔗 <a href='https://www.coingecko.com/en/coins/{address}'>CoinGecko</a>")

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

    print("\n[1/4] StockTwits scan...")
    social_data = fetch_stocktwits()

    # Twitter for top StockTwits tickers (optional — skips if no bearer token)
    print("\n[2/4] Twitter/X scan (top tickers)...")
    top_tickers = sorted(
        social_data, key=lambda t: social_data[t].get("velocity", 0), reverse=True
    )[:10]
    twitter_data = fetch_twitter(top_tickers)

    print("\n[3/4] GeckoTerminal scan...")
    dex_coins = fetch_geckoterminal()
    dex_by_sym: dict = {}
    for c in dex_coins:
        sym = c["symbol"]
        if sym not in dex_by_sym or c.get("mcap", 0) > dex_by_sym[sym].get("mcap", 0):
            dex_by_sym[sym] = c

    print("\n[4/4] CoinGecko trending scan...")
    for c in fetch_coingecko_trending():
        sym = c["symbol"]
        if sym not in dex_by_sym:
            dex_by_sym[sym] = c

    all_tickers = set(social_data.keys()) | set(dex_by_sym.keys())
    print(f"\n{'='*52}")
    print(f"  Scoring {len(all_tickers)} tickers...")

    alerts_sent = 0
    now_iso = datetime.datetime.utcnow().isoformat()

    for ticker in all_tickers:
        reddit  = social_data.get(ticker)   # same shape, scorer unchanged
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
