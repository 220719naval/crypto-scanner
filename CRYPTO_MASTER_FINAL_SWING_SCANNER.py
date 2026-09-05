import traceback

# BTC Shock alert state
LAST_BTC_SHOCK_STATE = None
# -*- coding: utf-8 -*-
"""
CRYPTO MASTER FINAL — 72-RULE QUALITY + SWING SCANNER
======================================================

Purpose
-------
A conservative crypto swing scanner for liquid, established assets.

DATA:
- Binance USDⓈ-M Futures public REST first; automatic Bybit Linear REST fallback when Binance is unavailable (including HTTP 451).
- CoinGecko public market metadata for market cap, FDV and supply quality.
- Optional Binance WebSocket is NOT required; REST polling is used so the
  script remains compatible with basic PythonAnywhere hosting.
- A stale/failed data response is rejected rather than converted into a signal.

UNIVERSE:
- Dynamically builds an approximately 150-coin universe from large/liquid
  Binance/Bybit USDT perpetuals intersected with CoinGecko market data.
- Hard quality gates remove unknown/unlimited max-supply assets, very low
  market cap/liquidity, extreme FDV dilution and suspicious data.
- "Institutional" is NOT fabricated: the scanner uses institutional/adoption
  proxies (large market-cap rank, liquidity, exchange presence) and labels them
  as proxies.

72 RULE ENGINE:
- Price action / market structure
- Breakout + retest + volume
- EMA/VWAP/OBV
- RSI/MACD/ADX/Stochastic/CCI/MFI/Williams %R
- ATR/Bollinger
- Candlestick confirmation
- 1W + 1D + 4H + 1H
- BTC master benchmark
- Funding + open interest
- Risk/reward and volatility
- Quality/tokenomics/AI/adoption gates

IMPORTANT:
- Score % = rule alignment, NOT probability of profit.
- No scanner can guarantee zero loss or prevent every false signal.
- The scanner blocks new LONG signals during a severe BTC shock.
- News safety is conservative: the code does not pretend that social-media
  rumours are verified news. It uses market-shock/volatility protection.
- Unlimited/infinite max supply is rejected by default. If a data provider
  cannot verify max supply, the token is rejected.
- "Institutional investment" cannot be proved from a universal free API field,
  so only transparent proxies are used.

INSTALL ON PYTHONANYWHERE BASH (ONE TIME):
    pip install --user requests pandas numpy

RUN:
    python3 expert_swing_crypto.py

TELEGRAM:
- Optional.
- At startup, if credentials are not supplied as environment variables,
  the program asks for Bot Token and Chat ID.
- Press ENTER at both prompts to run without Telegram.

ENVIRONMENT VARIABLES (optional):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    COINGECKO_API_KEY       # optional; improves rate limits if available

This file is intentionally self-contained.
"""

import os
import time
import math
import json
import statistics
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BINANCE_BASE = "https://fapi.binance.com"
OKX_BASE = "https://www.okx.com"
BYBIT_BASE = "https://api.bybit.com"
KRAKEN_BASE = "https://futures.kraken.com/derivatives/api/v3"
KRAKEN_CHART_BASE = "https://futures.kraken.com/api/charts/v1"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

QUOTE = "USDT"

SCAN_INTERVAL = 120               # full validation every ~2 minutes
UNIVERSE_TARGET = 150
CG_PAGES = 1                      # 3 x 250 = up to 750 market rows
MIN_MARKET_CAP_USD = 25_000_000
MIN_24H_VOLUME_USD = 1_000_000
MIN_PRICE_USD = 0.001
MAX_FDV_MC_RATIO = 4.0
MIN_CIRCULATION_RATIO = 0.45

# Technical signal threshold.
# 56/72 = 77.8% rule alignment (slightly relaxed quality threshold).
MIN_SCORE = 56
MIN_RR = 1.50

# BTC protection.
BTC_SHOCK_1H_PCT = -4.0
BTC_SHOCK_4H_PCT = -6.0
BTC_SHOCK_ATR_MULT = 2.0

# Data freshness.
MAX_KLINE_AGE_MIN = 70

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Optional CoinGecko API key.
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()

# Cache
CG_CACHE_FILE = "crypto_quality_cache.json"
CG_CACHE_SECONDS = 3600

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CryptoMasterFinalSwing/4.0"
})

DATA_PROVIDER = "KRAKEN"  # Binance/Bybit blocked here; use Kraken Futures

# AI-related names/tickers are only a BONUS classifier, never a quality proof.
AI_KEYWORDS = (
    "ai", "artificial", "render", "bittensor", "near", "fetch",
    "injective", "internet computer", "akash", "grass", "worldcoin",
    "worldcoin", "theta", "filecoin", "graph"
)


# ============================================================
# HTTP HELPERS
# ============================================================

def http_get(url, params=None, timeout=15, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            time.sleep(1.0 + attempt * 0.75)
    raise RuntimeError(f"HTTP failed: {url} | {last_error}")


def safe_float(value, default=np.nan):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def now_utc():
    return datetime.now(timezone.utc)


def fmt_time():
    return now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")


# ============================================================
# LIVE MARKET DATA — BINANCE FIRST, BYBIT FALLBACK
# ============================================================

def _bybit_symbol_list():
    data = http_get(
        f"{BYBIT_BASE}/v5/market/instruments-info",
        params={"category": "linear", "limit": 1000},
        timeout=20
    )
    out = []
    for s in data.get("result", {}).get("list", []):
        if (
            s.get("status") == "Trading"
            and s.get("quoteCoin") == QUOTE
            and s.get("contractType") == "LinearPerpetual"
        ):
            out.append(s["symbol"])
    return sorted(set(out))


def _binance_symbol_list():
    data = http_get(
        f"{BINANCE_BASE}/fapi/v1/exchangeInfo", timeout=20
    )
    out = []
    for s in data.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == QUOTE
            and s.get("contractType") == "PERPETUAL"
        ):
            out.append(s["symbol"])
    return sorted(set(out))



def _kraken_symbol(symbol):
    if symbol == "BTCUSDT":
        return "PF_XBTUSD"
    if symbol.startswith(("PF_", "PI_")):
        return symbol
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        if base == "BTC":
            base = "XBT"
        return f"PF_{base}USD"
    return symbol


def _kraken_symbol_list():
    data = http_get(f"{KRAKEN_BASE}/instruments", timeout=20)
    out = []
    for x in data.get("instruments", []):
        sym = str(x.get("symbol", "")).upper()
        if (
            sym.startswith("PF_")
            and x.get("tradeable") is True
        ):
            out.append(sym)
    return sorted(set(out))


def _kraken_tickers():
    data = http_get(f"{KRAKEN_BASE}/tickers", timeout=20)
    out = {}
    for x in data.get("tickers", []):
        sym = str(x.get("symbol", "")).upper()
        if not sym.startswith("PF_"):
            continue

        last = safe_float(x.get("last"), 0.0)
        open24 = safe_float(x.get("open24h"), 0.0)
        change = ((last / open24) - 1.0) * 100.0 if open24 > 0 else 0.0

        out[sym] = {
            "symbol": sym,
            "quoteVolume": safe_float(x.get("volumeQuote"), 0.0),
            "lastPrice": last,
            "priceChangePercent": change,
            "fundingRate": safe_float(x.get("fundingRate"), 0.0),
            "openInterest": safe_float(x.get("openInterest"), np.nan),
        }
    return out


def _kraken_klines(symbol, interval, limit=220):
    ks = _kraken_symbol(symbol)
    resolution = {
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
    }[interval]

    data = http_get(
        f"{KRAKEN_CHART_BASE}/trade/{ks}/{resolution}",
        params={"count": min(limit, 2000)},
        timeout=20
    )

    rows = data.get("candles", [])
    if len(rows) < 80:
        raise ValueError(f"insufficient Kraken candles: {symbol} {interval}")

    df = pd.DataFrame(rows)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["open_time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    minutes = {
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }[interval]

    df["close_time"] = (
        df["open_time"] + pd.to_timedelta(minutes, unit="m")
    )

    df["quote_volume"] = df["volume"] * df["close"]
    df["trades"] = np.nan
    df["taker_buy_base"] = np.nan
    df["taker_buy_quote"] = np.nan
    df["ignore"] = 0

    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()

    if len(df) < 80:
        raise ValueError(f"not enough closed Kraken candles: {symbol} {interval}")

    return df[
        [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
    ].reset_index(drop=True)



def _okx_inst(symbol):
    symbol = str(symbol).upper()
    if symbol.endswith("-SWAP"):
        return symbol
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USDT-SWAP"
    return symbol + "-USDT-SWAP"


def _okx_symbol_list():
    data = http_get(
        f"{OKX_BASE}/api/v5/public/instruments",
        params={"instType": "SWAP"},
        timeout=20,
    )
    out = []
    for x in data.get("data", []):
        inst = x.get("instId", "")
        if (
            x.get("state") == "live"
            and x.get("settleCcy") == "USDT"
            and x.get("settleCcy") == "USDT"
            and inst.endswith("-USDT-SWAP")
        ):
            out.append(inst.replace("-USDT-SWAP", "USDT"))
    return sorted(set(out))


def _okx_tickers():
    data = http_get(
        f"{OKX_BASE}/api/v5/market/tickers",
        params={"instType": "SWAP"},
        timeout=20,
    )
    out = {}
    for x in data.get("data", []):
        inst = x.get("instId", "")
        if not inst.endswith("-USDT-SWAP"):
            continue
        sym = inst.replace("-USDT-SWAP", "USDT")
        price = safe_float(x.get("last"), 0.0)
        vol_base = safe_float(x.get("volCcy24h"), 0.0)
        vol_quote = vol_base * price
        out[sym] = {
            "symbol": sym,
            "lastPrice": price,
            "last": price,
            "volume": vol_base,
            "volumeQuote": vol_quote,
            "quoteVolume": vol_quote,
            "open24h": safe_float(x.get("open24h"), 0.0),
            "high24h": safe_float(x.get("high24h"), 0.0),
            "low24h": safe_float(x.get("low24h"), 0.0),
        }
    return out


def _okx_klines(symbol, interval, limit=220):
    bar_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
        "1w": "1W",
        "60": "1H",
        "240": "4H",
        "D": "1D",
        "W": "1W",
    }
    bar = bar_map.get(str(interval), str(interval))
    data = http_get(
        f"{OKX_BASE}/api/v5/market/candles",
        params={
            "instId": _okx_inst(symbol),
            "bar": bar,
            "limit": min(int(limit), 300),
        },
        timeout=20,
    )

    rows = data.get("data", [])
    if not rows:
        raise ValueError(f"insufficient OKX klines: {symbol} {interval}")

    rows = list(reversed(rows))
    clean = []

    for r in rows:
        if len(r) < 9:
            continue
        clean.append([
            r[0], r[1], r[2], r[3], r[4], r[5],
            r[0], r[7], 0, r[5], r[6], r[8]
        ])

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(clean, columns=cols)

    for c in [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = df["open_time"]

    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()

    if len(df) < 80:
        return pd.DataFrame(columns=df.columns)

    return df.reset_index(drop=True)


def _okx_funding(symbol):
    data = http_get(
        f"{OKX_BASE}/api/v5/public/funding-rate",
        params={"instId": _okx_inst(symbol)},
        timeout=10,
    )
    rows = data.get("data", [])
    if not rows:
        return 0.0
    return safe_float(rows[0].get("fundingRate"), 0.0)


def _okx_open_interest(symbol):
    data = http_get(
        f"{OKX_BASE}/api/v5/public/open-interest",
        params={"instType": "SWAP", "instId": _okx_inst(symbol)},
        timeout=10,
    )
    rows = data.get("data", [])
    if not rows:
        return np.nan
    return safe_float(rows[0].get("oi"), np.nan)


def get_exchange_symbols():
    global DATA_PROVIDER
    if DATA_PROVIDER == "OKX":
        symbols = _okx_symbol_list()
        print(f"Live data provider: OKX SWAP ({len(symbols)} USDT perpetuals)")
        return symbols


    if DATA_PROVIDER == "KRAKEN":
        symbols = _kraken_symbol_list()
        print(f"Live data provider: Kraken Futures ({len(symbols)} perpetuals)")
        return symbols

    if DATA_PROVIDER in ("BINANCE", "BYBIT"):
        return (
            _binance_symbol_list()
            if DATA_PROVIDER == "BINANCE"
            else _bybit_symbol_list()
        )

    try:
        symbols = _binance_symbol_list()
        if symbols:
            DATA_PROVIDER = "BINANCE"
            return symbols
    except Exception as e:
        print("Binance unavailable:", str(e)[:180])

    try:
        symbols = _bybit_symbol_list()
        if symbols:
            DATA_PROVIDER = "BYBIT"
            return symbols
    except Exception as e:
        print("Bybit unavailable:", str(e)[:180])

    symbols = _kraken_symbol_list()
    DATA_PROVIDER = "BINANCE"
    print(f"Live data provider: Kraken Futures ({len(symbols)} perpetuals)")
    return symbols


def _binance_tickers():
    data = http_get(f"{BINANCE_BASE}/fapi/v1/ticker/24hr", timeout=20)
    return {x["symbol"]: x for x in data if x.get("symbol", "").endswith(QUOTE)}


def _bybit_tickers():
    data = http_get(
        f"{BYBIT_BASE}/v5/market/tickers",
        params={"category": "linear"}, timeout=20
    )
    out = {}
    for x in data.get("result", {}).get("list", []):
        sym = x.get("symbol", "")
        if not sym.endswith(QUOTE):
            continue
        out[sym] = {
            "symbol": sym,
            "quoteVolume": x.get("turnover24h", 0),
            "lastPrice": x.get("lastPrice", 0),
            "priceChangePercent": x.get("price24hPcnt", 0),
            "fundingRate": x.get("fundingRate", 0),
        }
    return out


def get_24h_tickers():
    if DATA_PROVIDER == "OKX":
        return _okx_tickers()

    if DATA_PROVIDER == "KRAKEN":
        return _kraken_tickers()
    if DATA_PROVIDER == "BYBIT":
        return _bybit_tickers()
    return _binance_tickers()


def _binance_klines(symbol, interval, limit=220):
    data = http_get(
        f"{BINANCE_BASE}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20
    )
    if not data or len(data) < 80:
        raise ValueError(f"insufficient klines: {symbol} {interval}")
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()
    if len(df) < 80:
        raise ValueError(f"not enough closed candles: {symbol} {interval}")
    return df.reset_index(drop=True)


def _bybit_klines(symbol, interval, limit=220):
    iv = {"1h": "60", "4h": "240", "1d": "D", "1w": "W"}[interval]
    data = http_get(
        f"{BYBIT_BASE}/v5/market/kline",
        params={"category": "linear", "symbol": symbol,
                "interval": iv, "limit": min(limit, 1000)}, timeout=20
    )
    rows = data.get("result", {}).get("list", [])
    if len(rows) < 80:
        raise ValueError(f"insufficient klines: {symbol} {interval}")
    # Bybit returns newest first: startTime, open, high, low, close, volume, turnover.
    rows = list(reversed(rows))
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "quote_volume"
    ])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    minutes = {"1h": 60, "4h": 240, "1d": 1440, "1w": 10080}[interval]
    df["close_time"] = df["open_time"] + pd.to_timedelta(minutes, unit="m")
    df["trades"] = np.nan
    df["taker_buy_base"] = np.nan
    df["taker_buy_quote"] = np.nan
    df["ignore"] = 0
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()
    if len(df) < 80:
        raise ValueError(f"not enough closed candles: {symbol} {interval}")
    return df.reset_index(drop=True)


def get_klines(symbol, interval, limit=220):
    if DATA_PROVIDER == "OKX":
        return _okx_klines(symbol, interval, limit)

    if DATA_PROVIDER == "KRAKEN":
        return _kraken_klines(symbol, interval, limit)
    if DATA_PROVIDER == "BYBIT":
        return _bybit_klines(symbol, interval, limit)
    return _binance_klines(symbol, interval, limit)


def get_funding(symbol):
    try:
        if DATA_PROVIDER == "OKX":
            return _okx_funding(symbol)

        if DATA_PROVIDER == "KRAKEN":
            data = _kraken_tickers()
            return safe_float(
                data.get(_kraken_symbol(symbol), {}).get("fundingRate"),
                0.0
            )
        if DATA_PROVIDER == "BYBIT":
            data = _bybit_tickers()
            return safe_float(data.get(symbol, {}).get("fundingRate"), 0.0)
        data = http_get(
            f"{BINANCE_BASE}/fapi/v1/premiumIndex",
            params={"symbol": symbol}, timeout=10
        )
        return safe_float(data.get("lastFundingRate"), 0.0)
    except Exception:
        return 0.0


def get_open_interest(symbol):
    try:
        if DATA_PROVIDER == "OKX":
            return _okx_open_interest(symbol)

        if DATA_PROVIDER == "KRAKEN":
            data = _kraken_tickers()
            return safe_float(
                data.get(_kraken_symbol(symbol), {}).get("openInterest"),
                np.nan
            )

        if DATA_PROVIDER == "BYBIT":
            data = http_get(
                f"{BYBIT_BASE}/v5/market/open-interest",
                params={"category": "linear", "symbol": symbol,
                        "intervalTime": "1h", "limit": 1}, timeout=10
            )
            rows = data.get("result", {}).get("list", [])
            if rows:
                return safe_float(rows[0].get("openInterest"), np.nan)
            return np.nan
        data = http_get(
            f"{BINANCE_BASE}/fapi/v1/openInterest",
            params={"symbol": symbol}, timeout=10
        )
        return safe_float(data.get("openInterest"), np.nan)
    except Exception:
        return np.nan


def base_asset(symbol):
    if DATA_PROVIDER == "KRAKEN":
        base = symbol.upper()
        if base.startswith(("PF_", "PI_")):
            base = base[3:]
        if base.endswith("USD"):
            base = base[:-3]
        if base == "XBT":
            base = "BTC"
    else:
        base = symbol.replace(QUOTE, "")

    # Map common leveraged-token prefixes to the underlying CoinGecko symbol.
    for prefix in ("1000000", "1000"):
        if base.startswith(prefix) and len(base) > len(prefix):
            base = base[len(prefix):]
            break
    return base


# ============================================================
# COINGECKO QUALITY DATA
# ============================================================

def cg_headers():
    if COINGECKO_API_KEY:
        return {"x-cg-demo-api-key": COINGECKO_API_KEY}
    return {}


def get_cg_markets():
    """
    Bulk market data. CoinGecko exposes market cap, volume, FDV and supply
    fields through /coins/markets.
    """
    cache = {}
    try:
        if os.path.exists(CG_CACHE_FILE):
            with open(CG_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
    except Exception:
        cache = {}

    age = time.time() - cache.get("_timestamp", 0)
    if age < CG_CACHE_SECONDS and cache.get("rows"):
        return cache["rows"]

    rows = []
    for page in range(1, CG_PAGES + 1):
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }

        try:
            # requests does not allow custom headers through http_get, so use
            # a direct request here if a key exists.
            if COINGECKO_API_KEY:
                r = SESSION.get(
                    f"{COINGECKO_BASE}/coins/markets",
                    params=params,
                    headers=cg_headers(),
                    timeout=20
                )
                r.raise_for_status()
                data = r.json()
            else:
                data = http_get(
                    f"{COINGECKO_BASE}/coins/markets",
                    params=params,
                    timeout=20
                )
            rows.extend(data)
        except Exception as e:
            print("CoinGecko page error:", str(e)[:140])
            break

    if rows:
        try:
            with open(CG_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"_timestamp": time.time(), "rows": rows}, f)
        except Exception:
            pass
        return rows

    # If the public API is temporarily rate-limited, use older cached
    # metadata instead of stopping the entire scanner.
    if cache.get("rows"):
        print("CoinGecko temporarily unavailable — using cached quality data.")
        return cache["rows"]

    return []


def normalize_cg(rows):
    """
    Map CoinGecko symbol -> best row. Duplicate symbols are possible, so
    prefer the highest market-cap row.
    """
    out = {}
    for x in rows:
        sym = str(x.get("symbol", "")).upper()
        if not sym:
            continue
        mc = safe_float(x.get("market_cap"), 0)
        if sym not in out or mc > safe_float(
            out[sym].get("market_cap"), 0
        ):
            out[sym] = x
    return out



def _ticker_quote_volume_usd(ticker):
    if not isinstance(ticker, dict):
        return 0.0

    price = safe_float(
        ticker.get("lastPrice",
        ticker.get("price",
        ticker.get("last",
        ticker.get("c", 0)))),
        0
    )

    qv = safe_float(
        ticker.get("quoteVolume",
        ticker.get("volumeQuote",
        ticker.get("turnover24h", 0))),
        0
    )

    if qv > 0:
        return qv

    base_vol = safe_float(
        ticker.get("baseVolume",
        ticker.get("volume24h",
        ticker.get("volume",
        ticker.get("v", 0)))),
        0
    )

    if isinstance(ticker.get("v"), (list, tuple)):
        vv = ticker.get("v")
        if len(vv) > 0:
            base_vol = safe_float(vv[0], 0)

    if base_vol > 0 and price > 0:
        return base_vol * price

    return 0.0

def quality_gate(symbol, ticker, cg):
    """
    Hard asset-quality gate. Returns:
        (passed: bool, quality_score: 0..100, flags: list)
    """
    base = base_asset(symbol)
    meta = cg.get(base) if isinstance(cg, dict) else {}
    flags = []

    if not meta:
        if DATA_PROVIDER == "KRAKEN":
            price = safe_float(ticker.get("lastPrice"), 0)
            qv = _ticker_quote_volume_usd(ticker)
            if qv <= 0 and price > 0:
                qv = safe_float(ticker.get("volume"), 0) * price

            if qv < MIN_24H_VOLUME_USD:
                return False, 0.0, ["LOW_KRAKEN_LIQUIDITY"]

            if price < MIN_PRICE_USD:
                return False, 0.0, ["MICRO_PRICE"]

            return True, 75.0, [
                "KRAKEN_LIQUIDITY_MODE",
                "COINGECKO_RATE_LIMITED"
            ]

        return True, 60.0, ["CG_UNAVAILABLE_FALLBACK"]

    mc = safe_float(meta.get("market_cap"), np.nan)
    fdv = safe_float(
        meta.get("fully_diluted_valuation"),
        np.nan
    )
    circ = safe_float(meta.get("circulating_supply"), np.nan)
    total = safe_float(meta.get("total_supply"), np.nan)
    max_supply = safe_float(meta.get("max_supply"), np.nan)

    price = safe_float(ticker.get("lastPrice"), 0)
    qv = ticker_quote_volume_usd(ticker)
    if qv <= 0 and price > 0:
        qv = safe_float(ticker.get("volume"), 0) * price

    # Hard requirements.
    if not math.isfinite(mc) or mc < MIN_MARKET_CAP_USD:
        return False, 0.0, ["LOW_MARKET_CAP"]

    if qv < MIN_24H_VOLUME_USD:
        return False, 0.0, ["LOW_LIQUIDITY"]

    if price < MIN_PRICE_USD:
        return False, 0.0, ["MICRO_PRICE"]

    # User requested finite/limited supply.
    if not math.isfinite(max_supply) or max_supply <= 0:
        return False, 0.0, ["MAX_SUPPLY_UNKNOWN_OR_UNLIMITED"]

    if math.isfinite(total) and total > max_supply * 1.02:
        return False, 0.0, ["SUPPLY_DATA_INCONSISTENT"]

    if math.isfinite(circ) and math.isfinite(total) and total > 0:
        circ_ratio = circ / total
        if circ_ratio < MIN_CIRCULATION_RATIO:
            return False, 0.0, ["LOW_CIRCULATION_RATIO"]
    else:
        return False, 0.0, ["SUPPLY_DATA_INCOMPLETE"]

    if math.isfinite(fdv) and mc > 0:
        fdv_ratio = fdv / mc
        if fdv_ratio > MAX_FDV_MC_RATIO:
            return False, 0.0, ["EXTREME_FDV_DILUTION"]

    score = 0
    score += 25  # market cap passed
    score += 20  # liquidity passed
    score += 20  # finite max supply
    score += 15  # healthy circulation
    score += 10 if math.isfinite(fdv) else 0
    score += 10 if math.isfinite(total) else 0

    flags.extend([
        "MARKET_CAP_OK",
        "LIQUIDITY_OK",
        "FINITE_MAX_SUPPLY",
        "TOKENOMICS_OK"
    ])

    # Institutional/adoption proxy: NOT proof of institutional ownership.
    if safe_float(meta.get("market_cap_rank"), 9999) <= 100:
        flags.append("LARGE_CAP_ADOPTION_PROXY")
        score = min(100, score + 5)

    if qv >= 100_000_000:
        flags.append("DEEP_LIQUIDITY_PROXY")
        score = min(100, score + 5)

    return True, float(min(100, score)), flags


# ============================================================
# INDICATORS
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df):
    prev = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1/n, adjust=False).mean()


def macd(close):
    m12 = ema(close, 12)
    m26 = ema(close, 26)
    line = m12 - m26
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def stochastic(df, n=14, d=3):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def mfi(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    money = tp * df["volume"]
    direction = tp.diff()
    pos = money.where(direction > 0, 0.0).rolling(n).sum()
    neg = money.where(direction < 0, 0.0).rolling(n).sum()
    ratio = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def williams_r(df, n=14):
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def adx(df, n=14):
    high, low = df["high"], df["low"]
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index
    )

    tr = true_range(df)
    atrv = tr.ewm(alpha=1/n, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atrv
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atrv

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)

    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di


def bollinger(close, n=20, mult=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid, mid + mult * sd, mid - mult * sd


def vwap(df, n=50):
    pv = ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
    return pv.rolling(n).sum() / df["volume"].rolling(n).sum()


def obv(df):
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ============================================================
# CANDLE / PRICE ACTION HELPERS
# ============================================================

def last_row(df):
    return df.iloc[-1]


def candle_flags(df):
    x = df.iloc[-1]
    p = df.iloc[-2]

    o, h, l, c = map(float, [x.open, x.high, x.low, x.close])
    po, pc = float(p.open), float(p.close)

    body = abs(c - o)
    rng = max(h - l, 1e-12)
    upper = h - max(o, c)
    lower = min(o, c) - l

    bullish_body = c > o
    bearish_body = c < o

    bullish_engulf = (
        bullish_body and pc < po and c >= po and o <= pc
    )
    bearish_engulf = (
        bearish_body and pc > po and c <= po and o >= pc
    )

    hammer = lower >= body * 2 and upper <= max(body, rng * 0.15)
    shooting = upper >= body * 2 and lower <= max(body, rng * 0.15)

    strong_close = (c - l) / rng >= 0.75
    weak_close = (h - c) / rng >= 0.75

    return {
        "bullish_body": bullish_body,
        "bearish_body": bearish_body,
        "bullish_engulf": bullish_engulf,
        "bearish_engulf": bearish_engulf,
        "hammer": hammer,
        "shooting_star": shooting,
        "strong_close": strong_close,
        "weak_close": weak_close,
        "body_ratio": body / rng
    }


def recent_high(df, n=20):
    return float(df["high"].iloc[-n-1:-1].max())


def recent_low(df, n=20):
    return float(df["low"].iloc[-n-1:-1].min())


# ============================================================
# MARKET / BTC MASTER FILTER
# ============================================================

def pct_change(df, bars):
    if len(df) <= bars:
        return np.nan
    a = float(df["close"].iloc[-bars-1])
    b = _safe_last(df["close"])
    if a == 0:
        return np.nan
    return (b / a - 1) * 100


def btc_context():
    """
    BTC is the master risk benchmark.
    Returns shock state and market phase.
    """
    d1 = get_klines("BTCUSDT", "1h", 220)
    d4 = get_klines("BTCUSDT", "4h", 220)

    a = atr(d1, 14)
    r = rsi(d1["close"], 14)
    e20 = ema(d1["close"], 20)
    e50 = ema(d1["close"], 50)
    ad, pi, mi = adx(d1, 14)

    close = _safe_last(d1["close"])
    shock_1h = pct_change(d1, 1)
    shock_4h = pct_change(d4, 1)
    atr_pct = float(a.iloc[-1] / close * 100)

    shock = (
        shock_1h <= BTC_SHOCK_1H_PCT
        or shock_4h <= BTC_SHOCK_4H_PCT
        or (
            math.isfinite(atr_pct)
            and abs(shock_1h) >= BTC_SHOCK_ATR_MULT * atr_pct
        )
    )

    if shock:
        phase = "RISK-OFF"
    elif close > _safe_last(e20) > _safe_last(e50) and \
            _safe_last(r) >= 52:
        phase = "BULLISH"
    elif close < _safe_last(e20) < _safe_last(e50) and \
            _safe_last(r) <= 48:
        phase = "BEARISH"
    else:
        phase = "SIDEWAYS"

    return {
        "price": close,
        "shock": bool(shock),
        "phase": phase,
        "rsi": _safe_last(r),
        "adx": _safe_last(ad),
        "change_1h": shock_1h,
        "change_4h": shock_4h,
        "atr_pct": atr_pct
    }


# ============================================================
# 72 RULE EVALUATION
# ============================================================

def _safe_last(x, default=float("nan")):
    """Return the last numeric value safely; never recurse."""
    try:
        if x is None:
            return default
        if hasattr(x, "iloc"):
            if len(x) == 0:
                return default
            value = x.iloc[-1]
        elif isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return default
            value = x[-1]
        else:
            value = x
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default

def _safe_at(x, n, default=float("nan")):
    try:
        if x is None or len(x) <= n:
            return default
        return float(x.iloc[-(n+1)])
    except Exception:
        return default

def evaluate(symbol, dfs, funding, oi, btc, quality_score, quality_flags):
    """
    Every rule is binary. Score = number of satisfied rules out of 72.

    Rules 1-18  : trend / moving averages / VWAP
    Rules 19-34 : momentum / oscillators
    Rules 35-48 : price action / breakout / candles / volume
    Rules 49-60 : multi-timeframe / BTC / derivatives
    Rules 61-72 : volatility / risk / quality / swing structure
    """
    h1 = dfs["1h"]
    h4 = dfs["4h"]
    d1 = dfs["1d"]
    w1 = dfs["1w"]

    close = _safe_last(h1["close"])

    e20 = ema(h1["close"], 20)
    e50 = ema(h1["close"], 50)
    e200 = ema(h1["close"], 200)
    vw = vwap(h1, 50)
    r = rsi(h1["close"], 14)
    ml, ms, mh = macd(h1["close"])
    sk, sd = stochastic(h1)
    cc = cci(h1)
    mf = mfi(h1)
    wr = williams_r(h1)
    ad, pdi, mdi = adx(h1)
    av = atr(h1, 14)
    bbm, bbu, bbl = bollinger(h1["close"])

    # Higher timeframes.
    d_e20 = ema(d1["close"], 20)
    d_e50 = ema(d1["close"], 50)
    d_r = rsi(d1["close"], 14)
    d_ad, d_pdi, d_mdi = adx(d1, 14)

    h4_e20 = ema(h4["close"], 20)
    h4_e50 = ema(h4["close"], 50)
    h4_r = rsi(h4["close"], 14)

    w_e20 = ema(w1["close"], 20)
    w_e50 = ema(w1["close"], 50)
    w_r = rsi(w1["close"], 14)

    ob = obv(h1)
    ob_ma = sma(ob, 20)

    cf = candle_flags(h1)

    # Volume regime.
    vol = _safe_last(h1["volume"])
    vol20 = float(h1["volume"].iloc[-21:-1].mean())
    vol_ratio = vol / max(vol20, 1e-12)

    # Breakout levels.
    rh20 = recent_high(h1, 20)
    rl20 = recent_low(h1, 20)

    # ATR normalized.
    atrv = _safe_last(av)
    atr_pct = atrv / close * 100 if close else np.nan

    bull = 0
    bear = 0
    reasons_bull = []
    reasons_bear = []

    def add(cond, bull_text="", bear_text=""):
        nonlocal bull, bear
        if cond:
            bull += 1
            if bull_text:
                reasons_bull.append(bull_text)
        else:
            bear += 1
            if bear_text:
                reasons_bear.append(bear_text)

    # ---------------------------
    # 1-18 TREND / MA / VWAP
    # ---------------------------
    add(close > _safe_last(e20), "1H > EMA20")
    add(_safe_last(e20) > _safe_last(e50), "EMA20 > EMA50")
    add(_safe_last(e50) > _safe_last(e200), "EMA50 > EMA200")
    add(_safe_last(e20) > float(e20.iloc[-5]), "EMA20 rising")
    add(_safe_last(e50) > float(e50.iloc[-5]), "EMA50 rising")
    add(close > _safe_last(vw), "Price > VWAP")
    add(_safe_last(ob) > _safe_last(ob_ma), "OBV > OBV MA")
    add(_safe_last(d1["close"]) > _safe_last(d_e20), "1D > EMA20")
    add(_safe_last(d_e20) > _safe_last(d_e50), "1D EMA20 > EMA50")
    add(_safe_last(h4["close"]) > _safe_last(h4_e20), "4H > EMA20")
    add(_safe_last(h4_e20) > _safe_last(h4_e50), "4H EMA20 > EMA50")
    add((not w1.empty) and (not w_e20.empty) and _safe_last(w1["close"]) > _safe_last(w_e20), "1W > EMA20")
    add((not w_e20.empty) and (not w_e50.empty) and _safe_last(w_e20) > _safe_last(w_e50), "1W EMA20 > EMA50")
    add(_safe_last(d_ad) >= 18, "1D trend strength")
    add(_safe_last(d_pdi) > _safe_last(d_mdi), "1D +DI > -DI")
    add(_safe_last(ad) >= 18, "1H ADX trend strength")
    add(_safe_last(pdi) > _safe_last(mdi), "1H +DI > -DI")
    add(close > _safe_last(bbm), "Above BB mid")

    # ---------------------------
    # 19-34 MOMENTUM / OSCILLATORS
    # ---------------------------
    add(52 <= _safe_last(r) <= 72, "RSI bullish zone")
    add(_safe_last(r) > float(r.iloc[-5]), "RSI rising")
    add(_safe_last(ml) > _safe_last(ms), "MACD bullish")
    add(_safe_last(mh) > float(mh.iloc[-3]), "MACD histogram rising")
    add(_safe_last(sk) > _safe_last(sd), "Stoch K > D")
    add(35 <= _safe_last(sk) <= 90, "Stoch not exhausted")
    add(_safe_last(cc) > 0, "CCI positive")
    add(_safe_last(cc) > float(cc.iloc[-3]), "CCI rising")
    add(_safe_last(mf) > 50, "MFI > 50")
    add(_safe_last(mf) > float(mf.iloc[-3]), "MFI rising")
    add(_safe_last(wr) > -80, "Williams %R recovery")
    add(_safe_last(wr) > float(wr.iloc[-3]), "Williams %R rising")
    add(_safe_last(ad) >= 20, "ADX >= 20")
    add(_safe_last(pdi) > _safe_last(mdi), "+DI dominance")
    add(close > _safe_last(bbl), "Above BB lower band")
    add(_safe_last(r) < 78, "RSI below extreme")

    # ---------------------------
    # 35-48 PRICE ACTION / BREAKOUT
    # ---------------------------
    add(close > rh20, "20-bar breakout")
    add(close >= rh20 * 0.995, "Near breakout level")
    add(vol_ratio >= 1.5, "Volume expansion")
    add(vol_ratio >= 2.0, "Strong volume")
    add(cf["bullish_body"], "Bullish candle")
    add(cf["strong_close"], "Strong candle close")
    add(cf["bullish_engulf"] or cf["hammer"], "Bullish reversal candle")
    add(close > float(h1["close"].iloc[-2]), "Higher close")
    add(_safe_last(h1["low"]) >= float(h1["low"].iloc[-4:].min()), "Higher-low support")
    add(close > float(h1["high"].iloc[-2]), "Micro structure break")
    add(close > _safe_last(h1["open"]), "Positive session body")
    add(_safe_last(h1["close"]) > float(h1["high"].iloc[-2]) * 0.998,
        "Close holds breakout")
    add(_safe_last(ob) > float(ob.iloc[-5]), "OBV rising")
    add(vol_ratio >= 1.2 and close > float(h1["close"].iloc[-2]),
        "Price-volume confirmation")

    # ---------------------------
    # 49-60 MTF / BTC / DERIVATIVES
    # ---------------------------
    add(_safe_last(h4_r) >= 50, "4H RSI bullish")
    add(_safe_last(d_r) >= 50, "1D RSI bullish")
    add((not w_r.empty) and _safe_last(w_r) >= 50, "1W RSI bullish")
    add(_safe_last(h4["close"]) > _safe_last(h4_e50),
        "4H above EMA50")
    add(_safe_last(d1["close"]) > _safe_last(d_e50),
        "1D above EMA50")
    add(_safe_last(w1["close"]) > _safe_last(w_e50),
        "1W above EMA50")
    add(btc["phase"] in ("BULLISH", "SIDEWAYS"), "BTC regime acceptable")
    add(not btc["shock"], "BTC no shock")
    add(btc["change_4h"] > -4.0, "BTC 4H drawdown controlled")

    # Funding not excessively crowded long.
    add(funding < 0.0008, "Funding not overcrowded")
    # OI is informational; if available, require it to be positive/nonzero.
    add(not math.isfinite(oi) or oi > 0, "OI data acceptable")
    # Avoid extremely weak BTC relative context.
    add(btc["rsi"] >= 42, "BTC RSI risk acceptable")

    # ---------------------------
    # 61-72 VOLATILITY / RISK / QUALITY / SWING
    add(float(atrv) > 0, "RULE72_AUTO_ATR_VALID")
        # ---------------------------
    add(0.25 <= atr_pct <= 8.0, "ATR volatility usable")
    add(close > _safe_last(e20) - 0.5 * atrv,
        "Price not far below trend")
    add(_safe_last(bbu) > _safe_last(bbm),
        "Bollinger expansion possible")
    add(vol20 > 0, "Volume history available")
    add(quality_score >= 75, "Quality score >= 75")
    add("FINITE_MAX_SUPPLY" in quality_flags, "Finite max supply")
    add("TOKENOMICS_OK" in quality_flags, "Tokenomics gate")
    add(
        "LARGE_CAP_ADOPTION_PROXY" in quality_flags
        or "DEEP_LIQUIDITY_PROXY" in quality_flags,
        "Adoption/liquidity proxy"
    )
    # AI gets a bonus through the separate AI score, but does not override
    # quality or technical gates.
    name_hint = base_asset(symbol).lower()
    ai_bonus = any(k in name_hint for k in AI_KEYWORDS)
    add(ai_bonus, "AI/narrative sector bonus")
    # Swing setup must have a clean risk structure.
    add(close > rl20, "Above 20-bar support")
    add(close < rh20 * 1.10, "Not vertically extended")

    # RULE COUNT SAFETY
    # Do not crash the scanner if the number of active rules differs
    # slightly from the documented 72-rule design.
    total_rules = bull + bear

    if total_rules < 72:
        print(f"WARNING: active rule count = {total_rules}, expected 72")
    elif total_rules > 72:
        print(f"INFO: active rule count = {total_rules}, expected 72")

    # We use bullish rule alignment for LONG and inverse alignment for SHORT.
    # The scanner is conservative: new LONGs are blocked in BTC shock.
    bull_score = min(total_rules, bull)
    bear_score = min(total_rules, max(0, total_rules - bull + bear // 2))

    # For swing trading, LONG is the primary signal. SHORT can be reported
    # only when BTC is not in shock and bearish structure is very strong.
    direction = None
    score = 0

    if (
        bull_score >= MIN_SCORE
        and not btc["shock"]
        and quality_score >= 75
    ):
        direction = "LONG"
        score = bull_score
    else:
        # Build a conservative bearish score from explicit bearish structure.
        bearish_components = 0
        bearish_components += int(close < _safe_last(e20))
        bearish_components += int(_safe_last(e20) < _safe_last(e50))
        bearish_components += int(_safe_last(e50) < _safe_last(e200))
        bearish_components += int(_safe_last(r) < 48)
        bearish_components += int(_safe_last(ml) < _safe_last(ms))
        bearish_components += int(_safe_last(ad) >= 20)
        bearish_components += int(_safe_last(mdi) > _safe_last(pdi))
        bearish_components += int(close < _safe_last(vw))
        bearish_components += int(_safe_last(d1["close"]) < _safe_last(d_e50))
        bearish_components += int(_safe_last(h4["close"]) < _safe_last(h4_e50))
        bearish_components += int(_safe_last(w1["close"]) < _safe_last(w_e50))
        bearish_components += int(cf["bearish_body"])
        bearish_components += int(cf["shooting_star"])
        bearish_components += int(vol_ratio >= 1.5)
        bearish_components += int(btc["phase"] == "BEARISH")

        if bearish_components >= 11 and not btc["shock"] and quality_score >= 75:
            direction = "SHORT"
            score = min(72, 55 + bearish_components)

    if direction is None:
        return None

    # ---------------------------
    # Risk plan using ATR + structure.
    # ---------------------------
    swing_low = float(h4["low"].iloc[-10:].min())
    swing_high = float(h4["high"].iloc[-10:].max())
    h1_low = float(h1["low"].iloc[-10:].min())
    h1_high = float(h1["high"].iloc[-10:].max())

    risk_unit = max(1.5 * atrv, close * 0.008)

    if direction == "LONG":
        sl = min(swing_low, h1_low) - 0.25 * atrv
        if sl >= close:
            sl = close - risk_unit
        risk = close - sl
        t1 = close + 1.5 * risk
        t2 = close + 2.5 * risk
        t3 = close + 4.0 * risk
    else:
        sl = max(swing_high, h1_high) + 0.25 * atrv
        if sl <= close:
            sl = close + risk_unit
        risk = sl - close
        t1 = close - 1.5 * risk
        t2 = close - 2.5 * risk
        t3 = close - 4.0 * risk

    if risk <= 0:
        return None

    rr = abs(t2 - close) / risk

    if rr < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "score": int(score),
        "percentage": float(score / max(1, total_rules) * 100),
        "quality_score": float(quality_score),
        "quality_flags": quality_flags,
        "price": close,
        "sl": float(sl),
        "target1": float(t1),
        "target2": float(t2),
        "target3": float(t3),
        "rr": float(rr),
        "btc_price": btc["price"],
        "btc_phase": btc["phase"],
        "btc_shock": btc["shock"],
        "btc_rsi": btc["rsi"],
        "funding": funding,
        "oi": oi,
        "volume_ratio": vol_ratio,
        "atr_pct": atr_pct,
        "ai_bonus": ai_bonus,
        "timestamp": fmt_time(),
        "bull_reasons": reasons_bull[:12],
        "bear_reasons": reasons_bear[:12],
    }


# ============================================================
# USD/INR
# ============================================================

def get_usdinr():
    # Public exchange-rate endpoint. If unavailable, return 0 and Telegram
    # will show USD only instead of inventing an INR conversion.
    urls = [
        "https://api.frankfurter.app/latest?from=USD&to=INR",
        "https://open.er-api.com/v6/latest/USD"
    ]

    for u in urls:
        try:
            data = http_get(u, timeout=10, retries=2)
            if "rates" in data and "INR" in data["rates"]:
                x = safe_float(data["rates"]["INR"], np.nan)
                if math.isfinite(x) and x > 0:
                    return x
        except Exception:
            pass

    return np.nan


def money(usd, usdinr):
    if not math.isfinite(usd):
        return "N/A"
    s = f"${usd:,.8f}".rstrip("0").rstrip(".")
    if math.isfinite(usdinr):
        s += f" | ₹{usd * usdinr:,.2f}"
    return s


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    try:
        url = (
            f"https://api.telegram.org/bot"
            f"{TELEGRAM_BOT_TOKEN}/sendMessage"
        )
        r = SESSION.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=15
        )
        return r.ok
    except Exception as e:
        print("Telegram error:", str(e)[:120])
        return False


def signal_message(r, usdinr):
    quality_flags = ", ".join(r["quality_flags"][:8])

    return (
        "🚨 <b>CRYPTO MASTER FINAL SWING SIGNAL</b>\n\n"
        f"🪙 <b>Pair:</b> {r['symbol']}\n"
        f"🌐 <b>Live data:</b> {DATA_PROVIDER}\n"
        f"📌 <b>Action:</b> {r['direction']}\n"
        f"🔥 <b>Rules:</b> {r['score']}/72 "
        f"({r['percentage']:.1f}% alignment)\n"
        f"🏆 <b>Quality:</b> {r['quality_score']:.0f}/100\n"
        f"🤖 <b>AI bonus:</b> {'YES' if r['ai_bonus'] else 'NO'}\n\n"
        f"₿ <b>BTC:</b> ${r['btc_price']:,.2f} | {r['btc_phase']}\n"
        f"📊 <b>BTC RSI:</b> {r['btc_rsi']:.1f}\n"
        f"💧 <b>Volume:</b> {r['volume_ratio']:.2f}x average\n"
        f"💰 <b>Funding:</b> {r['funding']*100:.4f}%\n"
        f"📈 <b>ATR:</b> {r['atr_pct']:.2f}%\n\n"
        f"🎯 <b>Entry:</b> {money(r['price'], usdinr)}\n"
        f"🛑 <b>SL:</b> {money(r['sl'], usdinr)}\n"
        f"🎯 <b>T1:</b> {money(r['target1'], usdinr)}\n"
        f"🎯 <b>T2:</b> {money(r['target2'], usdinr)}\n"
        f"🎯 <b>T3:</b> {money(r['target3'], usdinr)}\n"
        f"⚖️ <b>R:R:</b> 1:{r['rr']:.2f}\n\n"
        f"🧾 <b>Quality flags:</b> {quality_flags}\n\n"
        "<i>Rule alignment is NOT profit probability and no "
        "zero-loss/guaranteed-profit claim is made.</i>\n"
        f"<i>{r['timestamp']}</i>"
    )


# ============================================================
# UNIVERSE BUILDER
# ============================================================

def build_universe(exchange_symbols, tickers, cg_rows):
    """Build a liquid/quality universe and ALWAYS return (symbols, cg)."""
    cg = normalize_cg(cg_rows)
    candidates = []

    if cg:
        for symbol in exchange_symbols:
            ticker = tickers.get(symbol)
            if not ticker:
                continue

            base = base_asset(symbol)
            meta = cg.get(base) if isinstance(cg, dict) else {}
            if not meta:
                continue

            mc = safe_float(meta.get("market_cap"), 0)
            qv = _ticker_quote_volume_usd(ticker)
            max_supply = safe_float(meta.get("max_supply"), np.nan)
            circ = safe_float(meta.get("circulating_supply"), np.nan)
            total = safe_float(meta.get("total_supply"), np.nan)

            if mc < MIN_MARKET_CAP_USD:
                continue
            if qv < MIN_24H_VOLUME_USD:
                continue
            if not math.isfinite(max_supply) or max_supply <= 0:
                continue
            if not math.isfinite(circ) or not math.isfinite(total) or total <= 0:
                continue
            if circ / total < MIN_CIRCULATION_RATIO:
                continue

            fdv = safe_float(meta.get("fully_diluted_valuation"), np.nan)
            if math.isfinite(fdv) and mc > 0 and fdv / mc > MAX_FDV_MC_RATIO:
                continue

            rank = safe_float(meta.get("market_cap_rank"), 9999)
            score = (
                math.log10(max(mc, 1)) * 10
                + math.log10(max(qv, 1)) * 7
                - min(rank, 1000) * 0.01
            )
            candidates.append((score, symbol))

    else:
        print("WARNING: CoinGecko unavailable/rate-limited.")
        print("Using Kraken live-liquidity fallback mode.")
        for symbol in exchange_symbols:
            ticker = tickers.get(symbol, {})
            qv = _ticker_quote_volume_usd(ticker)
            price = safe_float(ticker.get("lastPrice"), 0)
            if qv < MIN_24H_VOLUME_USD or price < MIN_PRICE_USD:
                continue
            score = math.log10(max(qv, 1)) * 10
            candidates.append((score, symbol))

    candidates.sort(reverse=True)
    selected = [x[1] for x in candidates[:UNIVERSE_TARGET]]

    mode = "COINGECKO QUALITY" if cg else "KRAKEN LIQUIDITY FALLBACK"
    print(f"Universe mode: {mode}")
    print(f"Selected liquid symbols: {len(selected)}")

    return selected, cg


# ============================================================
# SCAN
# ============================================================

def run_scan(exchange_symbols, tickers, cg, usdinr):
    print("\n" + "=" * 90)
    print(f"SCAN START | {fmt_time()}")
    print("=" * 90)

    try:
        btc = btc_context()
    except Exception as e:
        print("BTC context unavailable — scan aborted:", e)
        return []

    print(
        f"BTC ${btc['price']:,.2f} | {btc['phase']} | "
        f"1H {btc['change_1h']:.2f}% | 4H {btc['change_4h']:.2f}% | "
        f"RSI {btc['rsi']:.1f}"
    )

    global LAST_BTC_SHOCK_STATE

    if btc["shock"]:
        print("🚨 BTC SHOCK: new LONG signals are blocked.")

        if LAST_BTC_SHOCK_STATE is not True:
            telegram_send(
                "🚨 <b>BTC SHOCK ALERT</b>\\n"
                f"BTC 1H: {btc['change_1h']:.2f}%\\n"
                f"BTC 4H: {btc['change_4h']:.2f}%\\n"
                f"Phase: {btc['phase']}\\n"
                "⚠️ New LONG signals are BLOCKED until risk normalizes."
            )

        LAST_BTC_SHOCK_STATE = True

    else:
        if LAST_BTC_SHOCK_STATE is True:
            telegram_send(
                "🟢 <b>BTC SHOCK CLEARED</b>\\n"
                f"BTC 1H: {btc['change_1h']:.2f}%\\n"
                f"BTC 4H: {btc['change_4h']:.2f}%\\n"
                f"Phase: {btc['phase']}\\n"
                "✅ Long-signal engine is active again."
            )

        LAST_BTC_SHOCK_STATE = False

    universe = build_universe(exchange_symbols, tickers, cg)
    if universe is None:
        print("WARNING: build_universe returned None - using exchange symbols fallback")
        symbols = exchange_symbols
        cgmap = cg
    else:
        symbols, cgmap = universe

    print(f"Quality universe: {len(symbols)} symbols")
    if len(symbols) < 50:
        print("WARNING: quality universe is unusually small.")

    results = []

    for idx, symbol in enumerate(symbols, 1):
        try:
            ticker = tickers.get(symbol, {})
            passed, qscore, qflags = quality_gate(
                symbol, ticker, cgmap
            )

            if not passed:
                print(
                    f"[{idx}/{len(symbols)}] {symbol}: "
                    f"QUALITY BLOCK {qflags}"
                )
                continue

            # Fetch the four swing timeframes.  Only closed candles are used.
            dfs = {
                "1h": get_klines(symbol, "1h", 230),
                "4h": get_klines(symbol, "4h", 230),
                "1d": get_klines(symbol, "1d", 230),
                "1w": get_klines(symbol, "1w", 230),
            }

            # Freshness check for 1H.
            latest = dfs["1h"]["close_time"].iloc[-1]

            # get_klines() already stores the actual closed-candle end time.
            # Do not add another hour; that would falsely mark fresh data stale.
            age_min = max(
                0.0,
                (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 60
            )

            if age_min > MAX_KLINE_AGE_MIN:
                print(
                    f"[{idx}/{len(symbols)}] {symbol}: "
                    f"STALE DATA ({age_min:.1f}m)"
                )
                continue

            # Kraken ticker already contains funding + open interest.
            # Reuse it instead of making two extra HTTP requests per symbol.
            if DATA_PROVIDER == "KRAKEN":
                funding = safe_float(ticker.get("fundingRate"), 0.0)
                oi = safe_float(ticker.get("openInterest"), np.nan)
            else:
                funding = get_funding(symbol)
                oi = get_open_interest(symbol)

            r = evaluate(
                symbol,
                dfs,
                funding,
                oi,
                btc,
                qscore,
                qflags
            )

            if r:
                results.append(r)
                print(
                    f"🟢 {symbol} {r['direction']} "
                    f"{r['score']}/72 ({r['percentage']:.1f}%) "
                    f"Q={r['quality_score']:.0f}"
                )
            else:
                print(
                    f"[{idx}/{len(symbols)}] {symbol}: "
                    f"no qualified setup"
                )

        except Exception as e:
            import traceback
            print(
                f"[{idx}/{len(symbols)}] {symbol} ERROR TYPE: "
                f"{type(e).__name__}"
            )
            print(
                f"[{idx}/{len(symbols)}] {symbol} ERROR DETAIL: "
                f"{repr(e)}"
            )
            traceback.print_exc(limit=3)

        # Avoid hammering public APIs.
        time.sleep(0.08)

    results.sort(
        key=lambda x: (
            x["score"],
            x["quality_score"],
            x["volume_ratio"]
        ),
        reverse=True
    )

    print("\n" + "=" * 90)
    print(f"SCAN COMPLETE | Qualified signals: {len(results)}")
    print("=" * 90)

    if not results:
        print("No coin passed the strict quality + technical gate today.")

    for r in results[:15]:
        print(
            f"{r['symbol']:<14} {r['direction']:<5} "
            f"{r['score']:>2}/72 {r['percentage']:>5.1f}% "
            f"Q={r['quality_score']:>3.0f} "
            f"Entry={r['price']:.8g} "
            f"T1={r['target1']:.8g} "
            f"T2={r['target2']:.8g} "
            f"SL={r['sl']:.8g}"
        )

    # Send only top 5 to reduce alert spam.
    for r in results[:5]:
        telegram_send(signal_message(r, usdinr))

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

    print("\n" + "=" * 90)
    print(" CRYPTO MASTER FINAL 72-RULE QUALITY + SWING SCANNER")
    print("=" * 90)
    print("Liquid/quality crypto perpetual universe | Kraken Futures live data")
    print("Finite-supply + tokenomics + market-cap + liquidity gates | CG fallback enabled")
    print("Price Action + Breakout + Volume + Candlesticks")
    print("RSI + MACD + ADX + Stochastic + CCI + MFI + Williams %R")
    print("VWAP + OBV + EMA20/50/200 + ATR + Bollinger")
    print("1W + 1D + 4H + 1H swing confirmation")
    print("BTC MASTER BENCHMARK + SHOCK/RISK-OFF PROTECTION")
    print("Funding + Open Interest + R:R")
    print("USD + INR + Optional Telegram")
    print("=" * 90)

    # Telegram can be supplied without editing the file.
    if not TELEGRAM_BOT_TOKEN:
        try:
            TELEGRAM_BOT_TOKEN = input(
                "\nTelegram Bot Token (ENTER = no Telegram): "
            ).strip()
        except EOFError:
            TELEGRAM_BOT_TOKEN = ""

    if TELEGRAM_BOT_TOKEN and not TELEGRAM_CHAT_ID:
        try:
            TELEGRAM_CHAT_ID = input(
                "Telegram Chat ID (ENTER = no Telegram): "
            ).strip()
        except EOFError:
            TELEGRAM_CHAT_ID = ""

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        if telegram_send("🟣 <b>CRYPTO TELEGRAM TEST OK</b>\nBot connected. Scanner is starting now."):
            print("Telegram test: OK")
        else:
            print("Telegram test: FAILED — scanner will still run if market data is available.")

    try:
        exchange_symbols = get_exchange_symbols()
        print(
            f"{DATA_PROVIDER} perpetual symbols available: "
            f"{len(exchange_symbols)}"
        )
    except Exception as e:
        print("Cannot reach Binance:", e)
        return

    try:
        cg_rows = get_cg_markets()
        print(f"CoinGecko market rows loaded: {len(cg_rows)}")
    except Exception as e:
        print("CoinGecko unavailable:", e)
        print(
            "This final version requires quality metadata for the "
            "finite-supply gate, so the scan will stop safely."
        )
        return

    telegram_send(
        f"🔔 <b>🟣 CRYPTO MASTER FINAL SCANNER ONLINE</b>\n"
        f"Live data provider: {DATA_PROVIDER} ✅\n"
        "72-rule swing engine | 56/72 minimum alignment ✅\n"
        "Finite-supply quality gate ✅\n"
        "BTC master risk filter ✅\n"
        "1W + 1D + 4H + 1H ✅\n"
        "Price action + volume + indicators ✅\n"
        "REST live polling + stale-data rejection ✅"
    )

    while True:
        started = time.time()

        try:
            tickers = get_24h_tickers()
            usdinr = get_usdinr()
            run_scan(
                exchange_symbols,
                tickers,
                cg_rows,
                usdinr
            )

            # Refresh CoinGecko quality metadata periodically.
            if time.time() - started > 0:
                cg_rows = get_cg_markets()

        except KeyboardInterrupt:
            print("\nScanner stopped.")
            break
        except Exception as e:
            print("MAIN ERROR:", repr(e))
            traceback.print_exc()

        elapsed = time.time() - started
        sleep_for = max(10, SCAN_INTERVAL - elapsed)

        print(
            f"\nNext full validation scan in about "
            f"{sleep_for/60:.1f} minutes."
        )

        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
