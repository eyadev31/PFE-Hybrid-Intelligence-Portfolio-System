"""
Hybrid Intelligence Portfolio System — Market Data Layer
=========================================================
Production-grade market data fetchers using:
  - Binance public API for crypto (exchange-direct, no key required)
  - TwelveData API for equities, ETFs, forex (free tier, key required)

All fetchers return normalized pandas DataFrames with consistent
column naming and timezone-aware timestamps.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config.settings import MarketDataConfig, APIKeys

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
#  DATA CACHE — In-Memory TTL Cache
# ══════════════════════════════════════════════════════
class DataCache:
    """
    Simple in-memory cache with TTL to avoid redundant API calls.
    Production systems would use Redis; this is optimized for MVP.
    """

    def __init__(self, ttl_minutes: int = MarketDataConfig.CACHE_TTL_MINUTES):
        self._store: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._store:
            timestamp, data = self._store[key]
            if datetime.utcnow() - timestamp < self._ttl:
                logger.debug(f"Cache HIT: {key}")
                return data.copy()
            else:
                del self._store[key]
                logger.debug(f"Cache EXPIRED: {key}")
        return None

    def set(self, key: str, data: pd.DataFrame) -> None:
        self._store[key] = (datetime.utcnow(), data.copy())
        logger.debug(f"Cache SET: {key} ({len(data)} rows)")

    def clear(self) -> None:
        self._store.clear()


# Global cache instance
_cache = DataCache()


# ══════════════════════════════════════════════════════
#  BINANCE DATA FETCHER — Crypto
# ══════════════════════════════════════════════════════
class BinanceDataFetcher:
    """
    Fetches cryptocurrency OHLCV data from Binance public API.
    No API key required. Rate limit: 1200 requests/min.
    
    Endpoint: GET /api/v3/klines
    Returns: DataFrame with [open, high, low, close, volume] columns.
    """

    BASE_URL = MarketDataConfig.BINANCE_BASE_URL

    @classmethod
    def fetch_ohlcv(
        cls,
        symbol: str,
        interval: str = "1d",
        limit: int = MarketDataConfig.BINANCE_KLINES_LIMIT,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV kline data for a single crypto pair.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval ('1d', '1h', '4h', etc.)
            limit: Number of candles to fetch (max 1000)

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns
        """
        cache_key = f"binance_{symbol}_{interval}_{limit}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{cls.BASE_URL}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            raw_data = response.json()

            if not raw_data:
                logger.warning(f"Binance returned empty data for {symbol}")
                return pd.DataFrame()

            # Binance kline format:
            # [open_time, open, high, low, close, volume, close_time,
            #  quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
            df = pd.DataFrame(raw_data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            # Parse and clean
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df[["open", "high", "low", "close", "volume", "quote_volume"]].copy()
            df["symbol"] = symbol
            df["source"] = "binance"

            _cache.set(cache_key, df)
            logger.info(f"Binance: fetched {len(df)} candles for {symbol}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_multi(
        cls,
        symbols: Optional[list[str]] = None,
        interval: str = "1d",
        limit: int = MarketDataConfig.BINANCE_KLINES_LIMIT,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple crypto pairs.

        Args:
            symbols: List of trading pairs (default: config crypto symbols)
            interval: Candle interval
            limit: Number of candles

        Returns:
            Dict mapping symbol -> DataFrame
        """
        symbols = symbols or MarketDataConfig.CRYPTO_SYMBOLS
        result: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            df = cls.fetch_ohlcv(symbol, interval, limit)
            if not df.empty:
                result[symbol] = df
            # Respect rate limits
            time.sleep(0.1)

        logger.info(f"Binance: fetched data for {len(result)}/{len(symbols)} symbols")
        return result

    @classmethod
    def fetch_close_prices(
        cls,
        symbols: Optional[list[str]] = None,
        interval: str = "1d",
        limit: int = 252,
    ) -> pd.DataFrame:
        """
        Fetch close prices for multiple crypto pairs as a single DataFrame.
        Each column is a symbol. Suitable for correlation analysis.
        """
        data = cls.fetch_multi(symbols, interval, limit)
        if not data:
            return pd.DataFrame()

        closes = {}
        for symbol, df in data.items():
            closes[symbol] = df["close"]

        return pd.DataFrame(closes).sort_index()


# ══════════════════════════════════════════════════════
#  TWELVEDATA FETCHER — Equities, ETFs, Forex
# ══════════════════════════════════════════════════════
class TwelveDataFetcher:
    """
    Fetches equities, ETFs, and forex data from TwelveData API.
    Free tier: 800 API credits/day, 8 credits/minute.

    Strategy:
      - Use batch endpoint where possible (1 credit for up to 5 symbols)
      - Cache aggressively to minimize API calls
      - Graceful degradation if rate limited
    """

    BASE_URL = MarketDataConfig.TWELVEDATA_BASE_URL

    @classmethod
    def _get_api_key(cls) -> str:
        key = APIKeys.TWELVEDATA_API_KEY
        if not key or key == "your_twelvedata_api_key_here":
            raise ValueError(
                "TwelveData API key not configured. "
                "Set TWELVEDATA_API_KEY in your .env file. "
                "Get a free key at https://twelvedata.com/pricing"
            )
        return key

    @classmethod
    def fetch_time_series(
        cls,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 252,
    ) -> pd.DataFrame:
        """
        Fetch time series data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g., 'SPY', 'EUR/USD')
            interval: Data interval ('1day', '1h', '5min', etc.)
            outputsize: Number of data points (max 5000 on free tier)

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns
        """
        cache_key = f"twelvedata_{symbol}_{interval}_{outputsize}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{cls.BASE_URL}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": cls._get_api_key(),
            "format": "JSON",
            "timezone": "UTC",
        }

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "code" in data and data["code"] != 200:
                error_msg = data.get("message", "Unknown TwelveData error")
                logger.error(f"TwelveData API error for {symbol}: {error_msg}")
                return pd.DataFrame()

            if "values" not in data:
                logger.warning(f"TwelveData: no values returned for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(data["values"])
            df["timestamp"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.set_index("timestamp").sort_index()

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Keep only OHLCV columns
            available_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[available_cols].copy()
            df["symbol"] = symbol
            df["source"] = "twelvedata"

            _cache.set(cache_key, df)
            logger.info(f"TwelveData: fetched {len(df)} bars for {symbol}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"TwelveData API error for {symbol}: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_multi(
        cls,
        symbols: list[str],
        interval: str = "1day",
        outputsize: int = 252,
        delay: float = 8.0,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch time series for multiple symbols with rate limiting.

        TwelveData free tier: 8 credits/minute, so we space requests.

        Args:
            symbols: List of ticker symbols
            interval: Data interval
            outputsize: Number of data points
            delay: Seconds between requests (for rate limiting)

        Returns:
            Dict mapping symbol -> DataFrame
        """
        result: dict[str, pd.DataFrame] = {}

        for i, symbol in enumerate(symbols):
            # Check cache first (no delay needed for cache hits)
            cache_key = f"twelvedata_{symbol}_{interval}_{outputsize}"
            cached = _cache.get(cache_key)
            if cached is not None:
                result[symbol] = cached
                continue

            df = cls.fetch_time_series(symbol, interval, outputsize)
            if not df.empty:
                result[symbol] = df

            # Rate limit: wait between uncached requests
            if i < len(symbols) - 1:
                time.sleep(delay)

        logger.info(f"TwelveData: fetched data for {len(result)}/{len(symbols)} symbols")
        return result

    @classmethod
    def fetch_close_prices(
        cls,
        symbols: list[str],
        interval: str = "1day",
        outputsize: int = 252,
    ) -> pd.DataFrame:
        """
        Fetch close prices for multiple symbols as a single DataFrame.
        Each column = one symbol's close price. Index = date.
        """
        data = cls.fetch_multi(symbols, interval, outputsize)
        if not data:
            return pd.DataFrame()

        closes = {}
        for symbol, df in data.items():
            if "close" in df.columns:
                closes[symbol] = df["close"]

        return pd.DataFrame(closes).sort_index()

    @classmethod
    def fetch_equities(cls, outputsize: int = 252) -> dict[str, pd.DataFrame]:
        """Fetch all configured equity symbols."""
        return cls.fetch_multi(MarketDataConfig.EQUITY_SYMBOLS, outputsize=outputsize)

    @classmethod
    def fetch_etfs(cls, outputsize: int = 252) -> dict[str, pd.DataFrame]:
        """Fetch all configured ETF symbols."""
        return cls.fetch_multi(MarketDataConfig.ETF_SYMBOLS, outputsize=outputsize)

    @classmethod
    def fetch_forex(cls, outputsize: int = 252) -> dict[str, pd.DataFrame]:
        """Fetch all configured forex pairs."""
        return cls.fetch_multi(MarketDataConfig.FOREX_PAIRS, outputsize=outputsize)

    @classmethod
    def fetch_vix(cls, outputsize: int = 252) -> pd.DataFrame:
        """Fetch VIX volatility index data."""
        return cls.fetch_time_series(MarketDataConfig.VIX_SYMBOL, outputsize=outputsize)


# ══════════════════════════════════════════════════════
#  UNIFIED MARKET DATA AGGREGATOR
# ══════════════════════════════════════════════════════
class MarketDataAggregator:
    """
    Orchestrates data fetching across all asset classes.
    Returns a unified market snapshot consumed by the feature engine.
    """

    @classmethod
    def fetch_full_snapshot(cls, outputsize: int = 252) -> dict:
        """
        Fetch complete market data snapshot across all asset classes.

        Returns:
            {
                "crypto": {symbol: DataFrame, ...},
                "equities": {symbol: DataFrame, ...},
                "etfs": {symbol: DataFrame, ...},
                "forex": {symbol: DataFrame, ...},
                "vix": DataFrame,
                "close_prices": {
                    "crypto": DataFrame (columns=symbols),
                    "equities": DataFrame (columns=symbols),
                    "etfs": DataFrame (columns=symbols),
                },
                "metadata": {
                    "timestamp": str,
                    "asset_counts": dict,
                    "data_sources": list,
                }
            }
        """
        logger.info("═" * 60)
        logger.info("MARKET DATA AGGREGATOR — Fetching full snapshot")
        logger.info("═" * 60)

        snapshot = {
            "crypto": {},
            "equities": {},
            "etfs": {},
            "forex": {},
            "vix": pd.DataFrame(),
            "close_prices": {},
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data_sources": [],
                "asset_counts": {},
                "errors": [],
            }
        }

        # ── 1. Crypto (Binance) ──────────────────────────
        try:
            logger.info("Fetching crypto data from Binance...")
            snapshot["crypto"] = BinanceDataFetcher.fetch_multi(limit=outputsize)
            snapshot["close_prices"]["crypto"] = BinanceDataFetcher.fetch_close_prices(limit=outputsize)
            snapshot["metadata"]["data_sources"].append("binance")
            snapshot["metadata"]["asset_counts"]["crypto"] = len(snapshot["crypto"])
            logger.info(f"  ✓ Crypto: {len(snapshot['crypto'])} symbols")
        except Exception as e:
            logger.error(f"  ✗ Crypto fetch failed: {e}")
            snapshot["metadata"]["errors"].append(f"crypto: {str(e)}")

        # ── 2. Equities (TwelveData) ─────────────────────
        try:
            logger.info("Fetching equity data from TwelveData...")
            snapshot["equities"] = TwelveDataFetcher.fetch_equities(outputsize=outputsize)
            equity_symbols = list(snapshot["equities"].keys())
            if equity_symbols:
                snapshot["close_prices"]["equities"] = TwelveDataFetcher.fetch_close_prices(
                    equity_symbols, outputsize=outputsize
                )
            snapshot["metadata"]["data_sources"].append("twelvedata")
            snapshot["metadata"]["asset_counts"]["equities"] = len(snapshot["equities"])
            logger.info(f"  ✓ Equities: {len(snapshot['equities'])} symbols")
        except Exception as e:
            logger.error(f"  ✗ Equity fetch failed: {e}")
            snapshot["metadata"]["errors"].append(f"equities: {str(e)}")

        # ── 3. ETFs (TwelveData) ─────────────────────────
        try:
            logger.info("Fetching ETF data from TwelveData...")
            snapshot["etfs"] = TwelveDataFetcher.fetch_etfs(outputsize=outputsize)
            etf_symbols = list(snapshot["etfs"].keys())
            if etf_symbols:
                snapshot["close_prices"]["etfs"] = TwelveDataFetcher.fetch_close_prices(
                    etf_symbols, outputsize=outputsize
                )
            snapshot["metadata"]["asset_counts"]["etfs"] = len(snapshot["etfs"])
            logger.info(f"  ✓ ETFs: {len(snapshot['etfs'])} symbols")
        except Exception as e:
            logger.error(f"  ✗ ETF fetch failed: {e}")
            snapshot["metadata"]["errors"].append(f"etfs: {str(e)}")

        # ── 4. Forex (TwelveData) ────────────────────────
        try:
            logger.info("Fetching forex data from TwelveData...")
            snapshot["forex"] = TwelveDataFetcher.fetch_forex(outputsize=outputsize)
            snapshot["metadata"]["asset_counts"]["forex"] = len(snapshot["forex"])
            logger.info(f"  ✓ Forex: {len(snapshot['forex'])} symbols")
        except Exception as e:
            logger.error(f"  ✗ Forex fetch failed: {e}")
            snapshot["metadata"]["errors"].append(f"forex: {str(e)}")

        # ── 5. VIX (TwelveData) ──────────────────────────
        try:
            logger.info("Fetching VIX data from TwelveData...")
            snapshot["vix"] = TwelveDataFetcher.fetch_vix(outputsize=outputsize)
            logger.info(f"  ✓ VIX: {len(snapshot['vix'])} data points")
        except Exception as e:
            logger.error(f"  ✗ VIX fetch failed: {e}")
            snapshot["metadata"]["errors"].append(f"vix: {str(e)}")

        total_assets = sum(snapshot["metadata"]["asset_counts"].values())
        logger.info(f"═ Snapshot complete: {total_assets} assets from {len(snapshot['metadata']['data_sources'])} sources")

        return snapshot

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached data."""
        _cache.clear()
        logger.info("Market data cache cleared")
