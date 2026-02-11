"""
Gas Price Predictor - Data Collection
Fetches data from EIA API, Yahoo Finance, and FRED.
"""

import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    CACHE_FILE,
    DEFAULT_HISTORY_YEARS,
    EIA_RETAIL_GAS,
    EIA_WEEKLY_SERIES,
    FRED_SERIES,
    YAHOO_TICKERS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
#  EIA DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_eia_gas_prices(api_key: str, years: int = DEFAULT_HISTORY_YEARS) -> pd.DataFrame:
    """Fetch weekly retail gasoline prices from EIA API v2."""
    if not api_key:
        raise ValueError(
            "EIA API key is required. Get one free at "
            "https://www.eia.gov/opendata/register.php"
        )

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    url = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[product][]": "EPM0",
        "facets[duoarea][]": "NUS",
        "facets[process][]": "PTE",
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "response" not in data or "data" not in data["response"]:
        return _fetch_eia_v2_fallback(api_key, years)

    records = data["response"]["data"]
    if not records:
        return _fetch_eia_v2_fallback(api_key, years)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"])
    df["gas_price"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "gas_price"]].dropna().sort_values("date").reset_index(drop=True)
    return df


def _fetch_eia_v2_fallback(api_key: str, years: int = DEFAULT_HISTORY_YEARS) -> pd.DataFrame:
    """Fallback: fetch retail gas prices via EIA API v2 seriesid endpoint."""
    url = f"https://api.eia.gov/v2/seriesid/{EIA_RETAIL_GAS}"
    params = {
        "api_key": api_key,
        "data[0]": "value",
        "length": 5000,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "response" not in data or "data" not in data["response"] or not data["response"]["data"]:
        raise ValueError("No data returned from EIA API. Check your API key.")

    records = data["response"]["data"]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"])
    df["gas_price"] = pd.to_numeric(df["value"], errors="coerce")

    cutoff = datetime.now() - timedelta(days=365 * years)
    df = df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)
    return df[["date", "gas_price"]]


def fetch_eia_weekly_series(
    api_key: str,
    series_id: str,
    col_name: str,
    years: int = DEFAULT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Fetch a single EIA weekly petroleum series via API v2 seriesid endpoint."""
    url = f"https://api.eia.gov/v2/seriesid/PET.{series_id}.W"
    params = {
        "api_key": api_key,
        "data[0]": "value",
        "length": 5000,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "response" not in data or "data" not in data["response"] or not data["response"]["data"]:
            return pd.DataFrame(columns=["date", col_name])

        records = data["response"]["data"]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["period"])
        df[col_name] = pd.to_numeric(df["value"], errors="coerce")

        cutoff = datetime.now() - timedelta(days=365 * years)
        df = df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)
        return df[["date", col_name]]
    except Exception as e:
        print(f"  Warning: Could not fetch EIA series {series_id}: {e}")
        return pd.DataFrame(columns=["date", col_name])


def fetch_all_eia_supply_demand(
    api_key: str, years: int = DEFAULT_HISTORY_YEARS
) -> pd.DataFrame:
    """Fetch all EIA weekly petroleum supply/demand series and merge."""
    dfs = []
    for col_name, series_id in EIA_WEEKLY_SERIES.items():
        print(f"  Fetching EIA: {col_name} ({series_id})...")
        df = fetch_eia_weekly_series(api_key, series_id, col_name, years)
        if not df.empty and len(df) > 10:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["date"])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").ffill().dropna(subset=["date"])
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  YAHOO FINANCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_market_data(
    ticker: str,
    column_prefix: str,
    years: int = DEFAULT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Fetch historical weekly closing prices from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=365 * years + 30)

    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if data.empty:
        return pd.DataFrame(columns=["date", f"{column_prefix}_price"])

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data[["Close"]].copy()
    df.index = pd.to_datetime(df.index)

    # Resample to weekly (week-ending Sunday)
    df_weekly = df.resample("W").last().ffill()
    col_name = f"{column_prefix}_price"
    df_weekly.columns = [col_name]
    df_weekly.index.name = "date"
    df_weekly = df_weekly.reset_index()
    return df_weekly


def fetch_all_yahoo_data(years: int = DEFAULT_HISTORY_YEARS) -> pd.DataFrame:
    """Fetch all Yahoo Finance tickers and merge into weekly DataFrame."""
    dfs = []
    for ticker, prefix in YAHOO_TICKERS.items():
        try:
            print(f"  Fetching Yahoo: {prefix} ({ticker})...")
            df = fetch_market_data(ticker, prefix, years=years)
            if not df.empty and len(df) > 10:
                dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not fetch {ticker}: {e}")

    if not dfs:
        raise ValueError("Could not fetch any market data from Yahoo Finance")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").ffill().dropna(
        subset=[c for c in merged.columns if c != "date"], how="all"
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  FRED DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fred_series(
    api_key: str,
    series_id: str,
    col_name: str,
    years: int = DEFAULT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Fetch a single FRED series via the FRED API."""
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "observations" not in data:
            return pd.DataFrame(columns=["date", col_name])

        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df[col_name] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", col_name]].dropna()
        return df
    except Exception as e:
        print(f"  Warning: Could not fetch FRED series {series_id}: {e}")
        return pd.DataFrame(columns=["date", col_name])


def fetch_all_fred_data(
    api_key: str, years: int = DEFAULT_HISTORY_YEARS
) -> pd.DataFrame:
    """Fetch all FRED series and merge into a weekly DataFrame."""
    if not api_key:
        return pd.DataFrame(columns=["date"])

    dfs = []
    for series_id, col_name in FRED_SERIES.items():
        print(f"  Fetching FRED: {col_name} ({series_id})...")
        df = fetch_fred_series(api_key, series_id, col_name, years)
        if not df.empty and len(df) > 5:
            # Resample to weekly (some FRED data is daily, some monthly)
            df = df.set_index("date").resample("W").last().ffill().reset_index()
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["date"])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    merged = merged.sort_values("date").ffill()
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def _snap_to_sunday(dates: pd.Series) -> pd.Series:
    """Snap all dates to the Sunday ending their ISO week."""
    return dates + pd.to_timedelta((6 - dates.dt.dayofweek) % 7, unit="D")

def build_combined_dataset(
    eia_api_key: str,
    fred_api_key: str = "",
    years: int = DEFAULT_HISTORY_YEARS,
) -> pd.DataFrame:
    """
    Fetch gas prices, market data, EIA supply/demand, and FRED economic data.
    Combine into a single weekly dataset.
    """
    print("Fetching EIA retail gas prices...")
    gas_df = fetch_eia_gas_prices(eia_api_key, years=years)

    print("Fetching Yahoo Finance market data...")
    market_df = fetch_all_yahoo_data(years=years)

    print("Fetching EIA supply/demand data...")
    eia_sd_df = fetch_all_eia_supply_demand(eia_api_key, years=years)

    print("Fetching FRED economic data...")
    fred_df = fetch_all_fred_data(fred_api_key, years=years)

    # Snap all dates to week-ending Sunday for consistent alignment
    gas_df["date"] = _snap_to_sunday(gas_df["date"])
    market_df["date"] = _snap_to_sunday(market_df["date"])

    if not eia_sd_df.empty and "date" in eia_sd_df.columns:
        eia_sd_df["date"] = _snap_to_sunday(eia_sd_df["date"])

    if not fred_df.empty and "date" in fred_df.columns:
        fred_df["date"] = _snap_to_sunday(fred_df["date"])

    # Merge all datasets
    combined = pd.merge(gas_df, market_df, on="date", how="inner")

    if not eia_sd_df.empty and len(eia_sd_df.columns) > 1:
        combined = pd.merge(combined, eia_sd_df, on="date", how="left")

    if not fred_df.empty and len(fred_df.columns) > 1:
        combined = pd.merge(combined, fred_df, on="date", how="left")

    combined = combined.sort_values("date").reset_index(drop=True)

    # Forward-fill any gaps from mismatched frequencies
    combined = combined.ffill()

    # RBOB sanity check (should be ~1-4 $/gal)
    if "rbob_price" in combined.columns:
        if combined["rbob_price"].median() > 100:
            combined["rbob_price"] = combined["rbob_price"] / 100

    print(f"Combined dataset: {len(combined)} weeks, {len(combined.columns)} columns")
    return combined


def cache_dataset(df: pd.DataFrame) -> None:
    """Save dataset to parquet cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)


def load_cached_dataset() -> Optional[pd.DataFrame]:
    """Load dataset from parquet cache if recent."""
    if not CACHE_FILE.exists():
        return None
    df = pd.read_parquet(CACHE_FILE)
    if df.empty:
        return None
    import os
    cache_age = datetime.now().timestamp() - os.path.getmtime(CACHE_FILE)
    if cache_age > 86400:
        return None
    return df
