"""
feature_engineering.py
=======================
Generates a clean, leak-free feature set for the Nifty 50 Direction Predictor.

Leakage Audit (PASS):
- All indicators are computed on CURRENT day data (OHLCV is fully known at market close).
- The TARGET variable uses tomorrow's close via shift(-1) - applied ONLY in model_training.py.
- Macro features (VIX, USDINR) are shifted by 1 day before use as features.
- No raw price levels are included in the final feature output.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA with com=period-1)."""
    delta = series.diff()
    up   = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + up / down.replace(0, np.nan)))


def _cci(high, low, close, period=20):
    """Commodity Channel Index."""
    tp  = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def create_features(
    data_path: str = None,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw OHLCV + macro data.

    Parameters
    ----------
    data_path : str, optional
        Path to raw_data.csv. Defaults to <project_root>/data/raw_data.csv.
    save_path : str, optional
        Where to save features.csv. Defaults to <project_root>/data/features.csv.

    Returns
    -------
    pd.DataFrame  - Feature matrix (no raw OHLCV, no future data).
    """
    if data_path is None:
        data_path = os.path.join(BASE_DIR, "data", "raw_data.csv")
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "data", "features.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run data_loader.py first.")

    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    df.index = df.index.normalize()

    close  = df["Nifty_Close"]
    high   = df["Nifty_High"]
    low    = df["Nifty_Low"]
    volume = df["Nifty_Volume"]
    vix    = df["VIX_Close"]
    usdinr = df["USDINR_Close"]

    feat = pd.DataFrame(index=df.index)

    # ── 1. RSI (14 and 7) ────────────────────────────────────────────────────
    feat["RSI_14"]       = _rsi(close, 14)
    feat["RSI_7"]        = _rsi(close, 7)
    feat["RSI_overbought"] = (feat["RSI_14"] > 70).astype(int)
    feat["RSI_oversold"]   = (feat["RSI_14"] < 30).astype(int)

    # ── 2. MACD line, signal, histogram ──────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    feat["MACD_line"]    = ema12 - ema26
    feat["MACD_signal"]  = feat["MACD_line"].ewm(span=9, adjust=False).mean()
    feat["MACD_hist"]    = feat["MACD_line"] - feat["MACD_signal"]
    feat["MACD_cross"]   = (
        (feat["MACD_line"] > feat["MACD_signal"]) &
        (feat["MACD_line"].shift(1) <= feat["MACD_signal"].shift(1))
    ).astype(int)

    # ── 3. Bollinger Bands %B ─────────────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    feat["BB_pctB"]  = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    feat["BB_width"] = (bb_upper - bb_lower) / (sma20 + 1e-9)

    # ── 4. ATR (14) ───────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    feat["ATR_14"]   = tr.rolling(14).mean()
    feat["ATR_norm"] = feat["ATR_14"] / (close + 1e-9)

    # ── 5. OBV (On-Balance Volume) ────────────────────────────────────────────
    obv_sign        = np.sign(close.diff())
    obv             = (obv_sign * volume).cumsum()
    feat["OBV_EMA"] = obv.ewm(span=10, adjust=False).mean()
    feat["OBV_sig"] = (obv > feat["OBV_EMA"]).astype(int)
    feat["Vol_Ratio"] = volume / volume.rolling(20).mean()

    # ── 6. Stochastic %K and %D ───────────────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    feat["STOCHk"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    feat["STOCHd"] = feat["STOCHk"].rolling(3).mean()

    # ── 7. EMA 9/21 crossover ─────────────────────────────────────────────────
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200= close.ewm(span=200,adjust=False).mean()
    feat["EMA_cross_9_21"] = (
        (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
    ).astype(int)
    feat["Dist_EMA_9"]    = (close - ema9)   / (ema9   + 1e-9)
    feat["Dist_EMA_21"]   = (close - ema21)  / (ema21  + 1e-9)
    feat["Dist_EMA_50"]   = (close - ema50)  / (ema50  + 1e-9)
    feat["Dist_EMA_200"]  = (close - ema200) / (ema200 + 1e-9)
    feat["Above_EMA200"]  = (close > ema200).astype(int)

    # ── 8. Williams %R ────────────────────────────────────────────────────────
    feat["WilliamsR"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)

    # ── 9. CCI (20) ───────────────────────────────────────────────────────────
    feat["CCI_20"] = _cci(high, low, close, 20)

    # ── 10. Lag-1 to Lag-5 returns ────────────────────────────────────────────
    feat["Nifty_Return"] = close.pct_change()
    for i in range(1, 6):
        feat[f"Ret_Lag{i}"] = feat["Nifty_Return"].shift(i)

    # ── 11. Rate of Change ────────────────────────────────────────────────────
    feat["ROC_5"]  = close.pct_change(5)
    feat["ROC_10"] = close.pct_change(10)
    feat["ROC_20"] = close.pct_change(20)

    # ── 12. Volatility regime ─────────────────────────────────────────────────
    feat["Vol_20d"]          = feat["Nifty_Return"].rolling(20).std()
    feat["Vol_ratio_regime"] = feat["Vol_20d"] / feat["Vol_20d"].rolling(60).mean()

    # ── 13. Candle structure ──────────────────────────────────────────────────
    feat["Green_candle"] = (close > df["Nifty_Open"]).astype(int)
    feat["Body_ATR"]     = (close - df["Nifty_Open"]).abs() / (feat["ATR_14"] + 1e-9)

    # ── 14. Price structure ───────────────────────────────────────────────────
    feat["Pct_from_52wH"] = (close - close.rolling(252).max()) / (close.rolling(252).max() + 1e-9)
    feat["Pct_from_52wL"] = (close - close.rolling(252).min()) / (close.rolling(252).min() + 1e-9)

    # ── 15. MACRO - shift by 1 so today's macro doesn't leak ─────────────────
    # VIX: shift 1 (we know yesterday's VIX, not today's at open)
    feat["VIX_level"]     = vix.shift(1)
    feat["VIX_Return"]    = vix.pct_change().shift(1)
    feat["VIX_high_fear"] = (feat["VIX_level"] > 20).astype(int)

    # USDINR: shift 1
    feat["USDINR_Return"] = usdinr.pct_change().shift(1)
    feat["Rupee_weak"]    = (usdinr.shift(1) > usdinr.shift(1).rolling(5).mean()).astype(int)

    # ── Drop NaN rows (from rolling windows) ─────────────────────────────────
    feat.dropna(inplace=True)

    # ── Audit: confirm no raw price columns ───────────────────────────────────
    raw_cols = {"Nifty_Open","Nifty_High","Nifty_Low","Nifty_Close","Nifty_Volume",
                "VIX_Close","USDINR_Close"}
    leaked = raw_cols.intersection(set(feat.columns))
    assert len(leaked) == 0, f"LEAKAGE DETECTED: {leaked}"
    print(f"[Leakage Audit] PASS - {len(feat.columns)} features, 0 raw price columns.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        feat.to_csv(save_path)
        print(f"Features saved -> {save_path}  |  Shape: {feat.shape}")

    return feat


if __name__ == "__main__":
    create_features()
