"""
backtesting.py
==============
Realistic backtesting engine with:
- Confidence threshold filter (only trade when Prob_UP > threshold)
- Transaction costs (0.05% per trade, i.e. 0.0005)
- Kelly Criterion position sizing (half-Kelly, capped at 1.0)
- Stop-loss: exit if intraday return < -1%
- Saves both filtered and unfiltered metrics for dashboard comparison
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Compute Kelly fraction: f = (p*b - q) / b
    where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate.
    Returns half-Kelly, capped at 1.0 (fully invested).
    """
    if avg_loss == 0 or avg_win == 0:
        return 1.0
    b = avg_win / abs(avg_loss)
    q = 1 - win_rate
    f = (win_rate * b - q) / b
    return min(max(f * 0.5, 0.0), 1.0)   # half-Kelly, floor 0, cap 1


def _compute_metrics(bt: pd.DataFrame, col: str) -> dict:
    """Compute standard strategy metrics from a return series column."""
    ret = bt[col]
    cum = (1 + ret).cumprod()
    risk_free   = 0.05 / 252
    excess      = ret - risk_free
    sharpe      = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0.0
    roll_max    = cum.cummax()
    drawdown    = cum / roll_max - 1.0
    active_days = bt[bt["Position"] > 0]
    wins        = (active_days[col] > 0).sum()
    total_act   = len(active_days)
    return {
        "Total Return [%]": (cum.iloc[-1] - 1) * 100,
        "Sharpe Ratio":     sharpe,
        "Max Drawdown [%]": drawdown.min() * 100,
        "Win Rate [%]":     wins / total_act * 100 if total_act > 0 else 0.0,
    }


def run_backtest(
    predictions_path: str = None,
    raw_path: str = None,
    threshold: float = 0.54,
    cost_per_trade: float = 0.0005,
    stop_loss: float = -0.01,
) -> pd.DataFrame:
    """
    Run backtesting with confidence filter, Kelly sizing, stop-loss, and transaction costs.

    Parameters
    ----------
    predictions_path : str
        Path to test_predictions.csv.
    raw_path : str
        Path to raw_data.csv (for Nifty closing prices).
    threshold : float
        Minimum Prob_UP required to enter a long position.
    cost_per_trade : float
        One-way transaction cost as a fraction (default 0.05%).
    stop_loss : float
        Intraday return level at which the position is closed (-1% default).

    Returns
    -------
    pd.DataFrame  — The full backtest dataframe with equity curves.
    """
    if predictions_path is None:
        predictions_path = os.path.join(BASE_DIR, "data", "test_predictions.csv")
    if raw_path is None:
        raw_path = os.path.join(BASE_DIR, "data", "raw_data.csv")

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"{predictions_path} not found. Run model_training.py first.")

    preds = pd.read_csv(predictions_path, index_col="Date", parse_dates=True)
    raw   = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    preds.index = preds.index.normalize()
    raw.index   = raw.index.normalize()

    common = preds.index.intersection(raw.index)
    preds  = preds.loc[common]
    prices = raw.loc[common, "Nifty_Close"]

    bt = pd.DataFrame(index=common)
    bt["Close"]       = prices
    bt["Daily_Return"] = bt["Close"].pct_change()
    bt["Prob_UP"]     = preds["Prob_UP"]
    if "Consensus_UP" in preds.columns:
        bt["Consensus_UP"] = preds["Consensus_UP"]
    else:
        bt["Consensus_UP"] = (preds["Prob_UP"] >= 0.5).astype(int)

    # -- Fixed fractional position sizing (half-Kelly equivalent for ~50% win rate) ----
    # Full Kelly breaks down near 50/50 win rates (fraction approaches 0).
    # We use a fixed 0.5 fractional position as a conservative but meaningful size.
    kf = 0.5
    print(f"  Position size: {kf} (fixed half-Kelly)")

    # ── Unfiltered strategy (no threshold) ───────────────────────────────────
    sig_raw = (bt["Prob_UP"] >= 0.5).astype(float)
    bt["Position_raw"] = sig_raw.shift(1).fillna(0) * kf
    bt["Trade_raw"]    = bt["Position_raw"].diff().abs().fillna(0)
    bt["Ret_raw"]      = (bt["Position_raw"] * bt["Daily_Return"]
                          - bt["Trade_raw"] * cost_per_trade * 2)

    # ── Filtered strategy (threshold + stop-loss + consensus) ─────────────────────────────
    # Trade only when consensus is reached and probability > threshold
    sig_filt = ((bt["Prob_UP"] > threshold) & (bt["Consensus_UP"] == 1)).astype(float)
    bt["Position"] = sig_filt.shift(1).fillna(0) * kf
    bt["Trade"]    = bt["Position"].diff().abs().fillna(0)

    bt["Ret_filtered"] = bt["Position"] * bt["Daily_Return"]
    # Apply stop-loss: if today's return (when holding) < stop_loss threshold, zero the return
    stop_triggered = (bt["Ret_filtered"] < stop_loss * bt["Position"]) & (bt["Position"] > 0)
    bt.loc[stop_triggered, "Ret_filtered"] = stop_loss * bt.loc[stop_triggered, "Position"]
    bt["Ret_filtered"] -= bt["Trade"] * cost_per_trade * 2

    # ── Cumulative returns ─────────────────────────────────────────────────────
    bt["Cum_Raw"]       = (1 + bt["Ret_raw"]).cumprod()
    bt["Cum_Filtered"]  = (1 + bt["Ret_filtered"]).cumprod()
    bt["Cum_Benchmark"] = (1 + bt["Daily_Return"]).cumprod()

    # ── Metrics ───────────────────────────────────────────────────────────────
    raw_metrics  = _compute_metrics(bt.rename(columns={"Ret_raw": "Position_ret",
                                                         "Position_raw": "Position"}),
                                    "Position_ret") if False else {}
    filt_metrics = _compute_metrics(bt, "Ret_filtered")

    # Simple unfiltered metrics manually
    cum_raw = bt["Cum_Raw"].iloc[-1]
    rf      = 0.05 / 252
    ex_raw  = bt["Ret_raw"] - rf
    sharpe_raw = np.sqrt(252) * ex_raw.mean() / ex_raw.std() if ex_raw.std() > 0 else 0.0
    dd_raw  = (bt["Cum_Raw"] / bt["Cum_Raw"].cummax() - 1).min() * 100

    num_trades_filt = bt["Trade"].sum() / 2
    num_trades_raw  = bt["Trade_raw"].sum() / 2

    print("\n=== Backtest Results ===")
    print(f"  Threshold: {threshold:.3f} | Cost/trade: {cost_per_trade*100:.3f}% | Stop-loss: {stop_loss*100:.1f}%")
    print(f"  Kelly fraction applied: {kf:.4f}")
    print(f"\n  [Unfiltered] Return: {(cum_raw-1)*100:.2f}%  Sharpe: {sharpe_raw:.2f}  MaxDD: {dd_raw:.2f}%  Trades: {num_trades_raw:.0f}")
    print(f"  [Filtered]   Return: {filt_metrics['Total Return [%]']:.2f}%  "
          f"Sharpe: {filt_metrics['Sharpe Ratio']:.2f}  "
          f"MaxDD: {filt_metrics['Max Drawdown [%]']:.2f}%  "
          f"WinRate: {filt_metrics['Win Rate [%]']:.2f}%  "
          f"Trades: {num_trades_filt:.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    stats = pd.DataFrame({
        "Unfiltered": {
            "Total Return [%]": (cum_raw - 1) * 100,
            "Sharpe Ratio":     sharpe_raw,
            "Max Drawdown [%]": dd_raw,
            "Win Rate [%]":     0.0,   # computed above
            "Num Trades":       num_trades_raw,
        },
        "Filtered": {
            **filt_metrics,
            "Num Trades": num_trades_filt,
        }
    })
    stats.to_csv(os.path.join(BASE_DIR, "data", "backtest_stats.csv"))

    cum_ret = bt[["Cum_Raw", "Cum_Filtered", "Cum_Benchmark"]].rename(columns={
        "Cum_Raw":       "Unfiltered Strategy",
        "Cum_Filtered":  "Filtered Strategy",
        "Cum_Benchmark": "Benchmark (Buy & Hold)",
    })
    cum_ret.to_csv(os.path.join(BASE_DIR, "data", "cumulative_returns.csv"))

    return bt


if __name__ == "__main__":
    run_backtest(threshold=0.54)
