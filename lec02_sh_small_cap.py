"""Lecture 2: Shanghai smallest-100 market-cap paper backtest.

The implementation follows the data layout and portfolio construction ideas in
``dataload_new/stock_strategies/small_cap_index_compose.py``, but uses only
Shanghai-listed stocks, total market capitalization, monthly rebalancing, and
equal weights.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


DATA_ROOT = Path(r"F:\Codes\data")
STOCK_DIR = DATA_ROOT / "Stock_data"
BENCHMARK_FILE = DATA_ROOT / "index" / "000300.SH.mat"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "lec02"
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
N_STOCKS = 100


def matlab_dates(values):
    """Convert MATLAB serial dates to a normalized DatetimeIndex."""
    return pd.to_datetime(np.asarray(values).reshape(-1) - 719529, unit="D").normalize()


def load_field(file_path, field):
    """Load one field from a MAT file as a date-indexed Series."""
    data = loadmat(file_path)
    dates = matlab_dates(data["time"])
    values = np.asarray(data[field], dtype=float).reshape(-1)
    length = min(len(dates), len(values))
    return pd.Series(values[:length], index=dates[:length])


def load_shanghai_panel():
    """Load adjusted close, total market cap, and amount for Shanghai stocks."""
    close_data = {}
    market_cap_data = {}
    amount_data = {}

    for file_path in sorted(STOCK_DIR.glob("6*.SH.mat")):
        code = file_path.stem
        data = loadmat(file_path)
        required = {"time", "close", "adj_factor", "total_mv", "amount"}
        if not required.issubset(data):
            continue

        dates = matlab_dates(data["time"])
        close = np.asarray(data["close"], dtype=float).reshape(-1)
        adj_factor = np.asarray(data["adj_factor"], dtype=float).reshape(-1)
        market_cap = np.asarray(data["total_mv"], dtype=float).reshape(-1)
        amount = np.asarray(data["amount"], dtype=float).reshape(-1)
        length = min(len(dates), len(close), len(adj_factor), len(market_cap), len(amount))
        index = dates[:length]

        close_data[code] = pd.Series(close[:length] * adj_factor[:length], index=index)
        market_cap_data[code] = pd.Series(market_cap[:length], index=index)
        amount_data[code] = pd.Series(amount[:length], index=index)

    adjusted_close = pd.DataFrame(close_data).sort_index().loc[START_DATE:END_DATE]
    market_cap = pd.DataFrame(market_cap_data).reindex(adjusted_close.index)
    amount = pd.DataFrame(amount_data).reindex(adjusted_close.index)
    return adjusted_close, market_cap, amount


def monthly_smallest_100_weights(adjusted_close, market_cap, amount):
    """Form equal-weighted targets at each month end using observable data."""
    month_end_dates = market_cap.groupby(market_cap.index.to_period("M")).tail(1).index
    targets = pd.DataFrame(0.0, index=month_end_dates, columns=market_cap.columns)

    for date in month_end_dates:
        eligible = market_cap.loc[date].where(
            adjusted_close.loc[date].notna() & amount.loc[date].notna()
        ).dropna()
        selected = eligible.nsmallest(N_STOCKS).index
        if len(selected):
            targets.loc[date, selected] = 1.0 / len(selected)

    # Targets formed after month-end close become holdings from the next day.
    return targets.reindex(adjusted_close.index).ffill().fillna(0.0)


def run_backtest():
    """Return paper-strategy and CSI 300 NAV series."""
    adjusted_close, market_cap, amount = load_shanghai_panel()
    weights = monthly_smallest_100_weights(adjusted_close, market_cap, amount)
    stock_returns = adjusted_close.pct_change(fill_method=None).fillna(0.0)
    strategy_returns = (weights.shift(1).fillna(0.0) * stock_returns).sum(axis=1)

    start = weights.sum(axis=1).gt(0).idxmax()
    strategy_nav = (1.0 + strategy_returns.loc[start:]).cumprod()

    benchmark_close = load_field(BENCHMARK_FILE, "close").loc[start:END_DATE]
    benchmark_close = benchmark_close.reindex(strategy_nav.index).ffill().dropna()
    strategy_nav = strategy_nav.reindex(benchmark_close.index)
    benchmark_nav = benchmark_close / benchmark_close.iloc[0]
    return pd.concat(
        [strategy_nav.rename("Shanghai smallest 100"), benchmark_nav.rename("CSI 300")],
        axis=1,
    ).dropna()


def plot_result(nav):
    """Plot strategy and benchmark NAV on a log scale."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Microsoft YaHei"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(nav.index, nav["Shanghai smallest 100"], color="#B84A4A", lw=2.2,
            label="沪市最小市值100只股票")
    ax.plot(nav.index, nav["CSI 300"], color="#28649B", lw=1.9, label="沪深300")
    ax.set_yscale("log")
    ax.set_ylabel("净值（对数坐标）")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.99,
        0.01,
        "纸面回测：月度等权；未计交易成本与成交限制",
        ha="right",
        fontsize=8,
        color="#66727D",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.97, bottom=0.12)
    output_file = OUTPUT_DIR / "shanghai_smallest_100_daily_reference.png"
    fig.savefig(output_file, dpi=220, facecolor="white")
    plt.close(fig)
    nav.to_csv(OUTPUT_DIR / "shanghai_smallest_100_daily_reference.csv")
    return output_file


def main():
    nav = run_backtest()
    output_file = plot_result(nav)
    print(nav.tail())
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
