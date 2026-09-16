"""Lecture 1 reference code: CSI 300 and an MA20 timing strategy."""

from pathlib import Path

import baostock as bs
import matplotlib.pyplot as plt
import pandas as pd


START_DATE = "2015-01-01"
INDEX_CODE = "sh.000300"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURE_DIR = PROJECT_ROOT / "output" / "lec01"


def download_csi300(start_date=START_DATE):
    """Download daily CSI 300 index data from BaoStock."""
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

    # Requested columns: date/code, OHLC prices, previous close, volume, amount.
    fields = "date,code,open,high,low,close,preclose,volume,amount"
    # start_date and end_date define the historical sample period.
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        result = bs.query_history_k_data_plus(
            INDEX_CODE,
            fields,
            start_date=start_date,
            end_date=end_date,
            # Daily frequency; adjustflag="3" means unadjusted prices.
            frequency="d",
            adjustflag="3",
        )
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {result.error_msg}")

        rows = []
        while result.next():
            rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()


def clean_data(data):
    """Convert fields, remove invalid rows, and sort by trading date."""
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
    ]
    data[numeric_cols] = data[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    return (
        data.sort_values("date")
        .drop_duplicates("date")
        .dropna(subset=["close"])
        .reset_index(drop=True)
    )


def calculate_strategy(data):
    """Calculate both the look-ahead-biased and corrected MA20 strategies."""
    data = data.copy()
    data["ma20"] = data["close"].rolling(20).mean()
    data["signal"] = (data["close"] > data["ma20"]).astype(int)
    data["index_ret"] = data["close"].pct_change()

    # Classroom trap: today's close is not known before today's return occurs.
    data["wrong_position"] = data["signal"]
    data["wrong_ret"] = data["wrong_position"] * data["index_ret"]

    # Correct rule: a signal observed at close t is used from day t+1.
    data["position"] = data["signal"].shift(1).fillna(0)
    data["strategy_ret"] = data["position"] * data["index_ret"]

    data["index_nav"] = (1 + data["index_ret"].fillna(0)).cumprod()
    data["wrong_nav"] = (1 + data["wrong_ret"].fillna(0)).cumprod()
    data["strategy_nav"] = (1 + data["strategy_ret"].fillna(0)).cumprod()
    return data


def plot_price_and_ma20(data, output_dir):
    """Plot the CSI 300 index and its 20-day moving average."""
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.plot(data["date"], data["close"], color="#215a9a", label="CSI 300")
    axis.plot(data["date"], data["ma20"], color="#e07a28", label="MA20")
    axis.set_title("CSI 300 and Its 20-Day Moving Average")
    axis.set_xlabel("Date")
    axis.set_ylabel("Index level")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "lec01_csi300_ma20.png", dpi=180)
    # plt.close(figure)


def plot_strategy_comparison(data, output_dir):
    """Compare the first-pass MA20 backtest with buy-and-hold."""
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.plot(
        data["date"],
        data["index_nav"],
        color="#555555",
        linewidth=2,
        label="CSI 300 Buy-and-Hold",
    )
    axis.plot(
        data["date"],
        data["wrong_nav"],
        color="#b12f2f",
        linewidth=2,
        label="MA20 Timing Strategy",
    )
    axis.set_yscale("log")
    axis.set_title("MA20 Strategy vs CSI 300 Buy-and-Hold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Net value (log scale)")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "lec01_strategy_nav_log.png", dpi=180)
    # plt.close(figure)


def plot_corrected_strategy_comparison(data, output_dir):
    """Compare the executable MA20 strategy with buy-and-hold."""
    index_final = data["index_nav"].iloc[-1]
    strategy_final = data["strategy_nav"].iloc[-1]
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.plot(
        data["date"],
        data["index_nav"],
        color="#555555",
        linewidth=2,
        label=f"CSI 300 Buy-and-Hold (final NAV: {index_final:.3f})",
    )
    axis.plot(
        data["date"],
        data["strategy_nav"],
        color="#215a9a",
        linewidth=2,
        label=f"Corrected MA20 Strategy (final NAV: {strategy_final:.3f})",
    )
    axis.set_yscale("log")
    axis.set_title("Corrected MA20 Strategy vs CSI 300 Buy-and-Hold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Net value (log scale)")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        output_dir / "lec01_corrected_strategy_nav_log.png", dpi=180
    )
    # plt.close(figure)


def main():
    data = clean_data(download_csi300())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "csi300_daily.csv"
    data.to_csv(output_file, index=False)
    print(f"Saved {len(data)} rows to {output_file}")

    result = calculate_strategy(data)
    print(result[["date", "close", "ma20", "signal"]].tail())
    plot_price_and_ma20(result, FIGURE_DIR)
    plot_strategy_comparison(result, FIGURE_DIR)
    plot_corrected_strategy_comparison(result, FIGURE_DIR)
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
