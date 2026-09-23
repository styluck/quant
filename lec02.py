"""Lecture 2 reference code: a Shanghai smallest-market-cap strategy."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output" / "lec02"


def plot_csi300_buy_and_hold():
    """Plot CSI 300 buy-and-hold NAV and drawdown for 2021-2025."""
    prices = pd.read_csv(DATA_DIR / "csi300_daily.csv", parse_dates=["date"])
    prices = prices.loc[
        prices["date"].between("2021-01-01", "2025-12-31"),
        ["date", "close"],
    ].dropna().sort_values("date")
    prices = prices.set_index("date")

    # Buying once and holding makes NAV proportional to the index close.
    nav = prices["close"] / prices["close"].iloc[0]
    high_water_mark = nav.cummax()
    drawdown = nav / high_water_mark - 1.0
    trough_date = drawdown.idxmin()
    max_drawdown = drawdown.loc[trough_date]
    peak_date = nav.loc[:trough_date].idxmax()
    max_drawdown_nav = nav.loc[peak_date:trough_date]

    plt.rcParams["font.family"] = ["Microsoft YaHei"]
    fig, (ax_nav, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(11, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10},
    )

    ax_nav.plot(nav.index, nav, color="#28649B", linewidth=2.2, label="沪深300净值")
    ax_nav.plot(
        max_drawdown_nav.index,
        max_drawdown_nav,
        color="#B84A4A",
        linewidth=3.0,
        label="最大回撤区间",
    )
    ax_nav.scatter(
        [peak_date, trough_date],
        [nav.loc[peak_date], nav.loc[trough_date]],
        color="#8E2020",
        s=24,
        zorder=3,
    )
    ax_nav.axhline(1.0, color="#8B96A1", linestyle="--", linewidth=0.9)
    ax_nav.set_ylabel("净值")
    ax_nav.legend(loc="upper right", frameon=False, fontsize=9)
    ax_nav.grid(axis="y", linestyle=":", alpha=0.45)
    ax_nav.spines[["top", "right"]].set_visible(False)

    ax_dd.fill_between(drawdown.index, drawdown, 0, color="#B84A4A", alpha=0.28)
    ax_dd.plot(drawdown.index, drawdown, color="#B84A4A", linewidth=1.2)
    ax_dd.scatter([trough_date], [max_drawdown], color="#8E2020", s=28, zorder=3)
    ax_dd.annotate(
        f"最大回撤 {max_drawdown:.1%}",
        xy=(trough_date, max_drawdown),
        xytext=(18, 10),
        textcoords="offset points",
        color="#8E2020",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#8E2020", "lw": 0.8},
    )
    ax_dd.set_ylabel("回撤")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dd.grid(axis="y", linestyle=":", alpha=0.45)
    ax_dd.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.99,
        0.01,
        "数据来源：BaoStock；按指数收盘价计算，不含交易成本",
        ha="right",
        fontsize=8,
        color="#66727D",
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.98, bottom=0.08, hspace=0.10)
    fig.savefig(OUTPUT_DIR / "csi300_buy_and_hold_2021_2025.png", dpi=200)
    plt.close(fig)


def backtest_smallest_100(panel):
    """Run the naive Shanghai smallest-market-cap portfolio."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["code"].astype(str).str.startswith("sh.6", na=False)].copy()
    panel["size_rank"] = panel.groupby("date")["market_cap"].rank(
        method="first", ascending=True
    )
    selected = panel[panel["size_rank"] <= 100]

    strategy_return = selected.groupby("date")["next_ret"].mean()
    benchmark_return = panel.groupby("date")["benchmark_ret"].first()
    result = pd.concat(
        [
            strategy_return.rename("Shanghai Smallest 100"),
            benchmark_return.rename("CSI 300"),
        ],
        axis=1,
    ).dropna()
    return result, selected


def apply_tradability_filters(panel):
    """Apply simple eligibility filters before ranking stocks by size."""
    panel = panel.copy()
    min_amount_thousand_cny = 10_000  # 1,000万元；amount 的单位为千元
    return panel[
        (panel["is_st"] == 0)
        & (panel["listed_days"] >= 120)
        & (panel["tradable"] == 1)
        & (panel["amount"] >= min_amount_thousand_cny)
    ].copy()


def plot_small_cap_result(result, filename):
    """Plot strategy and benchmark net values on a logarithmic scale."""
    nav = (1 + result).cumprod()
    axis = nav.plot(figsize=(11, 6), logy=True)
    axis.set_title("Shanghai Smallest 100 Stocks vs CSI 300")
    axis.set_ylabel("Net value (log scale)")
    axis.grid(alpha=0.2, which="both")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close()


def plot_paper_result(result):
    """Plot the classroom paper backtest from the monthly teaching panel."""
    nav = (1 + result).cumprod().rename(
        columns={
            "Shanghai Smallest 100": "沪市最小市值100只股票",
            "CSI 300": "沪深300",
        }
    )
    plt.rcParams["font.family"] = ["Microsoft YaHei"]
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.plot(nav.index, nav.iloc[:, 0], color="#B84A4A", lw=2.2, label=nav.columns[0])
    axis.plot(nav.index, nav.iloc[:, 1], color="#28649B", lw=1.9, label=nav.columns[1])
    axis.set_yscale("log")
    axis.set_ylabel("净值（对数坐标）")
    axis.grid(axis="y", which="both", linestyle=":", alpha=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    fig.text(
        0.99,
        0.01,
        "月度等权纸面回测；未计交易成本与成交限制",
        ha="right",
        fontsize=8,
        color="#66727D",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.97, bottom=0.12)
    fig.savefig(
        OUTPUT_DIR / "shanghai_smallest_100_paper_backtest.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(fig)
    nav.to_csv(OUTPUT_DIR / "shanghai_smallest_100_paper_backtest.csv")


def plot_filter_comparison(naive_result, realistic_result):
    """Compare naive, filtered, and benchmark NAV on a common sample."""
    returns = pd.concat(
        [
            naive_result["Shanghai Smallest 100"].rename("原始纸面策略"),
            realistic_result["Shanghai Smallest 100"].rename("基本过滤后策略"),
            naive_result["CSI 300"].rename("沪深300"),
        ],
        axis=1,
    ).dropna()
    nav = (1 + returns).cumprod()

    plt.rcParams["font.family"] = ["Microsoft YaHei"]
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    colors = ["#B84A4A", "#3B8C6E", "#28649B"]
    for column, color in zip(nav.columns, colors):
        axis.plot(nav.index, nav[column], label=column, color=color, linewidth=2.0)
    axis.set_yscale("log")
    axis.set_ylabel("净值（对数坐标）")
    axis.grid(axis="y", which="both", linestyle=":", alpha=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.97, bottom=0.09)
    fig.savefig(
        OUTPUT_DIR / "shanghai_smallest_100_filter_comparison.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(fig)

    stats = pd.DataFrame(index=returns.columns)
    stats["final_nav"] = nav.iloc[-1]
    stats["annual_return"] = (1 + returns).prod() ** (12 / len(returns)) - 1
    stats["annual_vol"] = returns.std() * 12 ** 0.5
    stats["sharpe"] = returns.mean().div(returns.std()) * 12 ** 0.5
    stats["max_drawdown"] = (nav / nav.cummax() - 1).min()
    stats.to_csv(OUTPUT_DIR / "shanghai_smallest_100_filter_stats.csv")
    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_csi300_buy_and_hold()

    panel_file = DATA_DIR / "sh_stock_monthly_panel.csv"
    if not panel_file.exists():
        print(f"Small-cap practice skipped: missing {panel_file}")
        return

    panel = pd.read_csv(panel_file)
    naive_result, _ = backtest_smallest_100(panel)
    plot_small_cap_result(naive_result, "shanghai_smallest_100_naive.png")
    plot_paper_result(naive_result)

    eligible = apply_tradability_filters(panel)
    realistic_result, _ = backtest_smallest_100(eligible)
    plot_small_cap_result(realistic_result, "shanghai_smallest_100_filtered.png")
    print(plot_filter_comparison(naive_result, realistic_result))


if __name__ == "__main__":
    main()
