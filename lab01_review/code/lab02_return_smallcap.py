"""Part II: return statistics and the Shanghai smallest-100 strategy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUT_DIR = PACKAGE_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------------------------
# Experiment 1: compare daily return distributions of two stocks.
# -------------------------------------------------------------------------
codes = ["sh.600000", "sh.601127"]
adjusted_close = {}

for code in codes:
    stock = pd.read_csv(DATA_DIR / f"{code}.csv", parse_dates=["date"])
    stock["close"] = pd.to_numeric(stock["close"], errors="coerce")
    stock["backAdjustFactor"] = pd.to_numeric(
        stock["backAdjustFactor"], errors="coerce"
    )
    stock["adj_close"] = stock["close"] * stock["backAdjustFactor"]
    adjusted_close[code] = stock.set_index("date")["adj_close"]

adjusted_close = pd.DataFrame(adjusted_close).sort_index()
returns = adjusted_close.pct_change(fill_method=None).loc["2020":"2025"].dropna()
# TODO: compute drawdown for each stock.

# TODO: compute annualized return, annualized volatility, and Sharpe ratio for each stock.

print(summary.round(4))
# summary.to_csv(OUTPUT_DIR / "part2_return_summary.csv", encoding="utf-8-sig")

common_bins = np.linspace(-0.11, 0.11, 61)
fig, ax = plt.subplots(figsize=(10, 5.4))
for code in codes:
    ax.hist(returns[code], bins=common_bins, alpha=0.5, label=code)
ax.set(xlabel="日收益率", ylabel="频数", xlim=(-0.11, 0.11))
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part2_return_hist.png", dpi=200)
plt.close(fig)

print("Part II return statistics:")
print(summary.round(4))

# -------------------------------------------------------------------------
# Experiment 2: hold the 100 smallest Shanghai stocks at each month end.
# -------------------------------------------------------------------------
panel = pd.read_csv(DATA_DIR / "sh_stock_monthly_panel.csv", parse_dates=["date"])
# TODO: 比较 lec01_download_sse.py 下载的日线宽表与本月度研究面板。
panel = panel[panel["code"].astype(str).str.startswith("sh.6")].copy()

# TODO: rank all observations with valid market cap and next return.

#TODO: Basic feasibility filters are applied before ranking.


benchmark_ret = panel.groupby("date")["benchmark_ret"].first().rename("沪深300")
strategy_returns = pd.concat(
    [naive_ret, filtered_ret, benchmark_ret], axis=1
).dropna()
strategy_nav = (1 + strategy_returns).cumprod()

annual_return = strategy_nav.iloc[-1] ** (12 / len(strategy_nav)) - 1
annual_vol = strategy_returns.std() * np.sqrt(12)
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(12)
strategy_drawdown = strategy_nav / strategy_nav.cummax() - 1
performance = pd.DataFrame(
    {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": strategy_drawdown.min(),
    }
)
performance.to_csv(OUTPUT_DIR / "part2_small_cap_performance.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(10.5, 5.6))
for column in strategy_nav:
    ax.plot(strategy_nav.index, strategy_nav[column], label=column, linewidth=1.8)
ax.set_yscale("log")
ax.set(xlabel="日期", ylabel="累计净值（对数坐标）")
ax.grid(alpha=0.25, which="both")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part2_small_cap_nav.png", dpi=200)
plt.close(fig)

print("\nPart II small-cap performance:")
print(performance.round(4))
print(f"Figures and tables saved to: {OUTPUT_DIR}")
