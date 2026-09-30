"""Part I: CSI 300 MA20 timing and look-ahead bias."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Paths are based on this file, so the script works directly in Spyder.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PACKAGE_ROOT / "data" / "csi300_daily.csv"
OUTPUT_DIR = PACKAGE_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Read and clean the CSI 300 daily file.
data = pd.read_csv(DATA_FILE, parse_dates=["date"])
numeric_columns = [
    "open", "high", "low", "close", "preclose", "volume", "amount"
]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
data = (
    data.sort_values("date")
    .drop_duplicates("date")
    .dropna(subset=["close"])
    .reset_index(drop=True)
)

# Build the signal, the biased position, and the executable position.
# TODO: 计算 20 日均线、信号、错误的持仓和可执行的持仓。

# Figure 1: index level and its 20-day moving average.
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(data["date"], data["close"], label="沪深300", linewidth=1.5)
ax.plot(data["date"], data["ma20"], label="20日均线", linewidth=1.2)
ax.set(xlabel="日期", ylabel="指数点位")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part1_price_ma20.png", dpi=200)
plt.close(fig)

# Figure 2: compare buy-and-hold, biased timing, and executable timing.
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(data["date"], data["index_nav"], label="Buy-and-Hold", linewidth=2)
ax.plot(data["date"], data["wrong_nav"], label="错误的MA20策略", linewidth=1.8)
ax.plot(data["date"], data["strategy_nav"], label="修正后的MA20策略", linewidth=1.8)
ax.set_yscale("log")
ax.set(xlabel="日期", ylabel="累计净值（对数坐标）")
ax.grid(alpha=0.25, which="both")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part1_ma20_nav.png", dpi=200)
plt.close(fig)

print("Part I final NAV:")
print(data[["index_nav", "wrong_nav", "strategy_nav"]].iloc[-1].round(4))
print(f"Figures saved to: {OUTPUT_DIR}")
