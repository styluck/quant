"""Part III: mean-variance analysis, rolling Beta, and a Beta-one portfolio."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
OUTPUT_DIR = PACKAGE_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
PLOT_START = "2020-01-01"
PLOT_END = "2025-12-31"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# The fixed 2026-08-31 top-ten universe intentionally ignores look-ahead bias.
names = {
    "sz.300750": "宁德时代",
    "sz.300308": "中际旭创",
    "sh.600519": "贵州茅台",
    "sh.601318": "中国平安",
    "sz.300502": "新易盛",
    "sh.601899": "紫金矿业",
    "sh.600036": "招商银行",
    "sz.000333": "美的集团",
    "sh.603259": "药明康德",
    "sh.688256": "寒武纪",
}
codes = list(names)

# Read back-adjusted closes and calculate daily stock returns.
adjusted_close = {}
for code in codes:
    stock = pd.read_csv(DATA_DIR / f"{code}.csv", parse_dates=["date"])
    stock["close"] = pd.to_numeric(stock["close"], errors="coerce")
    stock["backAdjustFactor"] = pd.to_numeric(
        stock["backAdjustFactor"], errors="coerce"
    )
    stock["adj_close"] = stock["close"] * stock["backAdjustFactor"]
    adjusted_close[code] = stock.set_index("date")["adj_close"]

# TODO: 计算股票的日度收益率数据。

# Read the CSI 300 return as the market return.
index_data = pd.read_csv(DATA_DIR / "csi300_daily.csv", parse_dates=["date"])
index_data["close"] = pd.to_numeric(index_data["close"], errors="coerce")

# TODO: 计算沪深300指数的日度收益率数据。

sample = stock_ret.loc[PLOT_START:PLOT_END]
common = pd.concat([sample, market_ret], axis=1).dropna()
print(
    "Common sample:", common.index.min().date(), "to",
    common.index.max().date(), f"({len(common)} trading days)"
)

# Task 1: locate the ten stocks in annual mean-volatility space.
# TODO: 计算每只股票的年化平均收益率和年化标准差，并保存到statistics DataFrame中。

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.scatter(statistics["vol_ann"], statistics["mean_ann"], s=55)
for code, row in statistics.iterrows():
    ax.annotate(
        names[code], (row["vol_ann"], row["mean_ann"]),
        xytext=(5, 4), textcoords="offset points"
    )
ax.set(xlabel="年化标准差", ylabel="年化平均收益率")
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part3_mean_variance.png", dpi=200)
plt.close(fig)

# Task 2: use random long-only weights to show the feasible set.
# TODO: 计算协方差矩阵，并使用随机长仓权重生成可行集。

rng = np.random.default_rng(20260914)
weights = rng.dirichlet(np.ones(len(codes)), size=50_000)
# TODO: 计算每组随机权重对应的组合年化平均收益率和年化标准差，并保存到portfolio_mean和portfolio_vol中。

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.scatter(portfolio_vol, portfolio_mean, s=4, alpha=0.12, color="gray")
asset_vol = np.sqrt(np.diag(sigma))
ax.scatter(asset_vol, mu, s=35)
for code, x_value, y_value in zip(codes, asset_vol, mu):
    ax.annotate(
        names[code], (x_value, y_value), xytext=(4, 3),
        textcoords="offset points", fontsize=8
    )
ax.set(xlabel="年化标准差", ylabel="年化平均收益率")
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part3_feasible_set.png", dpi=200)
plt.close(fig)

# Task 3: estimate 252-observation rolling Beta separately for each stock.
# TODO: 计算每只股票的滚动Beta，并保存到rolling_beta DataFrame中。

fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
for ax, code in zip(axes.flat, codes):
    ax.plot(rolling_beta.index, rolling_beta[code], linewidth=1.1)
    ax.axhline(1, color="firebrick", linestyle="--", linewidth=0.8)
    ax.set_title(f"{names[code]}  {code}", fontsize=10, loc="left")
    ax.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    ax.grid(alpha=0.2)
for ax in axes.flat[len(codes):]:
    ax.set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part3_rolling_beta.png", dpi=200)
plt.close(fig)

# Task 4: use the minimum- and maximum-Beta stocks to create Beta=1.
# TODO: 计算Beta=1组合的权重，并保存到result DataFrame中。

# TODO: 计算Beta=1组合的累计净值，并与沪深300指数进行对比。
# 可以自己定义一个Beta=1的组合权重。

result = pd.DataFrame(
    {
        "name": [names[code] for code in codes],
        "weight": weight,
        "beta": beta,
        "beta_contribution": weight * beta,
    }
)
result.to_csv(OUTPUT_DIR / "part3_beta_one_weights.csv", encoding="utf-8-sig")

selected = result[result["weight"] > 0]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].bar(selected["name"], selected["weight"])
axes[0].set_ylabel("组合权重")
axes[1].bar(selected["name"], selected["beta_contribution"], color="firebrick")
axes[1].set_ylabel(r"Beta贡献 $w_i\beta_i$")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part3_beta_one_portfolio.png", dpi=200)
plt.close(fig)

portfolio_ret = common[codes] @ weight
comparison_nav = pd.DataFrame(
    {
        "Beta=1组合": (1 + portfolio_ret).cumprod(),
        "沪深300买入持有": (1 + common["market"]).cumprod(),
    }
)
comparison_nav = comparison_nav / comparison_nav.iloc[0]
ax = comparison_nav.plot(figsize=(10, 4.8), linewidth=2)
ax.set(xlabel="日期", ylabel="累计净值")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_beta_one_nav.png", dpi=200)
plt.close()

print("\nBeta=1 portfolio:")
print(selected.round(4))
print("sum of weights =", round(weight.sum(), 6))
print("portfolio beta =", round(weight @ beta, 6))
print(f"Figures and tables saved to: {OUTPUT_DIR}")
