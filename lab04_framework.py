"""Lecture 04: offline mean-variance laboratory; run the whole file in Spyder."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"  # Shared with lab01-lab03.
OUTPUT_DIR = ROOT / "output" / "lab04"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

names = {
    "sz.300750": "宁德时代", "sz.300308": "中际旭创",
    "sh.600519": "贵州茅台", "sh.601318": "中国平安",
    "sz.300502": "新易盛", "sh.601899": "紫金矿业",
    "sh.600036": "招商银行", "sz.000333": "美的集团",
    "sh.603259": "药明康德", "sh.688256": "寒武纪",
}  # Fixed 2026-08-31 teaching universe; contains look-ahead selection.
codes = list(names)
adjusted_close = {}
for code in codes:
    stock = pd.read_csv(DATA_DIR / f"{code}.csv", parse_dates=["date"])
    stock = stock.sort_values("date").drop_duplicates("date")
    price = pd.to_numeric(stock["close"], errors="coerce")
    factor = pd.to_numeric(stock["backAdjustFactor"], errors="coerce")
    adjusted_close[code] = pd.Series((price * factor).to_numpy(), index=stock["date"])
prices = pd.DataFrame(adjusted_close).sort_index()
stock_ret = prices.pct_change(fill_method=None).loc["2020":"2025"]
index_data = pd.read_csv(DATA_DIR / "csi300_daily.csv", parse_dates=["date"])
market_ret = index_data.sort_values("date").set_index("date")["close"]
market_ret = pd.to_numeric(market_ret).pct_change(fill_method=None).rename("market")
common = pd.concat([stock_ret, market_ret], axis=1).dropna()
R = common[codes]  # Same common-date sample as lab03.
assert len(R) > 252 and R.index.is_unique
print("Common sample:", R.index.min(), R.index.max(), len(R))
pd.Series({"start": str(R.index.min().date()), "end": str(R.index.max().date()),
           "n_obs": len(R), "rf_ann": 0.02}).to_csv(OUTPUT_DIR / "sample_info.csv")

# %% Preparation: annual inputs and random feasible portfolios
# TODO BEGIN L04-01
# 年化参数与随机可行域：用slide中同编号的完整代码替换下一行。
raise NotImplementedError("请先填入 L04-01")
# TODO END L04-01
assert np.linalg.eigvalsh(Sigma).min() > 0

# %% Task 1a: global minimum variance
# TODO BEGIN L04-02
# 求解GMV：用slide中同编号的完整代码替换下一行。
raise NotImplementedError("请先填入 L04-02")
# TODO END L04-02

# %% Task 1b: minimum variance at each target return
# TODO BEGIN L04-03
# 逐点计算有效前沿：用slide中同编号的完整代码替换下一行。
raise NotImplementedError("请先填入 L04-03")
# TODO END L04-03
frontier = pd.DataFrame(frontier_w, columns=codes)
frontier.insert(0, "vol_ann", frontier_vol)
frontier.insert(0, "mean_ann", targets)
frontier.to_csv(OUTPUT_DIR / "frontier.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.scatter(random_vol, random_mean, s=4, alpha=0.12, color="gray", label="随机组合")
ax.scatter(np.sqrt(np.diag(Sigma)), mu, s=28, label="单只股票")
ax.plot(frontier_vol, targets, color="firebrick", lw=2.5, label="非卖空有效前沿")
ax.scatter(vol_gmv, mu_gmv, marker="D", s=70, label="GMV")
ax.set(xlabel="年化标准差", ylabel="年化平均收益率（样本）")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "frontier.png", dpi=180)
plt.close(fig)

# %% Task 2: maximum Sharpe and portfolio comparison
# TODO BEGIN L04-04
# 最大Sharpe与等权比较：用slide中同编号的完整代码替换下一行。
raise NotImplementedError("请先填入 L04-04")
# TODO END L04-04
assert summary.loc["GMV", "vol_ann"] <= summary["vol_ann"].min() + 1e-7
assert summary.loc["最大Sharpe", "sharpe_ann"] >= summary["sharpe_ann"].max() - 1e-7
assert summary.loc["最大Sharpe", "sharpe_ann"] >= ((targets - rf) / frontier_vol).max() - 1e-5
weights.to_csv(OUTPUT_DIR / "weights.csv", encoding="utf-8-sig")
summary.to_csv(OUTPUT_DIR / "summary.csv", encoding="utf-8-sig")
ax = weights.rename(index=names).plot.barh(figsize=(10, 5.8))
ax.set(xlabel="组合权重")
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weights.png", dpi=180)
plt.close()

# %% Task 3: optimal sample CAL
# TODO BEGIN L04-05
# 绘制最优CAL：用slide中同编号的完整代码替换下一行。
raise NotImplementedError("请先填入 L04-05")
# TODO END L04-05
ax.set(xlabel="年化标准差", ylabel="年化平均收益率（样本）")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cal.png", dpi=180)
plt.close(fig)
pd.DataFrame({"vol_ann": cal_vol, "mean_ann": cal_return}).to_csv(
    OUTPUT_DIR / "cal.csv", index=False)
assert abs(rf + sharpe_t * vol_t - mu_t) < 1e-10

# %% Task 4: re-estimate and optimize in two windows
# TODO BEGIN L04-06
# 窗口敏感性：用slide中同编号的完整代码替换下一行。
raise NotImplementedError("请先填入 L04-06")
# TODO END L04-06
window_weights.to_csv(OUTPUT_DIR / "window_weights.csv", encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
for ax, model in zip(axes, ["GMV", "最大Sharpe"]):
    columns = [f"{model} {label}" for label in windows]
    window_weights[columns].rename(index=names).plot.barh(ax=ax)
    ax.set(title=model, xlabel="组合权重")
    ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "window_weights.png", dpi=180)
plt.close(fig)
print("Output:", OUTPUT_DIR)
print("Sample optimization only; no out-of-sample performance claim.")
