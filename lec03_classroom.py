"""Lecture 3 classroom script: mean-variance analysis, beta, and optimization.

Run this file from the ``quant_trading`` Spyder project.  Data downloading and
cleaning are prepared in ``lec03.py``; the four task functions below contain
the calculations discussed in class and save their results separately.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lec03 import (
    PLOT_END,
    PLOT_START,
    TOP10_FILE,
    configure_chinese_font,
    download_stock_data,
    load_market_return,
    load_stock_returns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "lec03" / "classroom"
TRADING_DAYS = 252
ROLLING_WINDOW = 252


def prepare_data():
    """Download data and return individual and common-date return samples."""
    top10 = pd.read_csv(TOP10_FILE)
    codes = top10["code"].tolist()
    names = top10.set_index("code")["name"].to_dict()

    download_stock_data(codes)
    stock_return = load_stock_returns(codes).loc[:PLOT_END]
    market_return = load_market_return().loc[:PLOT_END]

    study_sample = stock_return.loc[PLOT_START:PLOT_END]
    common = pd.concat(
        [study_sample, market_return.rename("market")], axis=1
    ).dropna()
    print(
        "Common sample:",
        common.index.min().date(),
        "to",
        common.index.max().date(),
        f"({len(common)} trading days)",
    )
    return codes, names, stock_return, market_return, common


def task1_mean_variance(stock_return, codes, names):
    """Task 1: locate the ten stocks in annual mean-volatility space."""
    sample = stock_return.loc[PLOT_START:PLOT_END, codes]
    statistics = pd.DataFrame(
        {
            "mean_ann": sample.mean() * TRADING_DAYS,
            "vol_ann": sample.std() * np.sqrt(TRADING_DAYS),
            "n_obs": sample.count(),
        }
    )
    statistics.to_csv(OUTPUT_DIR / "task1_mean_variance.csv", encoding="utf-8-sig")

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.scatter(statistics["vol_ann"], statistics["mean_ann"], s=55)
    for code, row in statistics.iterrows():
        axis.annotate(
            names[code],
            (row["vol_ann"], row["mean_ann"]),
            xytext=(5, 4),
            textcoords="offset points",
        )
    axis.set(xlabel="年化标准差", ylabel="年化平均收益率")
    axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axis.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "task1_mean_variance.png", dpi=220)
    plt.close(fig)
    return statistics


def solve_min_variance(covariance, target_return=None, mean_return=None, beta=None):
    """Solve a long-only minimum-variance portfolio under given constraints."""
    n_assets = len(covariance)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: w @ mean_return - target_return,
            }
        )
    if beta is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ beta - 1})

    result = minimize(
        lambda w: w @ covariance @ w,
        x0=np.repeat(1 / n_assets, n_assets),
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def task2_feasible_set(common, codes, names):
    """Task 2: approximate the long-only feasible set by random sampling."""
    stock_sample = common[codes]
    mean_ann = stock_sample.mean().to_numpy() * TRADING_DAYS
    cov_ann = stock_sample.cov().to_numpy() * TRADING_DAYS

    rng = np.random.default_rng(20260914)
    weights = rng.dirichlet(np.ones(len(codes)), size=50000)
    portfolio_mean = weights @ mean_ann
    portfolio_vol = np.sqrt(
        np.einsum("ij,jk,ik->i", weights, cov_ann, weights)
    )

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.scatter(portfolio_vol, portfolio_mean, s=4, alpha=0.12, color="gray")
    asset_vol = np.sqrt(np.diag(cov_ann))
    axis.scatter(asset_vol, mean_ann, s=35)
    for code, x_value, y_value in zip(codes, asset_vol, mean_ann):
        axis.annotate(
            names[code], (x_value, y_value), xytext=(4, 3),
            textcoords="offset points", fontsize=8
        )
    axis.set(xlabel="年化标准差", ylabel="年化平均收益率")
    axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axis.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "task2_feasible_set.png", dpi=220)
    plt.close(fig)
    return mean_ann, cov_ann


def task3_rolling_beta(stock_return, market_return, codes, names):
    """Task 3: estimate pairwise 252-observation rolling beta series."""
    rolling_beta = {}
    for code in codes:
        pair = pd.concat(
            [stock_return[code], market_return.rename("market")], axis=1
        ).dropna()
        rolling_beta[code] = (
            pair[code].rolling(ROLLING_WINDOW).cov(pair["market"])
            / pair["market"].rolling(ROLLING_WINDOW).var()
        )
    rolling_beta = pd.DataFrame(rolling_beta).loc[PLOT_START:PLOT_END]
    rolling_beta.to_csv(OUTPUT_DIR / "task3_rolling_beta.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
    for axis, code in zip(axes.flat, codes):
        axis.plot(rolling_beta.index, rolling_beta[code], linewidth=1.1)
        axis.axhline(1, color="firebrick", linestyle="--", linewidth=0.8)
        axis.set_title(f"{names[code]}  {code}", fontsize=10, loc="left")
        axis.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(codes):]:
        axis.set_visible(False)
    for index in (7, 8, 9):
        axes.flat[index].tick_params(labelbottom=True)
        axes.flat[index].set_xlabel("日期")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "task3_rolling_beta.png", dpi=220)
    plt.close(fig)
    return rolling_beta


def task4_beta_one_portfolio(common, codes, names, mean_ann, cov_ann):
    """Task 4: construct a beta-one portfolio from min- and max-beta stocks."""
    market = common["market"]
    beta = common[codes].apply(lambda series: series.cov(market)) / market.var()
    below_code = beta.idxmin()
    above_code = beta.idxmax()

    weight_below = (beta[above_code] - 1) / (
        beta[above_code] - beta[below_code]
    )
    weight_above = (1 - beta[below_code]) / (
        beta[above_code] - beta[below_code]
    )
    weight = pd.Series(0.0, index=codes)
    weight.loc[below_code] = weight_below
    weight.loc[above_code] = weight_above

    portfolio_mean = weight.to_numpy() @ mean_ann
    portfolio_vol = np.sqrt(weight.to_numpy() @ cov_ann @ weight.to_numpy())
    result = pd.DataFrame(
        {
            "code": codes,
            "name": [names[code] for code in codes],
            "weight": weight.to_numpy(),
            "beta": beta.to_numpy(),
            "beta_contribution": weight.to_numpy() * beta.to_numpy(),
        }
    )
    result.to_csv(
        OUTPUT_DIR / "task4_beta_one_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(result[result["weight"] > 0].round(4))
    print("sum of weights =", weight.sum())
    print("portfolio beta =", weight @ beta)

    selected = result[result["weight"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(selected["name"], selected["weight"])
    axes[0].set_ylabel("组合权重")
    axes[1].bar(selected["name"], selected["beta_contribution"], color="firebrick")
    axes[1].set_ylabel(r"Beta贡献 $w_i\beta_i$")
    fig.suptitle(
        f"两股票构造Beta=1组合：年化收益{portfolio_mean:.1%}，"
        f"年化波动{portfolio_vol:.1%}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "task4_beta_one_portfolio.png", dpi=220)
    

    portfolio_daily = common[codes] @ weight
    nav = pd.DataFrame(
        {
            "Beta=1组合": (1 + portfolio_daily).cumprod(),
            "沪深300买入持有": (1 + common["market"]).cumprod(),
        }
    )
    nav = nav / nav.iloc[0]
    nav.to_csv(
        OUTPUT_DIR / "task4_beta_one_nav.csv",
        encoding="utf-8-sig",
    )
    axis = nav.plot(figsize=(10, 4.8), linewidth=2)
    axis.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    axis.set(xlabel="日期", ylabel="累计净值")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task4_beta_one_nav.png", dpi=220)
    
    return result


def extension_beta_one_min_variance(common, codes, names, mean_ann, cov_ann):
    """Extension: optimize the beta-one portfolio by minimizing variance."""
    market = common["market"]
    beta = common[codes].apply(lambda series: series.cov(market)) / market.var()
    weight = solve_min_variance(cov_ann, beta=beta.to_numpy())

    result = pd.DataFrame(
        {
            "code": codes,
            "name": [names[code] for code in codes],
            "weight": weight,
            "beta": beta.to_numpy(),
            "beta_contribution": weight * beta.to_numpy(),
        }
    )
    result.to_csv(
        OUTPUT_DIR / "extension_beta_one_minvar_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    portfolio_mean = weight @ mean_ann
    portfolio_vol = np.sqrt(weight @ cov_ann @ weight)
    print(result.round(4))
    print("sum of weights =", weight.sum())
    print("portfolio beta =", weight @ beta.to_numpy())
    print("annual mean =", portfolio_mean)
    print("annual volatility =", portfolio_vol)

    ordered = result.sort_values("weight")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].barh(ordered["name"], ordered["weight"])
    axes[0].set_xlabel("组合权重")
    axes[1].barh(ordered["name"], ordered["beta_contribution"], color="firebrick")
    axes[1].set_xlabel(r"Beta贡献 $w_i\beta_i$")
    fig.suptitle(
        f"Beta=1最小方差组合：年化收益{portfolio_mean:.1%}，"
        f"年化波动{portfolio_vol:.1%}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "extension_beta_one_minvar.png", dpi=220)
    
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    codes, names, stock_return, market_return, common = prepare_data()

    # 在Spyder中可以逐行执行以下四项任务，观察变量和中间结果。
    task1_mean_variance(stock_return, codes, names)
    mean_ann, cov_ann = task2_feasible_set(common, codes, names)
    task3_rolling_beta(stock_return, market_return, codes, names)
    task4_beta_one_portfolio(common, codes, names, mean_ann, cov_ann)


if __name__ == "__main__":
    main()
