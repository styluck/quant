"""Lecture 4 reference code: real-data mean-variance optimization."""

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
OUTPUT_DIR = PROJECT_ROOT / "output" / "lec04"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
RANDOM_SEED = 20260914


def prepare_sample():
    """Load the Lecture 3 universe and form a common-date return sample."""
    top10 = pd.read_csv(TOP10_FILE)
    codes = top10["code"].tolist()
    names = top10.set_index("code")["name"].to_dict()
    download_stock_data(codes)
    stock_return = load_stock_returns(codes).loc[PLOT_START:PLOT_END]
    market_return = load_market_return().loc[PLOT_START:PLOT_END]
    common = pd.concat(
        [stock_return[codes], market_return.rename("market")], axis=1
    ).dropna()
    return codes, names, common


def portfolio_statistics(weight, mean_return, covariance, risk_free=RISK_FREE_RATE):
    """Return annual mean, volatility, and Sharpe ratio for one portfolio."""
    mean = float(weight @ mean_return)
    volatility = float(np.sqrt(weight @ covariance @ weight))
    sharpe = (mean - risk_free) / volatility
    return mean, volatility, sharpe


def solve_minimum_variance(covariance, mean_return=None, target_return=None):
    """Solve a long-only minimum-variance problem."""
    n_assets = len(covariance)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: w @ mean_return - target_return,
            }
        )
    result = minimize(
        lambda w: w @ covariance @ w,
        np.repeat(1 / n_assets, n_assets),
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def solve_maximum_sharpe(mean_return, covariance, risk_free=RISK_FREE_RATE):
    """Solve the long-only maximum-Sharpe portfolio."""
    n_assets = len(mean_return)

    def negative_sharpe(weight):
        return -portfolio_statistics(
            weight, mean_return, covariance, risk_free
        )[2]

    result = minimize(
        negative_sharpe,
        np.repeat(1 / n_assets, n_assets),
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def calculate_frontier(mean_return, covariance, n_points=80):
    """Calculate the upper long-only frontier from the GMV return upward."""
    gmv_weight = solve_minimum_variance(covariance)
    gmv_return = gmv_weight @ mean_return
    targets = np.linspace(gmv_return, mean_return.max(), n_points)
    weights = np.vstack(
        [
            solve_minimum_variance(covariance, mean_return, target)
            for target in targets
        ]
    )
    volatility = np.sqrt(
        np.einsum("ij,jk,ik->i", weights, covariance, weights)
    )
    return targets, volatility, weights, gmv_weight


def random_feasible_set(mean_return, covariance, n_portfolios=50000):
    """Sample long-only fully invested portfolios from a Dirichlet distribution."""
    rng = np.random.default_rng(RANDOM_SEED)
    weights = rng.dirichlet(np.ones(len(mean_return)), size=n_portfolios)
    means = weights @ mean_return
    volatility = np.sqrt(
        np.einsum("ij,jk,ik->i", weights, covariance, weights)
    )
    return weights, means, volatility


def plot_real_frontier(common, codes, names):
    """Plot the real-data feasible set, exact frontier, and key portfolios."""
    sample = common[codes]
    mean_return = sample.mean().to_numpy() * TRADING_DAYS
    covariance = sample.cov().to_numpy() * TRADING_DAYS
    _, random_mean, random_vol = random_feasible_set(mean_return, covariance)
    targets, frontier_vol, _, gmv_weight = calculate_frontier(
        mean_return, covariance
    )
    tangent_weight = solve_maximum_sharpe(mean_return, covariance)
    gmv = portfolio_statistics(gmv_weight, mean_return, covariance)
    tangent = portfolio_statistics(tangent_weight, mean_return, covariance)

    configure_chinese_font()
    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.scatter(random_vol, random_mean, s=4, alpha=0.10, color="#7f8c8d")
    axis.plot(frontier_vol, targets, color="#aa2d2d", linewidth=2.7, label="有效前沿")
    axis.scatter(gmv[1], gmv[0], s=90, color="#17804f", marker="D", label="全局最小方差")
    axis.scatter(tangent[1], tangent[0], s=110, color="#e08b22", marker="*", label="最大Sharpe")
    axis.set_xlabel("年化标准差")
    axis.set_ylabel("年化平均收益率")
    axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axis.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_frontier_key_portfolios.png", dpi=220)
    plt.close(fig)

    result = pd.DataFrame(
        {
            "code": codes,
            "name": [names[code] for code in codes],
            "gmv_weight": gmv_weight,
            "max_sharpe_weight": tangent_weight,
        }
    )
    result.to_csv(
        OUTPUT_DIR / "key_portfolio_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(
        [gmv, tangent],
        index=["GMV", "Maximum Sharpe"],
        columns=["mean_ann", "vol_ann", "sharpe"],
    )
    summary.to_csv(OUTPUT_DIR / "key_portfolio_summary.csv")
    return mean_return, covariance, targets, frontier_vol, result, summary


def plot_cal(targets, frontier_vol, summary):
    """Add the optimal sample CAL through the maximum-Sharpe portfolio."""
    tangent = summary.loc["Maximum Sharpe"]
    cml_vol = np.linspace(0, max(frontier_vol.max(), tangent["vol_ann"] * 1.3), 100)
    cml_return = RISK_FREE_RATE + tangent["sharpe"] * cml_vol

    configure_chinese_font()
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.plot(frontier_vol, targets, color="#225896", linewidth=2.7, label="风险资产有效前沿")
    axis.plot(cml_vol, cml_return, color="#aa2d2d", linewidth=2.5, label="样本股票池的最优CAL")
    axis.scatter(0, RISK_FREE_RATE, s=65, color="#17804f", zorder=3, label="无风险资产")
    axis.scatter(
        tangent["vol_ann"], tangent["mean_ann"],
        s=120, color="#e08b22", marker="*", zorder=3, label="切点组合"
    )
    axis.set_xlabel("年化标准差")
    axis.set_ylabel("年化平均收益率")
    axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axis.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_frontier_cal.png", dpi=220)
    plt.close(fig)


def plot_key_weights(result):
    """Compare the holdings of the GMV and maximum-Sharpe portfolios."""
    configure_chinese_font()
    ordered = result.sort_values("max_sharpe_weight")
    y_position = np.arange(len(ordered))
    width = 0.38
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.barh(
        y_position - width / 2,
        ordered["gmv_weight"],
        height=width,
        label="全局最小方差",
        color="#225896",
    )
    axis.barh(
        y_position + width / 2,
        ordered["max_sharpe_weight"],
        height=width,
        label="最大Sharpe",
        color="#aa2d2d",
    )
    axis.set_yticks(y_position, ordered["name"])
    axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axis.set_xlabel("组合权重")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "key_portfolio_weights.png", dpi=220)
    plt.close(fig)


def estimate_window_weights(sample, codes):
    """Estimate GMV and maximum-Sharpe weights in one sample window."""
    mean_return = sample[codes].mean().to_numpy() * TRADING_DAYS
    covariance = sample[codes].cov().to_numpy() * TRADING_DAYS
    return (
        solve_minimum_variance(covariance),
        solve_maximum_sharpe(mean_return, covariance),
    )


def plot_estimation_instability(common, codes, names):
    """Compare optimized weights across two adjacent estimation windows."""
    early = common.loc[:"2022-12-31"]
    late = common.loc["2023-01-01":]
    early_gmv, early_tangent = estimate_window_weights(early, codes)
    late_gmv, late_tangent = estimate_window_weights(late, codes)
    comparison = pd.DataFrame(
        {
            "name": [names[code] for code in codes],
            "GMV 2020-2022": early_gmv,
            "GMV 2023-2025": late_gmv,
            "MaxSharpe 2020-2022": early_tangent,
            "MaxSharpe 2023-2025": late_tangent,
        },
        index=codes,
    )
    comparison.to_csv(
        OUTPUT_DIR / "window_weight_comparison.csv", encoding="utf-8-sig"
    )

    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    y_position = np.arange(len(codes))
    for axis, prefix, title in zip(
        axes,
        ["GMV", "MaxSharpe"],
        ["全局最小方差组合", "最大Sharpe组合"],
    ):
        axis.barh(
            y_position - 0.18,
            comparison[f"{prefix} 2020-2022"],
            height=0.36,
            label="2020--2022",
        )
        axis.barh(
            y_position + 0.18,
            comparison[f"{prefix} 2023-2025"],
            height=0.36,
            label="2023--2025",
            color="#aa2d2d",
        )
        axis.set_title(title)
        axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        axis.set_xlabel("组合权重")
        axis.grid(axis="x", alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_yticks(y_position, comparison["name"])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "estimation_window_weights.png", dpi=220)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    codes, names, common = prepare_sample()
    _, _, targets, frontier_vol, result, summary = plot_real_frontier(
        common, codes, names
    )
    plot_cal(targets, frontier_vol, summary)
    plot_key_weights(result)
    plot_estimation_instability(common, codes, names)
    print(summary.round(4))


if __name__ == "__main__":
    main()
