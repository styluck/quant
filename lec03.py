"""Lecture 3 reference code: estimate CAPM exposures for two stocks."""

from pathlib import Path

import baostock as bs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCK_DIR = PROJECT_ROOT / "data" / "raw" / "sse_daily"
MARKET_FILE = PROJECT_ROOT / "data" / "raw" / "csi300_daily.csv"
TOP10_FILE = PROJECT_ROOT / "data" / "raw" / "csi300_top10_20260831.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "lec03"
CODES = ["sh.600000", "sh.601127"]
DATA_START = "2018-01-01"  # 为2020年初的252日滚动窗口预留充足数据
PLOT_START = "2020-01-01"
PLOT_END = "2025-12-31"


def download_stock_data(codes):
    """Download raw daily prices and backward-adjustment factors from BaoStock."""
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    missing_codes = [code for code in codes if not (STOCK_DIR / f"{code}.csv").exists()]
    if not missing_codes:
        return

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

    fields = "date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST"
    try:
        for code in missing_codes:
            result = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=DATA_START,
                end_date=PLOT_END,
                frequency="d",
                adjustflag="3",  # 下载不复权价格，随后显式合并复权因子
            )
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0":
                raise RuntimeError(f"Price query failed for {code}: {result.error_msg}")
            daily = pd.DataFrame(rows, columns=fields.split(","))
            daily["date"] = pd.to_datetime(daily["date"])

            factor_result = bs.query_adjust_factor(
                code=code,
                start_date="1990-01-01",  # 保留2019年以前最近一次生效的累计因子
                end_date=PLOT_END,
            )
            factor_rows = []
            while factor_result.error_code == "0" and factor_result.next():
                factor_rows.append(factor_result.get_row_data())
            if factor_result.error_code != "0":
                raise RuntimeError(
                    f"Adjustment-factor query failed for {code}: {factor_result.error_msg}"
                )
            factors = pd.DataFrame(factor_rows, columns=factor_result.fields)
            factors["date"] = pd.to_datetime(factors["dividOperateDate"])
            factors["backAdjustFactor"] = pd.to_numeric(
                factors["backAdjustFactor"], errors="coerce"
            )
            factors = factors[["date", "backAdjustFactor"]].sort_values("date")

            daily = pd.merge_asof(
                daily.sort_values("date"), factors, on="date", direction="backward"
            )
            daily["backAdjustFactor"] = daily["backAdjustFactor"].fillna(1.0)
            daily.to_csv(STOCK_DIR / f"{code}.csv", index=False)
    finally:
        bs.logout()


def load_stock_returns(codes):
    """Load BaoStock CSV files and construct back-adjusted daily returns."""
    frames = []
    for code in codes:
        file_path = STOCK_DIR / f"{code}.csv"
        stock = pd.read_csv(file_path, parse_dates=["date"])
        required = {"date", "code", "close", "backAdjustFactor"}
        missing = required.difference(stock.columns)
        if missing:
            raise ValueError(f"{file_path.name} missing columns: {sorted(missing)}")
        frames.append(stock)

    panel = pd.concat(frames, ignore_index=True)
    numeric = ["close", "backAdjustFactor"]
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="coerce")
    panel["adj_close"] = panel["close"] * panel["backAdjustFactor"]
    close = panel.pivot(index="date", columns="code", values="adj_close").sort_index()
    return close.pct_change(fill_method=None)


def load_market_return():
    """Load CSI 300 prices prepared in Lecture 1 and compute daily returns."""
    market = pd.read_csv(MARKET_FILE, parse_dates=["date"]).set_index("date")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    return market["close"].pct_change(fill_method=None).rename("market")


def estimate_capm(stock_return, market_return):
    """Estimate a market model and report HAC inference and risk statistics."""
    data = pd.concat([stock_return, market_return], axis=1).dropna()
    design = sm.add_constant(data["market"])
    model = sm.OLS(data.iloc[:, 0], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    stats = pd.Series(
        {
            "alpha_ann": model.params["const"] * 252,
            "beta": model.params["market"],
            "alpha_t": model.tvalues["const"],
            "beta_t": model.tvalues["market"],
            "r2": model.rsquared,
            "resid_vol_ann": model.resid.std() * np.sqrt(252),
            "n_obs": len(data),
        }
    )
    return model, data, stats


def plot_scatter(models, samples, codes):
    """Plot stock excess returns against market excess returns."""
    fig, axes = plt.subplots(1, len(codes), figsize=(11, 4.5), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, code in zip(axes, codes):
        data = samples[code]
        design = sm.add_constant(data["market"])
        order = data["market"].sort_values().index
        axis.scatter(data["market"], data[code], s=8, alpha=0.25)
        axis.plot(
            data.loc[order, "market"],
            models[code].predict(design.loc[order]),
            color="red",
        )
        axis.set_title(f"{code}: beta={models[code].params['market']:.2f}")
        axis.set_xlabel("Market excess return")
    axes[0].set_ylabel("Stock excess return")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "capm_scatter.png", dpi=200)
    


def plot_rolling_beta(stock_returns, market_return, window=252):
    """Estimate rolling covariance betas and save their time series."""
    market_var = market_return.rolling(window).var()
    rolling_beta = pd.DataFrame(index=stock_returns.index)
    for code in stock_returns.columns:
        rolling_cov = stock_returns[code].rolling(window).cov(market_return)
        rolling_beta[code] = rolling_cov / market_var

    rolling_beta = rolling_beta.loc[PLOT_START:PLOT_END].dropna(how="all")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    axis = rolling_beta.rename(
        columns={"sh.600000": "浦发银行", "sh.601127": "赛力斯"}
    ).plot(figsize=(11, 4), linewidth=1.7, color=["#225896", "#aa2d2d"])
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    axis.set_ylabel("Beta")
    axis.set_xlabel("日期")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rolling_beta_2020_2025.png", dpi=220)
    
    rolling_beta.to_csv(OUTPUT_DIR / "rolling_beta_2020_2025.csv")


def configure_chinese_font():
    """Use an installed Chinese font for figures."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_top10_mean_variance(stock_returns, names):
    """Plot each stock in annualized mean-standard-deviation space."""
    sample = stock_returns.loc[PLOT_START:PLOT_END]
    stats = pd.DataFrame(
        {
            "mean_ann": sample.mean() * 252,
            "vol_ann": sample.std() * np.sqrt(252),
            "n_obs": sample.count(),
        }
    )
    stats.to_csv(OUTPUT_DIR / "top10_mean_variance_stats.csv")

    configure_chinese_font()
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.scatter(stats["vol_ann"], stats["mean_ann"], s=55, color="#225896")
    offsets = {
        "sh.601318": (6, -18),
        "sh.600519": (6, -5),
        "sh.600036": (6, 12),
        "sz.000333": (6, 22),
    }
    for code, row in stats.iterrows():
        axis.annotate(
            names[code],
            (row["vol_ann"], row["mean_ann"]),
            xytext=offsets.get(code, (5, 4)),
            textcoords="offset points",
            fontsize=9,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("年化标准差")
    axis.set_ylabel("年化平均收益率")
    axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axis.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top10_mean_variance.png", dpi=220)
    
    return stats


def solve_min_variance(covariance, expected_return=None, beta=None):
    """Solve a long-only minimum-variance portfolio with optional constraints."""
    n_assets = len(covariance)
    objective = lambda w: w @ covariance @ w
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if expected_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ expected_return[0] - expected_return[1]}
        )
    if beta is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ beta - 1})
    result = minimize(
        objective,
        np.repeat(1 / n_assets, n_assets),
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return result.x


def plot_long_only_feasible_set(common_returns, names):
    """Approximate the long-only mean-variance feasible set by random sampling."""
    mean_ann = common_returns.mean().to_numpy() * 252
    cov_ann = common_returns.cov().to_numpy() * 252
    rng = np.random.default_rng(20260914)
    weights = rng.dirichlet(np.ones(len(mean_ann)), size=50000)
    portfolio_return = weights @ mean_ann
    portfolio_vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_ann, weights))

    configure_chinese_font()
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.scatter(
        portfolio_vol,
        portfolio_return,
        s=4,
        alpha=0.12,
        color="#7f8c8d",
        label="非卖空组合可行域（随机抽样）",
    )
    asset_vol = np.sqrt(np.diag(cov_ann))
    axis.scatter(asset_vol, mean_ann, s=35, color="#225896", label="单只股票")
    offsets = {
        "sh.601318": (5, 9),
        "sh.600519": (5, -15),
        "sh.600036": (5, 10),
        "sz.000333": (5, 19),
    }
    for code, x, y in zip(common_returns.columns, asset_vol, mean_ann):
        axis.annotate(
            names[code],
            (x, y),
            xytext=offsets.get(code, (4, 3)),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("年化标准差")
    axis.set_ylabel("年化平均收益率")
    axis.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axis.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top10_long_only_feasible_set.png", dpi=220)
    
    return mean_ann, cov_ann


def plot_top10_rolling_beta(stock_returns, market_return, names, window=252):
    """Plot pairwise 252-observation rolling betas for the ten-stock universe."""
    rolling = {}
    for code in stock_returns.columns:
        pair = pd.concat([stock_returns[code], market_return], axis=1).dropna()
        rolling[code] = (
            pair[code].rolling(window).cov(pair["market"])
            / pair["market"].rolling(window).var()
        )
    rolling_beta = pd.DataFrame(rolling).loc[PLOT_START:PLOT_END]
    rolling_beta.to_csv(OUTPUT_DIR / "top10_rolling_beta_2020_2025.csv")

    configure_chinese_font()
    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
    for axis, code in zip(axes.flat, stock_returns.columns):
        axis.plot(rolling_beta.index, rolling_beta[code], color="#225896", linewidth=1.1)
        axis.axhline(1, color="#aa2d2d", linestyle="--", linewidth=0.8)
        axis.set_title(f"{names[code]}  {code}", fontsize=10, loc="left")
        axis.grid(alpha=0.2)
        axis.set_ylabel("Beta")
        axis.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    for axis in axes.flat[len(stock_returns.columns):]:
        axis.set_visible(False)
    for index in (7, 8, 9):
        axes.flat[index].tick_params(labelbottom=True)
        axes.flat[index].set_xlabel("日期")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top10_rolling_beta_2020_2025.png", dpi=220)
    
    return rolling_beta


def plot_beta_one_portfolio(mean_ann, cov_ann, beta, codes, names):
    """Construct and plot the long-only minimum-variance portfolio with beta one."""
    weights = solve_min_variance(cov_ann, beta=beta)
    portfolio_return = weights @ mean_ann
    portfolio_vol = np.sqrt(weights @ cov_ann @ weights)
    portfolio_beta = weights @ beta
    result = pd.DataFrame(
        {"code": codes, "name": [names[c] for c in codes], "weight": weights, "beta": beta}
    )
    result.to_csv(OUTPUT_DIR / "top10_beta1_minvar_weights.csv", index=False)

    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ordered = result.sort_values("weight")
    axes[0].barh(ordered["name"], ordered["weight"], color="#225896")
    axes[0].xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[0].set_xlabel("组合权重")
    axes[0].grid(axis="x", alpha=0.25)

    beta_contribution = ordered["weight"] * ordered["beta"]
    axes[1].barh(ordered["name"], beta_contribution, color="#aa2d2d")
    axes[1].set_xlabel(r"Beta 贡献 $w_i\beta_i$")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].text(
        0.98,
        0.04,
        rf"$\sum_i w_i\beta_i={portfolio_beta:.2f}$",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="#aa2d2d",
    )
    fig.suptitle(f"Beta=1 最小方差组合：年化收益 {portfolio_return:.1%}，年化波动 {portfolio_vol:.1%}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top10_beta1_minvar.png", dpi=220)
    
    return result


def plot_simple_beta_one_portfolio(common, mean_ann, cov_ann, beta, codes, names):
    """Construct a beta-one portfolio from the minimum- and maximum-beta stocks."""
    beta = pd.Series(beta, index=codes)
    below_code = beta.idxmin()
    above_code = beta.idxmax()
    beta_below = beta[below_code]
    beta_above = beta[above_code]

    weight_below = (beta_above - 1) / (beta_above - beta_below)
    weight_above = (1 - beta_below) / (beta_above - beta_below)
    weights = pd.Series(0.0, index=codes)
    weights.loc[below_code] = weight_below
    weights.loc[above_code] = weight_above

    portfolio_return = weights.to_numpy() @ mean_ann
    portfolio_vol = np.sqrt(weights.to_numpy() @ cov_ann @ weights.to_numpy())
    result = pd.DataFrame(
        {
            "code": codes,
            "name": [names[code] for code in codes],
            "weight": weights.to_numpy(),
            "beta": beta.to_numpy(),
            "beta_contribution": weights.to_numpy() * beta.to_numpy(),
        }
    )
    result.to_csv(OUTPUT_DIR / "top10_beta1_simple_weights.csv", index=False)

    selected = result[result["weight"] > 0].copy()
    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(selected["name"], selected["weight"], color="#225896")
    axes[0].yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    axes[0].set_ylabel("组合权重")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(selected["name"], selected["beta_contribution"], color="#aa2d2d")
    axes[1].set_ylabel(r"Beta贡献 $w_i\beta_i$")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(
        0.98,
        0.94,
        rf"$\beta_p={weights @ beta:.2f}$",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color="#aa2d2d",
    )
    fig.suptitle(
        f"两股票构造Beta=1组合：年化收益{portfolio_return:.1%}，"
        f"年化波动{portfolio_vol:.1%}"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top10_beta1_simple.png", dpi=220)
    

    portfolio_daily = common[codes] @ weights
    nav = pd.DataFrame(
        {
            "Beta=1组合": (1 + portfolio_daily).cumprod(),
            "沪深300买入持有": (1 + common["market"]).cumprod(),
        }
    )
    nav = nav / nav.iloc[0]
    nav.to_csv(OUTPUT_DIR / "top10_beta1_simple_nav.csv")

    fig, axis = plt.subplots(figsize=(10, 4.8))
    nav.plot(ax=axis, linewidth=2, color=["#aa2d2d", "#225896"])
    axis.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    axis.set_xlabel("日期")
    axis.set_ylabel("累计净值")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top10_beta1_simple_nav.png", dpi=220)
    
    return result


def run_top10_practice(market_return):
    """Generate all four classroom-practice outputs."""
    top10 = pd.read_csv(TOP10_FILE)
    codes = top10["code"].tolist()
    names = top10.set_index("code")["name"].to_dict()
    download_stock_data(codes)
    stock_returns = load_stock_returns(codes).loc[:PLOT_END]

    plot_top10_mean_variance(stock_returns, names)
    common = pd.concat([stock_returns.loc[PLOT_START:PLOT_END], market_return], axis=1).dropna()
    common_stocks = common[codes]
    mean_ann, cov_ann = plot_long_only_feasible_set(common_stocks, names)
    plot_top10_rolling_beta(stock_returns, market_return, names)
    beta = common_stocks.apply(lambda x: x.cov(common["market"])) / common["market"].var()
    plot_simple_beta_one_portfolio(common, mean_ann, cov_ann, beta, codes, names)
    plot_beta_one_portfolio(mean_ann, cov_ann, beta.to_numpy(), codes, names)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top10_codes = pd.read_csv(TOP10_FILE)["code"].tolist()
    download_stock_data(list(dict.fromkeys(CODES + top10_codes)))
    stock_returns = load_stock_returns(CODES)
    market_return = load_market_return()

    common = pd.concat([stock_returns, market_return], axis=1).dropna()
    stock_returns = common[CODES]
    market_return = common["market"]

    models = {}
    samples = {}
    rows = []
    for code in CODES:
        model, data, stats = estimate_capm(stock_returns[code], market_return)
        models[code] = model
        samples[code] = data.rename(columns={data.columns[0]: code})
        stats.name = code
        rows.append(stats)

    comparison = pd.DataFrame(rows)
    comparison.index.name = "code"
    comparison.to_csv(OUTPUT_DIR / "capm_comparison.csv")
    print(comparison.round(3))

    plot_scatter(models, samples, CODES)
    plot_rolling_beta(stock_returns, market_return)
    run_top10_practice(load_market_return())


if __name__ == "__main__":
    main()
