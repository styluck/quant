"""第五讲实验：按 # %% 分段运行，包含单日回归 L05-BP、L05-BP-MV、评分选股 L05-00 和月度回归与风险练习 L05-01—06。"""

# %% L05-BP：单日 BP 横截面回归（本段可独立运行）
# 改编自统计分析课程 chp3_part3_answer.py 的 TODO 1。
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'lab05'
date, next_date = '2024-03-28', '2024-03-29'
pb = pd.read_csv(DATA_DIR / 'pb_sh.csv', index_col=0)
close = pd.read_csv(DATA_DIR / 'close_sh.csv', index_col=0)
adj = pd.read_csv(DATA_DIR / 'adj_factor_sh.csv', index_col=0)

bp = (1 / pb.loc[date]).rename('bp')
price = close * adj
return_next = (100 * (price.loc[next_date] / price.loc[date] - 1)).rename('return_next')
simple_cross = pd.concat([bp, return_next], axis=1)
simple_cross = simple_cross.replace([np.inf, -np.inf], np.nan)
simple_cross = simple_cross.loc[simple_cross.bp > 0].dropna()
simple_cross.index.name = 'code'
print(simple_cross.head())

# %% TODO：用 simple_cross 做含截距的 BP 一元回归，将拟合结果赋给 simple。
raise NotImplementedError("请完成 L05-BP")

# 用清理后的同一份样本绘制散点图与拟合直线。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(simple.summary())
print('样本数、截距、BP 系数、R2：', int(simple.nobs),
      simple.params['Intercept'], simple.params['bp'], simple.rsquared)
x = np.linspace(simple_cross['bp'].min(), simple_cross['bp'].max(), 200)
y_fit = simple.predict(pd.DataFrame({'bp': x}))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.scatter(simple_cross['bp'], simple_cross['return_next'],
           s=12, alpha=.45, color='#225896', label='每个点代表一只股票')
ax.plot(x, y_fit, color='#aa2d2d', linewidth=2, label='OLS 拟合直线')
ax.axhline(0, color='gray', linewidth=.6)
ax.set(xlabel=f'BP（{date}，原始值）',
       ylabel=f'次日收益（{next_date}，%）')
ax.legend(loc='upper right', frameon=False)
ax.text(.02, .98, f'N = {int(simple.nobs)}\nR² = {simple.rsquared:.4f}',
        transform=ax.transAxes, va='top', fontsize=11)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'daily_bp_regression.png', dpi=180, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / 'daily_bp_regression.pdf', bbox_inches='tight')
plt.close(fig)
simple_cross.to_csv(OUTPUT_DIR / 'daily_bp_sample.csv')
pd.Series({'date': date, 'next_date': next_date, 'n': int(simple.nobs),
           'intercept': simple.params['Intercept'], 'bp_coef': simple.params['bp'],
           'r_squared': simple.rsquared}).to_csv(OUTPUT_DIR / 'daily_bp_results.csv')

# 命令行加 --daily-only 时，只运行上面的单日练习。
if __name__ == '__main__' and '--daily-only' in sys.argv:
    raise SystemExit(0)

# %% L05-BP-MV：加入总市值的单日回归（先运行 L05-BP）
# 改编自统计分析课程 chp3_part3_answer.py 的 TODO 2。
mv = pd.read_csv(DATA_DIR / "daily_total_mv_sh.csv", index_col=0).loc[date]
log_mv = np.log(mv.where(mv > 0)).rename("log_mv")
cross = simple_cross.join(log_mv).replace([np.inf, -np.inf], np.nan).dropna()

# TODO：在 cross 上分别拟合 BP + log_mv 和 BP 模型，命名为 model、simple_same。
raise NotImplementedError("请完成 L05-BP-MV")

# 在相同股票样本上比较一元和多元模型。
daily_comparison = pd.DataFrame({
    "BP only": [simple_same.nobs, simple_same.params["Intercept"],
                simple_same.params["bp"], np.nan,
                simple_same.rsquared, simple_same.rsquared_adj],
    "BP + log MV": [model.nobs, model.params["Intercept"],
                    model.params["bp"], model.params["log_mv"],
                    model.rsquared, model.rsquared_adj]
}, index=["n", "intercept", "bp", "log_mv", "r_squared", "adjusted_r_squared"])
print('加入市值前后的样本数：', len(simple_cross), len(cross))
print(daily_comparison)
print(model.summary())
cross.to_csv(OUTPUT_DIR / 'daily_bp_mv_sample.csv')
daily_comparison.to_csv(OUTPUT_DIR / 'daily_bp_mv_comparison.csv')

# 命令行加 --daily-multi-only 时，完成两项单日回归后停止。
if __name__ == '__main__' and '--daily-multi-only' in sys.argv:
    raise SystemExit(0)

# %% 0. 导入程序包
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_loader import load_panel, DATA_DIR, OUTPUT_DIR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# %% L05-00：单月评分选股
# 直接读取日线宽表，保留前300只股票的月末数据。
pb_daily = pd.read_csv(DATA_DIR / 'pb_sh.csv', index_col=0, parse_dates=True)
codes = pb_daily.columns[:300]
pb_daily = pb_daily.loc['2019-12':'2025-12', codes]
dates = pb_daily.groupby(pb_daily.index.to_period('M')).tail(1).index
pb_month = pb_daily.reindex(dates)
mv_month = pd.read_csv(DATA_DIR / 'total_mv_sh.csv',
                      index_col=0, parse_dates=True).reindex(index=dates, columns=codes)
close_daily = pd.read_csv(DATA_DIR / 'close_sh.csv',
                         index_col=0, parse_dates=True)[codes]
adj_daily = pd.read_csv(DATA_DIR / 'adj_factor_sh.csv',
                       index_col=0, parse_dates=True)[codes]
st_month = pd.read_csv(DATA_DIR / 'is_st_sh.csv',
                      index_col=0, parse_dates=True).reindex(index=dates, columns=codes)
vol_month = pd.read_csv(DATA_DIR / 'vol_sh.csv',
                       index_col=0, parse_dates=True).reindex(index=dates, columns=codes)
# 月末无报价时，按最近一个交易日的复权收盘价估值。
price = (close_daily * adj_daily).ffill().reindex(dates)
next_ret = price.shift(-1) / price - 1
score_panel = pd.DataFrame({
    'pb': pb_month.stack(future_stack=True),
    'total_mv': mv_month.stack(future_stack=True),
    'price': price.stack(future_stack=True),
    'is_st': st_month.stack(future_stack=True),
    'volume': vol_month.stack(future_stack=True),
    'next_ret': next_ret.stack(future_stack=True)
}).rename_axis(['date', 'code']).reset_index()
pool = score_panel.loc[(score_panel['pb'] > 0)
                       & (score_panel['total_mv'] > 0)
                       & (score_panel['price'] > 0)
                       & (score_panel['is_st'] == 0)
                       & (score_panel['volume'] > 0)
                       & (score_panel['date'] < dates[-1])].copy()
month = pool.loc[pool['date'] == '2021-12-31'].copy()

# TODO：填写幻灯片 L05-00 的代码。
raise NotImplementedError("请完成 L05-00")

selected.to_csv(OUTPUT_DIR / 'score_selected.csv', index=False)

# %% L05-00A：逐月选股，计算下一月收益
# TODO：填写幻灯片 L05-00A 的代码。
raise NotImplementedError("请完成 L05-00A")

holdings.to_csv(OUTPUT_DIR / 'score_holdings.csv', index=False)
strategy_returns.to_csv(OUTPUT_DIR / 'score_monthly_returns.csv')
changes.to_csv(OUTPUT_DIR / 'score_position_changes.csv')

# %% L05-00B：计算净值和回撤
# TODO：填写幻灯片 L05-00B 的代码。
raise NotImplementedError("请完成 L05-00B")

nav.to_csv(OUTPUT_DIR / 'score_nav.csv')
performance.to_csv(OUTPUT_DIR / 'score_performance.csv')
print(performance)

# %% 绘制净值与持仓调整图
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
for name in nav:
    axes[0].plot(nav.index, nav[name], label=name, linewidth=1.6)
axes[0].set(ylabel='累计净值（初始值为1）')
axes[0].grid(alpha=.25)
axes[1].bar(changes.index - pd.Timedelta(days=5), changes['买入数量'],
            width=10, label='买入数量', color='#225896')
axes[1].bar(changes.index + pd.Timedelta(days=5), changes['卖出数量'],
            width=10, label='卖出数量', color='#aa2d2d')
axes[1].set(xlabel='日期', ylabel='股票数量')
axes[1].legend(loc='upper left', ncol=2)
axes[0].legend(loc='upper left')
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'score_nav.png', dpi=180, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / 'score_nav.pdf', bbox_inches='tight')
plt.close(fig)

# %% L05-01：构造当月 BP 与市值暴露
panel = load_panel()
# TODO：删除下一行，将幻灯片 L05-01 的代码原样粘贴到这里。
raise NotImplementedError("请完成 L05-01")

print('回归样本数：', len(sample), '月份数：', sample['date'].nunique())

# %% L05-02：只用 BP 做一次截面回归
# 已准备好：2021 年 12 月末特征，配对 2022 年 1 月收益。
month = sample.loc[sample['return_end'].dt.to_period('M') == '2022-01'].copy()

# TODO：删除下一行，将幻灯片 L05-02 的代码原样粘贴到这里。
raise NotImplementedError("请完成 L05-02")

# %% L05-03：加入市值并比较两个模型
# TODO：删除下一行，将幻灯片 L05-03 的代码原样粘贴到这里。
raise NotImplementedError("请完成 L05-03")

comparison.to_csv(OUTPUT_DIR / 'single_month_comparison.csv')
# 绘图（已写好，学生无需修改）
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
points = axes[0].scatter(month['bp_z'], month['next_ret'], c=month['size_z'], s=8, alpha=.6)
fig.colorbar(points, ax=axes[0], label='Size exposure')
axes[0].set(xlabel='BP exposure', ylabel='Next-month return', title='January 2022')
axes[1].scatter(two.fittedvalues, month['next_ret'], s=8, alpha=.6)
lo, hi = month['next_ret'].min(), month['next_ret'].max()
axes[1].plot([lo, hi], [lo, hi], color='gray')
axes[1].set(xlabel='Fitted return', ylabel='Actual return')
fig.savefig(OUTPUT_DIR / 'single_month_regression.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# %% L05-04：逐月估计系数和残差
# TODO：删除下一行，将幻灯片 L05-04 的代码原样粘贴到这里。
raise NotImplementedError("请完成 L05-04")

factor.to_csv(OUTPUT_DIR / 'factor_returns.csv')
residual.to_csv(OUTPUT_DIR / 'residuals.csv')
sample.groupby('date').size().rename('sample_count').to_csv(OUTPUT_DIR / 'sample_counts.csv')
# 绘图（已写好，学生无需修改）
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, name in zip(axes, ['const', 'bp_z', 'size_z']):
    ax.plot(factor['return_end'], factor[name])
    ax.axhline(0, color='gray', linewidth=.6)
    ax.set_ylabel(name)
axes[-1].set_xlabel('Return realization date')
fig.savefig(OUTPUT_DIR / 'monthly_coefficients.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('已完成逐月回归，月份数：', len(factor))

# %% L05-05：拓展：估计因子协方差
# 只用截至 2022 年末已经实现的收益训练。
cutoff = pd.Timestamp('2022-12-30')
cols = ['const', 'bp_z', 'size_z']
# features 在剔除下一月标签之前构造，风险暴露不使用未来交易信息。
asof = features.loc[features['date'] == cutoff].set_index('code')

# TODO：删除下一行，将幻灯片 L05-05 的代码原样粘贴到这里。
raise NotImplementedError("请完成 L05-05")

# 同一组股票、同一段历史收益，用于比较样本协方差。
returns = sample.pivot(index='date', columns='code', values='next_ret')
returns = returns.reindex(index=train_f.index, columns=codes)
Sigma_sample = returns.cov().to_numpy()
centered = returns.to_numpy() - returns.to_numpy().mean(axis=0)
print('股票数 N：', len(codes), '训练期数 T：', len(train_f),
      '样本协方差秩：', np.linalg.matrix_rank(centered))

# %% L05-06：拓展：等权组合的风险分解
# TODO：删除下一行，将幻灯片 L05-06 的代码原样粘贴到这里。
raise NotImplementedError("请完成 L05-06")

pd.Series({'common_var': common_var, 'specific_var': specific_var,
           'total_var': total_var,
           'annualized_volatility': np.sqrt(12 * total_var)}).to_csv(OUTPUT_DIR / 'risk_summary.csv')
# 绘图（已写好，学生无需修改）
covs = [Sigma_sample[:15, :15] * 12, Sigma_factor[:15, :15] * 12]
bound = max(np.abs(cov).max() for cov in covs)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, cov, title in zip(axes, covs, ['Sample covariance', 'Factor covariance']):
    im = ax.imshow(cov, cmap='RdBu_r', vmin=-bound, vmax=bound)
    ax.set_title(title + ' (annualized)')
fig.colorbar(im, ax=axes)
fig.savefig(OUTPUT_DIR / 'covariance_comparison.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# 单只股票与等权组合的共同方差、特质方差。
stock_exposure = B.to_numpy()[0]
weights = np.full(len(codes), 1 / len(codes))
portfolio_exposure = B.to_numpy().T @ weights
values = np.array([
    [stock_exposure @ Omega @ stock_exposure, Delta[0, 0]],
    [portfolio_exposure @ Omega @ portfolio_exposure, weights @ Delta @ weights]]) * 12
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, vals, title in zip(axes, [values, values / values.sum(axis=1, keepdims=True)],
                          ['Annualized variance', 'Variance shares']):
    ax.bar(['One stock', 'Equal-weight portfolio'], vals[:, 0], label='Common')
    ax.bar(['One stock', 'Equal-weight portfolio'], vals[:, 1], bottom=vals[:, 0], label='Specific')
    ax.set_title(title)
    ax.legend()
fig.savefig(OUTPUT_DIR / 'risk_decomposition.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('结果保存到：', OUTPUT_DIR)

