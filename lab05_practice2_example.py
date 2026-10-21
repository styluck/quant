"""实践二参考示例：沿用实践一的 BP 和市值，演示多因子建模与因子收益解释。

实践二作业要求学生自选因子构建模型；本文件提供可参照修改的实例。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# %% 0. 导入程序包
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_loader import DATA_DIR, OUTPUT_DIR
OUTPUT_DIR = OUTPUT_DIR / 'practice2_example'
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

month['score'] = (month['pb'].rank(ascending=False)
                  + month['total_mv'].rank(ascending=False))
selected = month.nlargest(20, 'score')
print(selected[['code', 'pb', 'total_mv', 'score']])

selected.to_csv(OUTPUT_DIR / 'score_selected.csv', index=False)

# %% L05-00A：逐月选股，计算下一月收益
return_rows, holding_rows, change_rows = [], [], []
previous = set()
for date, g in pool.groupby('date'):
    g = g.copy()
    g['score'] = (g['pb'].rank(ascending=False)
                  + g['total_mv'].rank(ascending=False))
    chosen = g.nlargest(20, 'score').copy()
    small = g.nsmallest(20, 'total_mv')
    end = dates[dates.get_loc(date) + 1]
    return_rows.append({'date': end,
        '低PB＋小市值': chosen['next_ret'].mean(skipna=False),
        '仅小市值': small['next_ret'].mean(skipna=False),
        '股票池等权': g['next_ret'].mean(skipna=False)})
    chosen['weight'] = 1 / len(chosen)
    chosen['return_end'] = end
    holding_rows.append(chosen)
    current = set(chosen['code'])
    change_rows.append({'date': date, '买入数量': len(current - previous),
                        '卖出数量': len(previous - current)})
    previous = current
strategy_returns = pd.DataFrame(return_rows).set_index('date')
holdings = pd.concat(holding_rows, ignore_index=True)
changes = pd.DataFrame(change_rows).set_index('date')

holdings.to_csv(OUTPUT_DIR / 'score_holdings.csv', index=False)
strategy_returns.to_csv(OUTPUT_DIR / 'score_monthly_returns.csv')
changes.to_csv(OUTPUT_DIR / 'score_position_changes.csv')

# %% L05-00B：计算净值和回撤
nav = (1 + strategy_returns).cumprod(skipna=False)
nav.loc[dates[0]] = 1.0
nav = nav.sort_index()
drawdown = nav / nav.cummax() - 1
performance = pd.DataFrame({
    '累计收益': nav.iloc[-1] - 1,
    '年化收益': nav.iloc[-1] ** (12 / len(strategy_returns)) - 1,
    '年化波动率': strategy_returns.std() * np.sqrt(12),
    '最大回撤': drawdown.min()
})

nav.to_csv(OUTPUT_DIR / 'score_nav.csv')
performance.to_csv(OUTPUT_DIR / 'score_performance.csv')
print(performance)


# %% 1. 构造 BP 与规模暴露
sample = pool.copy()
sample['bp'] = 1 / sample['pb']
sample['size'] = np.log(sample['total_mv'])
sample['return_end'] = sample['date'].map(
    pd.Series(dates[1:], index=dates[:-1]))
for column in ['bp', 'size']:
    group = sample.groupby('date')[column]
    sample[column + '_z'] = ((sample[column] - group.transform('mean'))
                              / group.transform('std'))

# %% 2. 单月回归并解释系数
month = sample.loc[sample['date'] == '2021-12-31']
X = sm.add_constant(month[['bp_z', 'size_z']])
one_month = sm.OLS(month['next_ret'], X).fit()
print('2022年1月回归系数（乘100后单位为百分点）：')
print(100 * one_month.params)
print('调整后 R²：', one_month.rsquared_adj)

# %% 3. 逐月估计因子收益和个股残差
factor_rows, stock_rows = [], []
for date, group in sample.groupby('date'):
    X = sm.add_constant(group[['bp_z', 'size_z']])
    fit = sm.OLS(group['next_ret'], X).fit()
    factor_rows.append({'date': date,
        'return_end': group['return_end'].iloc[0],
        **fit.params.to_dict(), 'n': int(fit.nobs),
        'adjusted_r_squared': fit.rsquared_adj})
    stock_rows.append(group.assign(fitted=fit.fittedvalues, resid=fit.resid))
factor = pd.DataFrame(factor_rows).set_index('date')
stock = pd.concat(stock_rows, ignore_index=True)
coefficients = factor[['const', 'bp_z', 'size_z']]
summary = pd.DataFrame({
    'mean_monthly': coefficients.mean(),
    'std_monthly': coefficients.std(),
    'positive_fraction': (coefficients > 0).mean()})
factor.to_csv(OUTPUT_DIR / 'factor_returns.csv', encoding='utf-8-sig')
stock.to_csv(OUTPUT_DIR / 'stock_regression_results.csv',
             index=False, encoding='utf-8-sig')
summary.to_csv(OUTPUT_DIR / 'factor_summary.csv', encoding='utf-8-sig')

# %% 4. 绘制 BP、规模因子收益曲线
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for column, label, ax in zip(['bp_z', 'size_z'], ['BP', '规模'], axes):
    ax.plot(factor['return_end'], 100 * factor[column], linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set_ylabel(label + '系数（百分点）')
    ax.grid(alpha=.2)
axes[-1].set_xlabel('收益实现月份')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'monthly_factor_returns.png', dpi=160)
fig.savefig(OUTPUT_DIR / 'monthly_factor_returns.pdf')
plt.close(fig)

# %% 5. 分解实践一评分组合的月收益
positions = holdings[['date', 'code', 'weight']].merge(
    stock[['date', 'code', 'bp_z', 'size_z', 'next_ret', 'resid']],
    on=['date', 'code'], how='left')
for column in ['bp_z', 'size_z', 'next_ret', 'resid']:
    positions['weighted_' + column] = positions['weight'] * positions[column]
portfolio = positions.groupby('date')[
    ['weighted_bp_z', 'weighted_size_z', 'weighted_next_ret', 'weighted_resid']
].sum().join(factor[['return_end', 'const', 'bp_z', 'size_z']])
portfolio['截距项'] = portfolio['const']
portfolio['BP贡献'] = portfolio['weighted_bp_z'] * portfolio['bp_z']
portfolio['规模贡献'] = portfolio['weighted_size_z'] * portfolio['size_z']
portfolio['残差部分'] = portfolio['weighted_resid']
parts = ['截距项', 'BP贡献', '规模贡献', '残差部分']
portfolio['分解合计'] = portfolio[parts].sum(axis=1)
portfolio['实际收益'] = portfolio['weighted_next_ret']
portfolio.to_csv(OUTPUT_DIR / 'portfolio_attribution.csv', encoding='utf-8-sig')
print('平均月收益分解（百分点）：')
print(100 * portfolio[parts + ['实际收益']].mean())

# %% 6. 保存简要运行报告
report = [
    '# 实践二运行报告', '',
    '模型：下一月收益 = 截距 + BP暴露×BP系数 + 规模暴露×规模系数 + 残差。', '',
    f'完成 {len(factor)} 个月、{len(sample)} 条股票月观测的回归。', '',
    '2021年12月末暴露解释2022年1月收益，控制另一项暴露后：',
    f"- BP暴露增加1，拟合月收益变化 {100 * one_month.params['bp_z']:.4f} 个百分点。",
    f"- 规模暴露增加1，拟合月收益变化 {100 * one_month.params['size_z']:.4f} 个百分点。",
    f"- 两项暴露均为0时，拟合收益为 {100 * one_month.params['const']:.4f}%。", '',
    f"平均月度BP系数：{100 * summary.loc['bp_z', 'mean_monthly']:.4f} 个百分点。",
    f"平均月度规模系数：{100 * summary.loc['size_z', 'mean_monthly']:.4f} 个百分点。",
    f"收益分解最大误差：{(portfolio['分解合计'] - portfolio['实际收益']).abs().max():.3g}。", '',
    'CSV中的收益与系数使用小数，乘100后单位为百分点。']
(OUTPUT_DIR / 'run_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print('\n'.join(report))
print('结果目录：', OUTPUT_DIR)
