"""实践二 example：用 BP 和市值展示完整的研究流程。

先按 PB、市值排名构建组合并回测，再用 BP、对数市值的横截面
回归解释同一组合的收益。选股评分与收益解释是两个不同环节。

在项目根目录运行：python code/lab05/lab05_practice2_example.py
也可在 IDE 中按 # %% 顺序分段运行。

输入：code/data/ 中的日度宽表；输出：code/output/lab05/practice2_example/ 下的图片。
收益使用小数，乘 100 后单位为百分点。
"""
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 让脚本能 import 到 code/data_io 下的载入函数
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# %% 0. 准备运行环境与结果目录
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_io import load_data

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'output' / 'lab05' / 'practice2_example'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# %% 1. 数据载入：读入全部日度宽表
# 宽表的行是交易日、列是股票。这里只读入，不做任何清洗。
START, END = '2019-12-01', '2025-12-31'
DAILY_FIELDS = ['close', 'adj_factor', 'pb', 'total_mv', 'vol',
                'amount', 'is_st', 'turnover_rate']

daily = {}
for name in DAILY_FIELDS:
    df = load_data(name, START, END)
    df.index.name = 'date'
    df.columns.name = 'code'
    daily[name] = df

# %% 2. 生成月末面板
# 月末日期：每月最后一个交易日
calendar = daily['close'].index
dates = pd.DatetimeIndex(
    pd.Series(calendar).groupby(calendar.to_period('M')).max())

# 在日度层面判断可交易：正收盘价、正成交量、正成交额、非 ST、有换手
tradable = ((daily['close'] > 0) & (daily['vol'] > 0)
            & (daily['amount'] > 0) & (daily['is_st'] == 0)
            & (daily['turnover_rate'] > 0))
# 只看月末那天的可交易状态：True 才把该股该月纳入样本
tradable_end = tradable.reindex(dates)

# 取每个字段在月末日那天的值，stack 成长表；不可交易的格子被置 NaN 后由 stack 丢掉
monthly = {}
for name in DAILY_FIELDS:
    monthly[name] = (daily[name].reindex(dates).where(tradable_end)
                     .stack(future_stack=True).rename_axis(['date', 'code']))
panel = pd.DataFrame(monthly).reset_index().sort_values(['code', 'date'])

panel['adj_close'] = panel['close'] * panel['adj_factor']
panel['bp'] = 1 / panel['pb']                # BP = 1/PB
panel['size'] = np.log(panel['total_mv'])    # 对数总市值（万元，单位对 log/排名无影响）

# 下一月收益：同一只股票下个月末复权价 / 本月末复权价 - 1
g = panel.groupby('code')
panel['return_end'] = g['date'].shift(-1)
panel['next_ret'] = g['adj_close'].shift(-1) / panel['adj_close'] - 1
panel['next_ret'] = panel['next_ret'].replace([np.inf, -np.inf], np.nan)

# 只留 pb>0（bp 才有意义）的样本；可交易性已在上面处理
panel = panel[panel['pb'] > 0]
panel = panel.sort_values(['date', 'code']).reset_index(drop=True)

# %% 3. 逐月调仓：按 PB、市值排名选股，记录持仓与下一月收益
# PB 越低、市值越小，降序排名数值越大；两项名次相加取前 20 只，等权持有。
holding_rows, return_rows = [], []
for date, g in panel.groupby('date'):
    g = g.dropna(subset=['next_ret'])    # 丢掉下一月收益缺失的（最后一个月）
    if g.empty:
        continue
    g = g.copy()
    g['score'] = g['pb'].rank(ascending=False) + g['total_mv'].rank(ascending=False)
    chosen = g.nlargest(20, 'score')
    chosen['weight'] = 1 / len(chosen)
    holding_rows.append(chosen)
    return_rows.append({
        'date': date,                                 # 形成持仓的月末
        'return_end': chosen['return_end'].iloc[0],   # 收益实现的下月末
        '组合收益': chosen['next_ret'].mean(),
    })

holdings = pd.concat(holding_rows, ignore_index=True)
strategy_returns = pd.DataFrame(return_rows)   # 列：date、return_end、组合收益

# %% 4. 回测：净值、收益与回撤
# 收益按「收益实现月末」对齐，起始净值为 1
nav = (1 + strategy_returns.set_index('return_end')['组合收益']).cumprod().sort_index()
nav = pd.concat([pd.Series([1.0], index=[strategy_returns['date'].min()]), nav])
drawdown = nav / nav.cummax() - 1

performance = pd.DataFrame({
    '累计收益': nav.iloc[-1] - 1,
    '年化收益': nav.iloc[-1] ** (12 / len(strategy_returns)) - 1,
    '年化波动率': strategy_returns['组合收益'].std() * np.sqrt(12),
    '最大回撤': drawdown.min(),
}, index=['组合'])
print(performance)

fig, ax = plt.subplots(figsize=(10, 4.5))
nav.plot(ax=ax, linewidth=1.3)
ax.set(xlabel='收益实现日期', ylabel='净值（初始值为1）')
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'score_nav.png', dpi=160)

# %% 5. 构造 BP、市值暴露并按月标准化
# 以下用模型解释已发生的收益。BP 高=净资产相对价格高；size 高=市值大。
sample = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=['bp', 'size', 'next_ret'])
for col in ['bp', 'size']:
    grp = sample.groupby('date')[col]
    sample[col + '_z'] = (sample[col] - grp.transform('mean')) / grp.transform('std')

# %% 6. 单月横截面回归：解释系数
# 观测是同一个月的不同股票，不是单只股票的时间序列
# 2021 年 12 月末暴露 配对 2022 年 1 月收益，含截距
month = sample[sample['date'] == '2021-12-31']
X = sm.add_constant(month[['bp_z', 'size_z']])
res = sm.OLS(month['next_ret'], X).fit()
print('2022 年 1 月回归系数（乘 100 后单位为百分点）：')
print(100 * res.params)
print('调整后 R²：', res.rsquared_adj)

# %% 7. 逐月估计：保存因子收益与个股残差
# 每个月独立回归：横截面系数=当月事后因子收益，残差=股票未解释收益
factor_rows, stock_rows = [], []
for date, g in sample.groupby('date'):
    X = sm.add_constant(g[['bp_z', 'size_z']])
    fit = sm.OLS(g['next_ret'], X).fit()
    factor_rows.append({'date': date, 'return_end': g['return_end'].iloc[0],
                        **fit.params.to_dict(), 'n': int(fit.nobs),
                        'adjusted_r_squared': fit.rsquared_adj})
    stock_rows.append(g.assign(fitted=fit.fittedvalues, resid=fit.resid))

factor = pd.DataFrame(factor_rows).set_index('date')
stock = pd.concat(stock_rows, ignore_index=True)

# %% 8. 观察因子收益的跨月变化
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for col, label, ax in zip(['bp_z', 'size_z'], ['BP', '规模'], axes):
    ax.plot(factor['return_end'], 100 * factor[col], linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set_ylabel(label + '系数（百分点）')
    ax.grid(alpha=.2)
axes[-1].set_xlabel('收益实现月份')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'monthly_factor_returns.png', dpi=160)

# %% 9. 利用原持仓分解评分组合的月收益
# 按 date、code 对齐原持仓与该月回归结果，再按原权重加总
# 一个因子的贡献 = 持仓加权平均暴露 × 当月因子系数
positions = holdings[['date', 'code', 'weight']].merge(
    stock[['date', 'code', 'bp_z', 'size_z', 'next_ret', 'resid']],
    on=['date', 'code'], how='left')
for col in ['bp_z', 'size_z', 'next_ret', 'resid']:
    positions['weighted_' + col] = positions['weight'] * positions[col]

portfolio = positions.groupby('date')[
    ['weighted_bp_z', 'weighted_size_z', 'weighted_next_ret', 'weighted_resid']
].sum().join(factor[['return_end', 'const', 'bp_z', 'size_z']])
portfolio['截距项'] = portfolio['const']
portfolio['BP贡献'] = portfolio['weighted_bp_z'] * portfolio['bp_z']
portfolio['规模贡献'] = portfolio['weighted_size_z'] * portfolio['size_z']
portfolio['残差部分'] = portfolio['weighted_resid']
portfolio['实际收益'] = portfolio['weighted_next_ret']

parts = ['截距项', 'BP贡献', '规模贡献', '残差部分']
fig, ax = plt.subplots(figsize=(8, 4))
(100 * portfolio[parts].mean()).plot.bar(ax=ax, rot=0, color='#225896')
ax.axhline(0, color='gray', linewidth=.7)
ax.set(ylabel='平均月收益贡献（百分点）', xlabel='')
ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'portfolio_attribution.png', dpi=160)

print('平均月收益分解（百分点）：')
print(100 * portfolio[parts + ['实际收益']].mean())
print('结果目录：', OUTPUT_DIR)
