"""第六讲课堂示例：因子预处理、IC、五档与多空检验。

按 # %% 在 Spyder 中逐段运行，或在项目根目录运行：
python code/lab06/lab06_practice_example.py
"""
# %% 1. 数据载入：行是交易日，列是股票
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from data_io import load_data, load_benchmark
from utils.FactorTest import factor_test
from utils.data_standardize import preprocess
from utils.utils import industry_neutral, calc_nav, plot_equity
from utils.stockselection import stock_selection
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

daily = {}
for name in ['close', 'adj_factor', 'pb', 'total_mv', 'vol', 'amount', 'is_st']:
    daily[name] = load_data(name, '2019-01-01', '2025-12-31')


# %% 2. 构建 BP、规模和动量，取月末观测
price = daily['close'] * daily['adj_factor']
raw = {
    'BP': 1 / daily['pb'].where(daily['pb'] > 0),
    'MV': np.log(daily['total_mv'].where(daily['total_mv'] > 0)),
    'Momentum': price / price.shift(21) - 1,
}
dates = pd.DatetimeIndex(
    pd.Series(price.index).groupby(price.index.to_period('M')).max())
dates = dates[dates >= '2019-12-01']

# 建立月末股票池。
stock_info = pd.read_csv(HERE.parent / 'data/sse_stock_basic.csv')
stock_info['code'] = stock_info['code'].str[3:] + '.SH'
ipo_date = pd.to_datetime(stock_info.set_index('code')['ipoDate'])
panel = pd.concat({name: daily[name].reindex(dates).stack(future_stack=True)
                   for name in ['is_st', 'amount', 'vol']}, axis=1)
panel = panel.rename_axis(['date', 'code']).reset_index()
panel['listed_days'] = (panel['date'] - panel['code'].map(ipo_date)).dt.days
panel['tradable'] = (panel['vol'] > 0).astype(int)  # 当日有成交

filtered_pool = panel[
    (panel['is_st'] == 0)
    & (panel['listed_days'] >= 120)
    & (panel['tradable'] == 1)
    & (panel['amount'] >= 10_000)  # 单位：千元，至少1000万元
].copy()
valid = (filtered_pool.assign(in_pool=True)
         .pivot(index='date', columns='code', values='in_pool')
         .reindex(index=dates, columns=price.columns).eq(True))
raw = {name: x.reindex(dates).where(valid).replace([np.inf, -np.inf], np.nan)
       for name, x in raw.items()}

# factor_test 将第 t 行因子与第 t+1 行收益配对，收益表不再提前 shift。
monthly_price = price.reindex(dates)
returns = monthly_price.pct_change(fill_method=None)


# %% 3. 因子预处理：MAD 缩尾后作 Z-score
factors = {name: preprocess(x) for name, x in raw.items()}


# %% 4. IC、五档与多空检验
results = {}
summary_rows = []

for name, x in factors.items():
    result = factor_test(x, returns, NumOfGroup=5,
                         DispFlag=False, PlotFlag=True, comsn=0,
                         name=name)
    results[name] = result

    # 第一个位置是对齐基期；2020年1月至2025年12月共72个月。
    for metric, key in [('Pearson IC', 'Normal_IC'), ('Rank IC', 'Rank_IC')]:
        ic = pd.Series(result[key][1:], index=dates[1:]).dropna()
        summary_rows.append({'因子': name, '指标': metric,
            '平均IC': ic.mean(), 'IC标准差': ic.std(ddof=1),
            'ICIR': ic.mean() / ic.std(ddof=1), 'p+': (ic > 0).mean()})

summary = pd.DataFrame(summary_rows)
print(summary.round(4).to_string(index=False))


# %% 5. 比较 BP 行业中性化前后的表现
# 行业工具按各月有效的申万一级行业分类，在行业内去均值。
neutralizer = industry_neutral('l1_code')
bp_neutral = neutralizer(factors['BP'], standardize=False)
bp_before = factors['BP'].where(bp_neutral.notna())

industry_rows = []
for name, x in {'BP_before': bp_before, 'BP_neutral': bp_neutral}.items():
    result = factor_test(x, returns, NumOfGroup=5,
                         DispFlag=False, PlotFlag=True, comsn=0,
                         name=name)
    results[name] = result
    group_returns = pd.DataFrame(result['GroupRtn'][1:], index=dates[1:])
    long_short = group_returns.iloc[:, -1] - group_returns.iloc[:, 0]
    for metric, key in [('Pearson IC', 'Normal_IC'), ('Rank IC', 'Rank_IC')]:
        ic = pd.Series(result[key][1:]).dropna()
        industry_rows.append({'因子': name, '指标': metric,
            '平均IC': ic.mean(), 'IC标准差': ic.std(ddof=1),
            'ICIR': ic.mean() / ic.std(ddof=1), 'p+': (ic > 0).mean(),
            '多空月均收益': long_short.mean(),
            '多空月收益标准差': long_short.std(ddof=1),
            '多空期末净值': (1 + long_short).prod()})

industry_summary = pd.DataFrame(industry_rows)
print(industry_summary.round(4).to_string(index=False))


# %% 6. 从行业中性化 BP 的高值组选30只股票，等权计算组合净值 
holdings = results['BP_neutral']['Code_G10'][:, :30]
weights = stock_selection(returns.columns, holdings, dates)
outputs = calc_nav(returns.fillna(0), weights)
benchmark = load_benchmark('2019-12-01', '2025-12-31')['close'].reindex(dates)
plot_equity(outputs['nav'], benchmark)
plt.show()
