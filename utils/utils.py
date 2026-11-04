# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:40:30 2024

@author: lich5
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from pathlib import Path


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent/'data'

def calc_nav(Pctchg, w, **kwargs):
    '''
        计算净值、pnl、换手率
    '''
    
    if 'comsn' in kwargs.keys():
        comsn = kwargs['comsn'] #加入交易费用
    else:
        comsn = 0
        
    Times = pd.to_datetime(Pctchg.index)
    # 时间期数
    T = len(Times)
    # 从价格序列得到的收益矩阵
    Pctchg = Pctchg.values
    w = w.fillna(0)
    w = w.values
    # 定义pnl
    pnl = np.zeros((T, ))
    # 定义净值
    nav = np.ones((T, ))
    # 定义换手率
    turnover = np.zeros((T, ))
    # 循环计算每期净值以及换手
    
    for i in range(1, T):
        turnover[i] = np.sum(np.abs(w[i] - ((1 + Pctchg[i]) * w[i-1]))) / 2
        pnl[i] = np.dot(w[i-1], Pctchg[i]) - (turnover[i] * comsn)
        nav[i] = (1 + pnl[i]) * nav[i-1]
        
    # 转换成pandas格式
    nav = pd.Series(nav, index = Times)
    pnl = pd.Series(pnl, index = Times)
    turnover = pd.Series(turnover, index = Times)
    
    return {'nav':nav, 'pnl':pnl, 'turnover':turnover}


def plot_equity(nav, bench_stats=None):

    def format_two_dec(x, pos):
        return '%.2f' % x
    
    equity = nav
    
    plt.figure()
    ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_two_dec)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.xaxis.set_tick_params(reset=True)
    ax.yaxis.grid(linestyle=':')
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.grid(linestyle=':')
    
    equity.plot(lw=2, color='blue', alpha=0.6, x_compat=False,
                label='Strategy', ax=ax)

    benchmark = bench_stats.pct_change()
    benchmark.iloc[0] = 0
    benchmark_nav = (1 + benchmark).cumprod()
    benchmark_nav = benchmark_nav.reindex(pd.date_range(benchmark_nav.index[0], benchmark_nav.index[-1], freq='D'))
    benchmark_nav = benchmark_nav.fillna(method = 'ffill')
    benchmark_nav = benchmark_nav.dropna()
    
    benchmark_nav.plot(lw=2, color='gray', alpha=0.6, x_compat=False,
                   label='Benchmark', ax=ax)
    ax.axhline(1.0, linestyle='--', color='black', lw=1)
    ax.set_ylabel('Cumulative returns')
    ax.legend(loc='best')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
    return ax
    

def stock_selection(finalfac: pd.DataFrame, n: int, thres: float) -> pd.DataFrame:
    mat = pd.DataFrame(False, index=finalfac.index, columns=finalfac.columns)
    for i, time in enumerate(finalfac.index):

        maxi = finalfac.loc[time].max()
        if i == 0:
            # 第一期直接选前n个最大因子值资产
            selected = finalfac.loc[time].nlargest(n).index
            mat.loc[time, selected] = True
        else:
            prev_time = finalfac.index[i-1]
            prev_selected = mat.loc[prev_time]
            # 保留T-1期被选中且因子值 > maxi*0.95的资产
            retained = prev_selected[(prev_selected) & (finalfac.loc[time] > maxi * thres)].index
            mat.loc[time, retained] = True
            # 计算还需选入的资产数量
            remain_num = n - mat.loc[time].sum()
            if remain_num > 0:
                # 剩余未选资产，按因子值排序选入
                candidates = finalfac.loc[time][~mat.loc[time]].sort_values(ascending=False)
                selected_candidates = candidates.head(remain_num).index
                mat.loc[time, selected_candidates] = True
        
    return mat
    


def _industry_on_date(industry, date):
    """选取当天有效的行业归属；同一股票保留最近生效的一条。"""
    date = pd.Timestamp(date)
    active = industry.loc[(industry.in_date <= date) &
                          (industry.out_date.isna() | (date < industry.out_date))]
    return active.sort_values("in_date").loc[
        lambda x: ~x.index.duplicated(keep="last")]


def _get_industries(ind='l1_code', date=None):
    """
    载入行业数据；指定 date 时，只返回当天有效的行业归属。
    ind 可取 l1_code、l2_code、l3_code 或对应的 name。
    """
    outp = pd.read_csv(_data_dir() / 'stk_company_info.csv', dtype=str)
    outp = outp.dropna(subset=['ts_code', ind]).set_index('ts_code')
    for col in ['in_date', 'out_date']:
        outp[col] = pd.to_datetime(outp[col], format='%Y%m%d', errors='coerce')
    outp = outp[[ind, 'in_date', 'out_date']]
    return outp if date is None else _industry_on_date(outp, date)


def filter_matrix_spl(times, codes, listdate, days = 52):

    T = len(times)
    S = len(codes)
#    idx = np.zeros(S)
    outs = np.ones((T, S))
    listdate = listdate[~listdate.duplicated(keep='first')]
    codeidx = pd.Series(np.nan, index=codes)
    for i in list(listdate.index):
        if (i in codes):
            codeidx[i] = listdate[i]
            
    # get the dates one year (or otherwise defined in 'days') after stocks' listing date
    valid_dates = pd.to_datetime([str(int(i)) if i == i else np.nan for i in codeidx]) + pd.Timedelta('%s days'%(days))
    
    for i in range(S):
        
        Match1 = (np.abs(times - valid_dates[i])).argmin()
        if not Match1: # Match1 == 0: valid date before the startdate
            continue
        else:
            outs[:Match1, i] = np.nan
            
    return outs


    
def get_listdate(ind = 'l1_code'):
    '''
    载入行业数据
    ind = 'l1_code', 'l2_code', 'l3_code'
    ind = 'l1_name', 'l2_name', 'l3_name'
    '''
    outp = pd.read_csv(_data_dir() /f'stk_company_info.csv')
    outp.index = outp['ts_code']
    outp = outp['in_date']
    return outp


class industry_neutral:
    """按当日行业分组，去均值或标准化；不控制规模。"""

    def __init__(self, ind: str = 'l1_code'):
        self.industry = _get_industries(ind)
        self.ind = ind

    def __call__(self, factor, standardize=True):
        """输入为日期 × 股票的 DataFrame，或由这些表组成的 dict。"""
        if isinstance(factor, dict):
            return {name: self(values, standardize=standardize)
                    for name, values in factor.items()}

        facs = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
        for day, values in factor.iterrows():
            industry = _industry_on_date(self.industry, day)[self.ind]
            groups = values.groupby(industry.reindex(values.index))
            centered = values - groups.transform("mean")
            if standardize:
                std = groups.transform(lambda x: x.std(ddof=0))
                centered = centered / std.replace(0, 1)
            facs.loc[day] = centered
        return facs
                
                
