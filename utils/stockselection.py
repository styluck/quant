# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:34:40 2020

@author: Admin
"""
import numpy as np
import pandas as pd


def fac_selection(df, n, **kwargs):
    '''
    获取每行数值最大的n个元素的column值
    '''
    if 'keep' in kwargs.keys():
        keep = kwargs['keep']
    else:
        keep = 'num'
        
    def nan_preserved_argsort(arr, n):
        return arr
    
    dfv = df.values
    t, d = np.shape(df)
    sorted_indices = np.empty((t,n))*np.nan
    
    for i in range(t):
        arr = dfv[i]
        nan_mask = np.isnan(arr)
        if np.sum(nan_mask) == d:
            continue
        
        non_nan_indices = np.where(~nan_mask)[0]
        sorted_arr = np.argsort(arr[~nan_mask])[::-1]
        
        m = np.minimum(len(sorted_arr),n)
        non_nan_indices = non_nan_indices[sorted_arr]
        sorted_indices[i, :m] = non_nan_indices[:m]
                
        
    result = []
    
    if keep == 'code':
        for i in range(t):
            if np.sum(np.isnan(sorted_indices[i])) == n:
                result.append([np.nan for j in sorted_indices[i]])
            else:
                result.append([df.columns[j] for j in sorted_indices[i]])
        result_df = pd.DataFrame(result)
        
    elif keep == 'num':
        for i in range(t):
            if np.sum(np.isnan(sorted_indices[i])) == n:
                result.append([np.nan for j in sorted_indices[i]])
            else:
                result.append([j for j in sorted_indices[i]])
        result_df = np.array(result)
    
    return result_df
    

def stock_selection(code, codeindex, times, **kwargs):
    '''
        资产权重矩阵, 截面上对资产进行等权以及市值加权分配
        params:
                code - 资产池的所有股票代码，e.g. 全市场代码，必须保证排序一致
                codeindex - 需要交易的股票代码位于code的位置矩阵，需要保证code排序一致
                times - 交易时间，需要保证和codeindex时间一致
        optional:
                cap - 股票总市值进行市值加权, 如不需要市值加权则不填
    '''
    
    # get the shape for stocks elected, e.g. codeindex = ft['Code_G10']
    T, S = np.shape(codeindex)
    # stock-pool length, code = np.array(stockList)
    stocknum = len(code)
    # define an empty DataFrame with index as times, columns as codes
    mat = pd.DataFrame(np.zeros((T, stocknum)), index = times, columns = code)
    k = 0
    for row in mat.iterrows():
        # iterate codeindex line-by-line
        indices = codeindex[k,:]
        # drop nans
        notnan = np.where(~np.isnan(indices))
        # convert it to int-type
        indices = indices[notnan].astype(int)
        # get trade stock codes
        cols = code[indices]
        if len(cols) == 0:
            mat.loc[row[0], cols] = 0
        else:
            # assign equal weights to trade stocks, and place them to mat
            mat.loc[row[0], cols] = 1 / len(cols)
        k += 1
        
    # this is for cap-weighted positions
    if len(kwargs) > 0:
        # has to set NAN values for column wise calculations, i.e. weighted.sum(axis = 1)
        mat = mat.replace(0, np.nan)
        cap = kwargs['cap'].reindex(index = times, columns = code)
        weighted = mat * cap
        capweight = weighted.T / (weighted.sum(axis = 1))
        mat = capweight.T
        mat = mat.fillna(0)
        
    # equal-weighted positions
    else:
        mat = mat.fillna(0)
        
    return mat
    
    

def arg_rank(code, codeindex, basis, num = 25, method = 'top'):
    '''
        排序(ascending order)
        params:
                code - 资产池的所有股票代码，e.g. 全市场代码，必须保证排序一致
                codeindex - 需要交易的股票代码位于code的位置矩阵，需要保证code排序一致
                basis - comparing basis，排序参照的因子
                num - 头或尾num只股票
                method - 升序序列选取最小'top'或最大'bottom'
    '''
    # get the shape for stocks elected, e.g. codeindex = ft['Code_G10']
    T, S = np.shape(codeindex)
    
    if len(basis) != T or len(code) != S or len(code) != basis.shape[1]:
        raise ValueError('Dimension Mismatch! ')
    
    # stock-pool length, code = np.array(stockList)
    stocknum = len(code)
    # define an empty DataFrame with index as times, columns as codes
    mat = np.ones((T, stocknum)) * np.nan#, index = basis.index, columns = code)
    fac = basis.values
#    k = 0
    for i in range(T):
        # iterate codeindex line-by-line
        indices = codeindex[i,:]
        # drop nans
        notnan = np.where(~np.isnan(indices))
        # convert it to int-type
        indices = indices[notnan].astype(int)
        # get trade stock codes
#        cols = code[indices]
        if len(indices) == 0:
            mat[i] = np.nan
            
        else:
            f = fac[i, indices]
#            Ind = np.where(~np.isnan(f))
#            f = f[Ind]
            if method == 'top':
                
                ranked_ind = indices[np.argsort(f)][:num]
                length = len(ranked_ind)
                
                if length < num:
                    mat[i, :length] = ranked_ind
                
                elif length == num:
                    mat[i, :num] = ranked_ind
                
                else:
                    raise ValueError('Mismatch of ranked_ind index, should NOT have reached here')
                
            elif method == 'bottom':
                
                ranked_ind = indices[np.argsort(f)][-num:]
                length = len(ranked_ind)
                
                if length < num:
                    mat[i, -length:] = ranked_ind
                    
                elif length == num:
                    mat[i, -num:] = ranked_ind            
                            
                else:
                    raise ValueError('Mismatch of ranked_ind index, should NOT have reached here')
                    
    return mat


def weight_constraint(weight, constraint_ratio = 0.1):
    
    wts = weight.copy()
    times = wts.index
    codes = wts.columns
    
    wt = wts.values
    T, S = wt.shape
    
    for i in range(T):
        length = len(wt[i][wt[i] != 0])
        
        if length <= 1/constraint_ratio:
            wt[i][wt[i] != 0] = constraint_ratio
    
    wt = pd.DataFrame(wt, index = times, columns = codes)
    
    return wt


def stock_union(basis, IndMat, num = 25):
    '''
        回测每一期对应的股票代码，
        先按照交集排序，3个策略都有则必选, 2个都有则其次，依次选取股票；
        再通过定义basis对应的因子进行排序
        
        参数：
        ------
            basis - 定义的因子
            
            IndMat - 通过factor_analysis的ft_sn['Code_G10'][:,:NumOfStock]得到
                    每期每个因子/策略股票代码数组，再把她们合成三维矩阵，可通过运行
                    Multifactor_Matrix(code1 = f1, code2 = f2, code3 = f3, code4 = f4)
                    
            num - 按顺序选取的多少只票
    '''
    
    if type(basis) == pd.core.frame.DataFrame:
        basis = basis.values
    
    T, S = basis.shape
    
    if T != IndMat.shape[1]:
        raise ValueError('The time length between basis and IndMat do not Match! ')
        
    # 定义全NAN矩阵
    mat = np.ones((T, S)) * np.nan
    
    for t in range(T):
        # 取出每期每个策略对应股票代码的位置
        index = IndMat[:, t, :]
        # 矩阵摊平
        flatindex = index.flatten()#
        # 如果该期全为NAN或者部分为NAN的情况则不做处理，留为NAAN
        if len(flatindex[~np.isnan(flatindex)]) < len(flatindex):
            continue
        else:
            flatindex = flatindex.astype('int')
            
        # 找出每个策略每只股票出现次数
        indexcounts = np.bincount(flatindex)
        # facnum是策略/因子数量
        facnum = index.shape[0]
        # 找出每只股票在多策略/因子中出现次数，并确认其所在位置
        tmp_ind = [np.where(indexcounts == facnum - i)[0] for i in range(facnum)]
        # 计数用
        c = 0
        for i in range(facnum):
            # 找出股票在basis策略/因子中所处位置
            bval = basis[t, tmp_ind[i]]
            # 按照basis策略/因子的值进行排序
            rankind = tmp_ind[i][np.argsort(bval)] # [::np.argsort(bval)]
            # 将排序后的股票顺序插入定义的全NAN矩阵中
            mat[t, c:c+len(rankind)] = rankind
            # 计数随循环增加
            c+=len(rankind)
    # 确认头部股票
    code_index = mat[:, :num]
    
    return code_index


#        union = reduce(np.union1d, tuple((index)))
#        intsct = reduce(np.intersect1d, tuple((index)))


def freq_elab(data, exampledata):
    '''
        Frequency Elaborator, 把月度dataframe重新转换成日度, 
        
        参数：
        ------
        data - 需要转换的dataframe，月频或周频
        exampledata - 参照转换的dataframe， 日频
        
    '''
    
    if type(data) != pd.core.frame.DataFrame:
        raise TypeError("input 'data' must be pd.DataFrame")
        
    if type(exampledata) != pd.core.frame.DataFrame:
        raise TypeError("input 'exampledata' must be pd.DataFrame")
        
    cols = data.columns
    timeseries = data.index
    startdate = timeseries[0]
    enddate = timeseries[-1]
    
    ## Define empty data
    dailyindex = pd.date_range(startdate, enddate, freq='D')
    strcheddf = pd.DataFrame(np.ones((len(dailyindex), len(cols))) * np.nan, index = dailyindex, columns = cols)
    # 将对应的月度/周度位置插入empty data
    strcheddf.loc[timeseries] = data
    strcheddf = strcheddf.fillna(method = 'ffill')
    
    # 截取data对应的位置
    exmp = exampledata.loc[startdate:enddate]
    # 找出对应交易日
    strcheddf = strcheddf.loc[exmp.index]
    # 找出对应的股票
    exmp = exmp[cols]
    
    return {'data':strcheddf, 'exampledata':exmp}





# def get_equity_future(contract, timeindex):
#     print('get equity future, this is going to take a few minutes...... ')
#     contracts = pd.Series(timeindex).apply(lambda x: get_dominant_future(contract, x))
#     future = pd.DataFrame(contracts.values, index = timeindex, columns = ['contracts'])#pd.DataFrame(timeindex, index = contracts, columns = ['date'])
#     close = np.ones(timeindex.shape) * np.nan

#     for i in range(len(future)):
#         date = future.iloc[i].name#['date']
#         code = future.iloc[i]['contracts']
#         close[i] = get_price(code, start_date=date, end_date=date, frequency='daily', fields=None, skip_paused=False, fq='none')['close']
    
#     future['close'] = close
    
#     return future




# def hedging(contract, timeindex, pnl, freq, hedging_ratio):
#     '''
#         excessive returns
#     '''
#     def format_two_dec(x, pos):
#         return '%.2f' % x
    
#     def format_perc(x, pos):
#         return '%.0f%%' % x
        
#     future = get_equity_future(contract, timeindex)
#     wk_future = x_io.Chg_period([], future.copy(), freq=freq, freqTy='last' ).fillna(method = 'ffill')
#     wk_future['ret'] = wk_future['close'].pct_change()
    
#     hedged_pnl = pnl - (wk_future['ret'] * hedging_ratio)
#     hedged_nav = (1 + (hedged_pnl)).cumprod()#.plot()
#     drawdowns = -1 * bt.DrawDowns(hedged_nav)[0] * 100
# #    (1 + (bt.pnl - benchmark.pct_change())).cumprod().plot()
    
#     Times = hedged_nav.index
#     TimeLength = (Times[-1] - Times[0]).days / 365.2422
#     cagr = (hedged_nav.iloc[-1] ** (1 / TimeLength)) - 1.0
#     sharpe = bt.Sharpe(nav = hedged_nav, pnl = hedged_pnl)
#     tot_ret = hedged_nav.iloc[-1] - 1.0
    
#     TimeStep = max(1, np.diff(hedged_nav.index.tolist()).mean().days)*.9 
#     AnnualStepPeriod = 365.2422 / TimeStep
#     volatility = hedged_pnl.std() * np.sqrt(AnnualStepPeriod)  
    
#     dd_max = bt.DrawDowns(hedged_nav)[1]
    
#     # Plotting
#     plt.figure(figsize=(16,12), dpi=80)
#     plt.suptitle('Hedged Backtest: hedging ratio - %.0f%%'%(hedging_ratio*100), y=0.94, weight='bold', fontsize=20)
#     gs = gridspec.GridSpec(5, 3, wspace=0.25, hspace=0.5)
    
#     plt.subplot(gs[:3, :])    
#     y_axis_formatter = FuncFormatter(format_two_dec)
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
#     hedged_nav.plot(lw=2, color='blue', alpha=0.6, x_compat=False,
#                     label='Strategy')
#     plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
#     plt.ylabel('NAV')
#     plt.title('Excess Return against %s'%contract, fontweight='bold', fontsize=15)
    
#     plt.subplot(gs[3:4, :])    
#     y_axis_formatter = FuncFormatter(format_perc)
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
#     drawdowns.plot(lw=2, kind = 'area', color='green', alpha=0.3)
#     plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
#     plt.ylabel('drawdowns')
#     plt.title('Excess Return Drawdowns (%)', fontweight='bold', fontsize=15)
    
#     plt.subplot(gs[4, :])   
#     y_axis_formatter = FuncFormatter(format_perc)
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
#     plt.gca().text(0.3, 8, 'Total Return', fontsize=15)
#     plt.gca().text(9, 8, '{:.0%}'.format(tot_ret), fontweight='bold', horizontalalignment='right', fontsize=15)

#     plt.gca().text(0.3, 6.4, 'CAGR', fontsize=15)
#     plt.gca().text(9, 6.4, '{:.2%}'.format(cagr), fontweight='bold', horizontalalignment='right', fontsize=15)

#     plt.gca().text(0.3, 4.8, 'Sharpe Ratio', fontsize=15)
#     plt.gca().text(9, 4.8, '{:.2f}'.format(sharpe), fontweight='bold', horizontalalignment='right', fontsize=15)

#     plt.gca().text(0.3, 3.2, 'Annual Volatility', fontsize=15)
#     plt.gca().text(9, 3.2, '{:.2%}'.format(volatility), fontweight='bold', horizontalalignment='right', fontsize=15)

#     plt.gca().text(0.3, 1.6, 'Max Drawdown', fontsize=15)
#     plt.gca().text(9, 1.6, '{:.2%}'.format(dd_max), color='red', fontweight='bold', horizontalalignment='right', fontsize=15)
    
#     plt.gca().set_title('Hedged Performance Ratios', fontweight='bold', fontsize=15)
    
#     plt.gca().grid(False)
#     plt.gca().spines['top'].set_linewidth(2.0)
#     plt.gca().spines['bottom'].set_linewidth(2.0)
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
#     plt.gca().get_yaxis().set_visible(False)
#     plt.gca().get_xaxis().set_visible(False)
#     plt.gca().set_ylabel('')
#     plt.gca().set_xlabel('')
#     plt.gca().axis([0, 10, 0, 10])
    
    
#     return hedged_nav




    