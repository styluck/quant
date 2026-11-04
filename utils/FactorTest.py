# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:23:46 2019

@author: Chechi-Siu
"""

#  imports ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import tqdm

#from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from scipy.stats.mstats import gmean
# import statsmodels.api as sm
# from statsmodels.tsa import stattools

# from statsmodels.stats.diagnostic import unitroot_adf


############## to_pytime #############
def to_pytime(Times, outTy='f'):
    '''
    将matlab时间转换成python的时间格式
     Inputs:
         Times: m-by-1 time series data(仅支持matlab格式的数值时间序列)
         args: 定义输出类型：
             'a' : numpy array
             'f' : dataframe
     example:
         a = array([737561., 737562., 737563., 737564.])
         to_pytime(a)
         >> DatetimeIndex(['2019-05-16', '2019-05-17', '2019-05-18', '2019-05-19'],
         dtype='datetime64[ns]', freq=None)
    '''
    
    if outTy == 'a' or outTy == 'array':
        if isinstance(Times, list) or isinstance(Times, np.ndarray) or \
         isinstance(Times, np.float64):
            Times = (Times - 719529.0).astype('float64')
            Outp = Times.astype('datetime64[D]')

    elif outTy == 'f' or outTy == 'df' or outTy == 'dataframe' or outTy == 'DataFrame':        
        if isinstance(Times, np.ndarray) or isinstance(Times, np.float64):
            Outp = pd.to_datetime((Times - 719529.0).astype('float64'),unit='D')
            
        elif isinstance(Times, list):
            try:
                Outp = pd.to_datetime((np.array(Times) - 719529.0).astype('float64'),unit='D')
            except:
                Outp = pd.to_datetime(Times - 719529.0,unit='D')
        elif isinstance(Times, float) or isinstance(Times, int):
            Outp = pd.to_datetime(Times - 719529.0,unit='D')
        else:
            raise TypeError('\' Times\' must be of dataframe, list or array type ')
     
    return Outp
    
############## pearson_corr ############## 
def pearson_corr(f,r):
    '''
    返回f与r的Pearson相关性系数，仅支持输入array变量
    example:
        a = np.random.randn(10,1)
        b = np.random.randn(10,1)
        spearman_corr(a,b)
        >> array([0.17038954])
    '''
    
    return pearsonr(f,r)[0]

############## spearman_corr ############## 
def spearman_corr(f,r):
    '''
    返回f与r的spearman相关性系数，仅支持输入array变量
    example:
        a = np.random.randn(10,1)
        b = np.random.randn(10,1)
        spearman_corr(a,b)
        >> array([0.17038954])
    '''
    return spearmanr(f,r)[0]

############## normal_ic ############## 
def normal_ic(Factors, Return):
    '''
     输入：
     f：t-1期的因子值n-by-1
     r：t期的股票收益率n-by-1
     Normal IC，即某时点某因子在全部股票的暴露值与其下期回报的截面相关系数；
     Normal IC = corr(f(t-1) - r(t));
    '''
    
    # check input type
    try:
        if type(Factors) == np.ndarray:
            pass
        elif isinstance(Factors,list):
            Factors = np.array(Factors)
        else:
            raise TypeError('Wrong input type')
    except TypeError:
        print('Factors must be of list or numpy.ndarray type ')
        raise
    
    try:
        if type(Return) == np.ndarray:
            pass
        elif isinstance(Return,list):
            Return = np.array(Return)
        else:
            raise TypeError('Wrong input type')
    except TypeError:
        print('Return must be of list or numpy.ndarray type ')
        raise
        
#    f = np.nan_to_num(Factors)
#    r = np.nan_to_num(Return)
#    c1 = np.where(f==0) # find the null data in returns
#    c2 = np.where(r==0)
#    c = np.union1d(c1[0],c2[0])
#    f = np.delete(f,c) # delete the null data in both returns and factors
#    r = np.delete(r,c)
        
    c1 = ~np.isnan(Factors)
    c2 = ~np.isnan(Return)
    c = c1&c2
    f = Factors[c]
    r = Return[c]
    
    if len(f)>1:
        return pearson_corr(f,r)
    else:
        return 0

############## rank_ic ############## 
def rank_ic(Factors, Return):
    '''
     输入：
     f：t-1期的因子值
     r：t期的股票收益率
     rank_ic，即某时点某因子在全部股票暴露值排名与其下期回报排名的截面相关系数。
     Rank IC = corr(order_f(t-1) -order_r(t));
    '''
    # check input type
    try:
        if type(Factors) == np.ndarray:
            pass
        elif isinstance(Factors,list):
            Factors = np.array(Factors)
        else:
            raise TypeError('Wrong input type')
    except TypeError:
        print('Factors must be of list or numpy.ndarray type ')
        raise
    
    try:
        if type(Return) == np.ndarray:
            pass
        elif isinstance(Return,list):
            Return = np.array(Return)
        else:
            raise TypeError('Wrong input type')
    except TypeError:
        print('Return must be of list or numpy.ndarray type ')
        raise


    c1 = ~np.isnan(Factors)# find the null data in returns
    c2 = ~np.isnan(Return)
    c = c1&c2
    f = Factors[c]# delete the null data in both returns and factors
    r = Return[c]
    # sort index
    if len(f)>1:
        return spearman_corr(f,r)
    else:
        return 0


############## regression_analysis ############## 
def regression_analysis(factor, returns, met = 3, plot = True):
       
    if type(factor) == np.ndarray:
        fac = factor
    elif type(factor) == pd.core.frame.DataFrame:
        fac = factor.values
        
    if type(returns) == pd.core.frame.DataFrame:
        ret = returns.values
    else:
        ret = returns
        
    T,S = np.shape(returns)
#        print('T:%d,S:%d'%(T,S))
    D = np.zeros((T,4))
    
    for i in range(T):

        tmp = reg(fac[i,:], ret[i,:])
        D[i,0] = tmp['beta']
        D[i,1] = tmp['alp']
        D[i,2] = tmp['rsquared']
        D[i,3] = tmp['t_beta']
        
    # define regression outputs
    D = pd.DataFrame(D, columns = ['beta', 'alp', 'rsquared', 't_beta'])

    if type(factor.index[0]) == pd._libs.tslibs.timestamps.Timestamp:
        x1 = factor.index.date.tolist()
    else:
        x1 = factor.index.tolist()
        
    timeStep = max(1,np.diff(x1).mean().days)*.9
    AbsMeanT = np.mean(abs(D['t_beta']))

    if plot:
        plt.figure(figsize = (14,8),dpi=80)
        plt.subplot(221)
        plt.bar(x1, np.cumsum(D['beta']), width=timeStep)
        plt.plot(x1, np.mean(D['beta'])*np.ones([T,]),color='r')
        plt.xlim(x1[0], x1[-1])
        plt.ylabel('Returns')
        plt.title('cumulated Beta')
        plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
    
        plt.subplot(222)
        plt.bar(x1, abs(D['t_beta']), color = 'r', width=timeStep)
        plt.plot(x1, AbsMeanT*np.ones([T,]),color='b')
        plt.xlim(x1[0], x1[-1])
        plt.ylabel('T-vals')
        plt.title('Absolute T-values')
        plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
    
        plt.subplot(223)
        plt.bar(x1, D['rsquared'], color = 'green', width=timeStep)
        plt.plot(x1, np.mean(D['rsquared'])*np.ones([T,]),color='r')
        plt.xlim(x1[0], x1[-1])
        plt.ylabel('R^2')
        plt.title('R Squared')
        plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
    
        plt.subplot(224)
        plt.hist(D['beta'].dropna(), bins = 'auto')
        plt.xlabel('Factor Return')
        plt.ylabel('Frequency')
        plt.title('Return Frequencies')
        plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
        plt.show()
        
        abs_tval_over2 = len(abs(D['t_beta'])[abs(D['t_beta'])>2]) / len(abs(D['t_beta']))
        
    
        print('Absolute mean T-val: %0.4f '%AbsMeanT)
        print('Mean T-val: %0.4f '%(np.mean(D['t_beta'])))
        print('Absolute T-val over 2: %0.4f '%(abs_tval_over2))
        print('Mean Factor Returns: %0.4f '%(np.mean(D['beta'])))
#    print('Annualised Factor Return: %s \n'%(np.mean(D['beta']) * 252))
    
    return D


def reg(fx, fy, **kwargs):
    '''
        Default OLS regression, accepts only pd.Series
        Params:
                factors - factor of Series
                returns - returns of Series
                plot - plot only applies to single variate regression
        optional:
                weight - once chosen, regress by WLS regression, with preferably 
                         circulating market cap as regression weight.
                         N.B. circulating market cap must take log
    '''
    if 't_stat' in kwargs.keys():
        t_flag = kwargs['t_stat']
    else:
        t_flag = True
        
    if 'R2' in kwargs.keys():
        R2_flag = kwargs['R2']
    else:
        R2_flag = True
    
        
    if type(fx) == np.ndarray and type(fy) == np.ndarray:
        f1_nonemp = np.where(~np.isnan(fx))[0]
        f2_nonemp = np.where(~np.isnan(fy))[0]
        f_nonemp = np.intersect1d(f1_nonemp, f2_nonemp)
        
        f_resid = np.full(fy.shape, np.nan)
        
        f_ytil = np.full(fy.shape, np.nan)
        
        if len(f1_nonemp) == 0 or len(f2_nonemp) == 0:
            return {'y_til':0, 'beta':0, 'alp':0, 'resid':0, 'rsquared':0, 't_beta':0}
        else:
            x = fx[f_nonemp]
            y = fy[f_nonemp]
            # print(np.sum(y))
            m_x = np.mean(x)
            m_y = np.mean(y)
            n = len(x)
            beta = (np.sum(x*y) - n*m_x*m_y)/(np.sum(x*x) - n*m_x*m_x)
            
            alp = m_y - beta*m_x
            
            SST = sum(y*y) - n*m_y*m_y
            y_til = alp + beta*x
            
            f_ytil[f_nonemp] = y_til
            
            resid = y - y_til
            f_resid[f_nonemp] = resid
            
            
            SSR = sum(y_til*y_til) - n*m_y*m_y
            R2 = SSR/SST
            
            t = np.sqrt(R2)/np.sqrt((1-R2)/(n-2))
            return {'y_til':f_ytil, 'beta':beta, 'alp':alp, 'resid':f_resid, 'rsquared':R2, 't_beta':t}

            
    else:
        raise TypeError('fx and fy must all be of np.ndarray type')

############## factor_test ############## 
def factor_test(Factors, Return, **kwargs): 
    '''
     因子测试函数，测试因子的有效性。当前有效性判断规则为 Normal IC 和 rank IC 。
     所测试因子将被标准化之后再进行测试。在标准化时，Normal IC 方法采用 atan+zscore
     方法， rank IC 采用 zscore 方法。
     Input:
        Factor: m-by-n factor matrix with m observations and n assets
        Return: m-by-n return matrix with m observations and n assets
        times: time vectore with m observations
        FSDflag: standardise the factor. 1 for yes (default), 0 for no
        DispFlag: Display test result. 1 for yes (Default), 0 for no
        PlotFlag: Plot test result.  1 for yes (Default), 0 for no
     Output:
        FT struct with:
           IC_rank: IC rank vector
           IC_normal: IC normal vector
           GroupRtnVec: m-by-10 Group return matrix with 10 groups
           GroupReturn: 1-by-10 vectore of group return, displayed in bar
           graph
           LongShort: return difference between group 10 and group 1(G10 - G1)
           LongShortNav: Net asset value for long G10 short G1
           Stat: statistics for the factore performance
    '''
    
    if 'times' in kwargs.keys():
        times = kwargs['times']
    else:
        times = [0] 

    if 'comsn' in kwargs.keys():
        comsn = kwargs['comsn']
    else:
        comsn = 0.000

    if 'NumOfGroup' in kwargs.keys():
        NumOfGroup = kwargs['NumOfGroup']
    else:
        NumOfGroup = 10

    if 'FSDflag' in kwargs.keys():
        FSDflag = kwargs['FSDflag']
        
    if 'met' in kwargs.keys():
        FSDflag = True
        met = kwargs['met']
    else:
        FSDflag = False
    
    if 'DispFlag' in kwargs.keys():
        DispFlag = kwargs['DispFlag']
    else:
        DispFlag = True

    if 'PlotFlag' in kwargs.keys():
        PlotFlag = kwargs['PlotFlag']
    else:
        PlotFlag = True
        
    if 'gp' in kwargs.keys():
        gp = kwargs['gp']
    else:
        gp = 0
    
    if 'met' in kwargs.keys():
        met = kwargs['met']
    else:
        met = 3
    
    if 'name' in kwargs.keys():
        name = kwargs['name']
    else:
        name = 'fac'
        
    if 'ma' in kwargs.keys():
        ma = kwargs['ma']
        
    if 'save_path' in kwargs.keys():
        save_path = kwargs['save_path']
    else:
        save_path = None
        
    if type(Factors) == pd.core.frame.DataFrame:        
        if type(Factors.index[0]) == pd._libs.tslibs.timestamps.Timestamp:
            x1 = Factors.index.date.tolist()
        else:
            x1 = Factors.index.tolist()
            
        timeStep = max(1,np.diff(x1).mean().days)*.9
        timeGap = (x1[-1]-x1[0]).days
        plt_flag = False # with str date display specific
        
        Factors = Factors.values
    else: 
        T = np.shape(Factors)[0]
        if len(times) == T:
            if type(times) == np.ndarray:
                x1 = to_pytime(times, outTy='a').tolist()
            elif not type(times) == list:
                x1 = times.tolist()
            
            timeStep = max(1,np.diff(x1).mean().days)*.9
            timeGap = (x1[-1]-x1[0]).days
            plt_flag = False # with str date display specific
        else:
            x1 = np.arange(0,T)
            timeStep = 1
            timeGap = T
            plt_flag = True # with num x-axis display specific
            
    if type(Return) == pd.core.frame.DataFrame:
        Return = Return.values
    
    nz_fac = Factors.copy()
    nz_fac[nz_fac==nz_fac] = 1
    nz_fac[nz_fac!=nz_fac] = 0
    nz_idx = np.sum(nz_fac,1)!=0
    warnings.filterwarnings("ignore") # ignore all warnings
    T,d = np.shape(Factors) # find factor size
    
    # Factor standardise or not
    SFactor1 = Factors
        
    Factorrank_ic = np.zeros(T)
    Factornormal_ic = np.zeros(T)
    
    for i in range(T-1): 
        if nz_idx[i]:
            Factorrank_ic[i+1] = rank_ic(SFactor1[i], Return[i+1])
            Factornormal_ic[i+1] = normal_ic(SFactor1[i], Return[i+1]) #SFactor2[i]
            
#    # plot the IC results    
#    TierNum = np.round(np.linspace(1,d,11)).astype('int32')
#    # find the order of factor, and drop the latest row to match return[t] vs factor[t-1]
#    IxFac = np.argsort(Factors)
#    IxFac = np.vstack([np.arange(0,d), IxFac])[:-1,:]
        
#    NumOfGroup = 50 # EDIT HERE FOR NUMBER OF GROUP BARS
    GroupRtnVec = np.zeros((T,NumOfGroup))
    d_10th = int(d/NumOfGroup)+1
    
    # preset for turnover calculation
    CodesG1 = np.empty([T,d_10th])
    CodesG1.fill(np.nan)
    CodesG10 = CodesG1.copy()
    turnoverG1 = np.zeros(T)
    turnoverG10 = turnoverG1.copy()
    G1 = np.array(0)
    G10 = np.array(0)
    
    for j in range(T):
        if nz_idx[j]:
            f = SFactor1[j,:]
            
            if j < T-1:
                r = Return[j+1,:]
                Ind = np.intersect1d(np.where(~np.isnan(f))[0],
                                     np.where(~np.isnan(r))[0]) # combination
            else:
                r = np.empty(d) 
                r.fill(0)
                Ind = np.where(~np.isnan(f))[0] 
            # find the null data in returns & factors
            
            f = f[Ind] # delete the null data in both returns and factors
            r = r[Ind]
            
            D = len(f)
            if D < NumOfGroup:
                continue
            else:
                IxFac = np.argsort(f)
                TierNum = np.round(np.linspace(0,len(f),NumOfGroup+1)).astype('int32') #10组分类
    
                RemainsG1 = len(np.intersect1d(G1, Ind[IxFac[:TierNum[1]]] )) #上期已在池中股票不计入换手
                RemainsG10 = len(np.intersect1d(G10, Ind[IxFac[TierNum[-2]:]] ))
                G1 = Ind[IxFac[:TierNum[1]]] #本期池中股票G1 = Ind[:10]
                G10 = Ind[IxFac[TierNum[-2]:]]
                
                G10Num = D-TierNum[-2]
                turnoverG1[j] = 1 - RemainsG1/TierNum[1]
                turnoverG10[j] = 1 - RemainsG10/G10Num
                CodesG1[j,:TierNum[1]] = G1 #CodesG1[j,:10] = G1\
                CodesG10[j,:G10Num] = G10[::-1]
                
                for i in range(NumOfGroup):
                    if j < T-1:
                        GroupRtnVec[j+1][i] = np.nanmean(r[IxFac[TierNum[i]:TierNum[i+1]]])
        elif j:
            CodesG1[j,:] = CodesG1[j-1,:]
            CodesG10[j,:] = CodesG10[j-1,:]

    GroupReturn = np.nanmean(GroupRtnVec,0) # the return of the group 
    mono = spearmanr(GroupReturn,np.arange(NumOfGroup))[0]*100
    if GroupReturn[0] <= GroupReturn[-1]: #根据分组收益率表现，确定多头组和空头组
        LongShort = GroupRtnVec[:,-1] - GroupRtnVec[:,0] - .5*comsn*(turnoverG1+turnoverG10)
        Excessive = GroupRtnVec[:,-1] - np.mean(GroupRtnVec,1) - .5*comsn*(turnoverG1+turnoverG10)
        Legend4 = f'Gp{NumOfGroup} - Gp1' # the return of the long-short strategy
    else:
        LongShort = GroupRtnVec[:,0] - GroupRtnVec[:,-1] - .5*comsn*(turnoverG1+turnoverG10)
        Excessive = GroupRtnVec[:,0] - np.mean(GroupRtnVec,1) - .5*comsn*(turnoverG1+turnoverG10)
        Legend4 = f'Gp1 - Gp{NumOfGroup}'
    
    GrpNav = np.cumprod(GroupRtnVec + 1, axis = 0)
    ExcessiveNav = np.cumprod(Excessive+1)
    LongShortNav = np.cumprod(LongShort+1)  # net asset value of long-short
    ExcessiveDif = Excessive.mean()
    LongShortDif = LongShort.mean() # calculate the average return of long-short


    if gp == 0: # not in the group-running type
        # print result
        c_rank = ~np.isnan(Factorrank_ic)
        c_normal = ~np.isnan(Factornormal_ic)
        RankMean = np.mean(Factorrank_ic[c_rank])
        RankStd = np.std(Factorrank_ic[c_rank])
        NormalMean = np.mean(Factornormal_ic[c_normal])
        NormalStd = np.std(Factornormal_ic[c_normal])

        WR_rank = len(Factorrank_ic[Factorrank_ic>0])/(T-1)*100.0
        WR_normal = len(Factornormal_ic[Factornormal_ic>0])/(T-1)*100.0
        
        Statrank_ic = [RankMean,
                      RankStd,
                      RankMean/RankStd,
                      WR_rank]
        Statnormal_ic = [NormalMean,
                        NormalStd,
                        NormalMean/NormalStd,
                        WR_normal]
        Sharpe = (LongShortNav[-1]**(365.2422/timeGap) - 1)\
            /((1e-6+np.std(LongShort))*np.sqrt(365.2422/timeStep))
        SharpeEx = (ExcessiveNav[-1]**(365.2422/timeGap) - 1)\
            /((1e-6+np.std(Excessive))*np.sqrt(365.2422/timeStep))
    
        FT = {'Rank_stats':Statrank_ic,
              'Normal_stats':Statnormal_ic,
              'Rank_IC':Factorrank_ic,
              'Normal_IC':Factornormal_ic,
              'GroupRtn':GroupRtnVec,
              'GroupNav':GrpNav,
              'Sharpe': Sharpe,
              'SharpeEx': SharpeEx,
              'Code_G1': CodesG1,
              'Code_G10': CodesG10,
              'Turnover_G1':turnoverG1,
              'Turnover_G10':turnoverG10,    
              'ExcessiveNav':ExcessiveNav,    
              'LongShortNaV':LongShortNav}
    
    elif gp == 1:
        # for gp_factor_test
        if not ma == 0:
            FT = np.empty((T,9))
            FT.fill(np.nan)
            for i in range(T-ma):
                # mean ic
                FT[i+ma-1, 2] = np.mean(Factorrank_ic[i:i+ma])
                FT[i+ma-1, 3] = np.mean(Factornormal_ic[i:i+ma])
                # IR
                FT[i+ma-1, 4] = np.std(Factorrank_ic[i:i+ma])
                FT[i+ma-1, 5] = np.std(Factornormal_ic[i:i+ma])
                # IC_IR  
                if FT[i+ma-1,4] == 0:
                    FT[i+ma-1,6] = 0
                else:
                    FT[i+ma-1,6] = FT[i+ma-1,2]/FT[i+ma-1,4] 
 
                if FT[i+ma-1,5] == 0:
                    FT[i+ma-1,7] = 0
                else:
                    FT[i+ma-1,7] = FT[i+ma-1,3]/FT[i+ma-1,5] 
                
                std = np.std(np.nan_to_num(Excessive[i:i+ma])) + 1e-9
                if std == 0:
                    FT[i+ma-1,8] = 0
                else:
                    FT[i+ma-1,8] = ((ExcessiveNav[i+ma-1]/ExcessiveNav[i])- 1)\
                    /(std)
 
        else:
            FT = np.empty((T,2))
            FT.fill(np.nan)
        
        FT[:,0] = Factorrank_ic
        FT[:,1] = Factornormal_ic
    elif gp == 2:
        # for gp_groupreturn
        GroupRtnVec = np.nan_to_num(GroupRtnVec)
        if not ma == 0:
            currgp = np.empty((T,4))
            currgp.fill(np.nan)
            
            FT = np.empty((T,2))
            FT.fill(np.nan)
            for i in range(ma-2):
                currgp[i+1, 0] = gmean(GroupRtnVec[:i, 0]+1) - 1
                currgp[i+1, 1] = gmean(GroupRtnVec[:i,-1]+1) - 1
                currgp[i+1, 2] = np.mean(GroupRtnVec[:i, 0])
                currgp[i+1, 3] = np.mean(GroupRtnVec[:i,-1])
                
                # compare the group return and generate factor
                FT[i+1, 0] = (currgp[i+1, 1] - currgp[i+1, 0])
                FT[i+1, 1] = (currgp[i+1, 3] - currgp[i+1, 2])
                    
                
            for i in range(T-ma):
                # mean group return
                currgp[i+ma-1, 0] = gmean(GroupRtnVec[i:i+ma, 0]+1) - 1
                currgp[i+ma-1, 1] = gmean(GroupRtnVec[i:i+ma,-1]+1) - 1
                currgp[i+ma-1, 2] = np.mean(GroupRtnVec[i:i+ma, 0])
                currgp[i+ma-1, 3] = np.mean(GroupRtnVec[i:i+ma,-1])

                # compare the group return and generate factor
                FT[i+ma-1, 0] = (currgp[i+ma-1, 1] - currgp[i+ma-1, 0])
                FT[i+ma-1, 1] = (currgp[i+ma-1, 3] - currgp[i+ma-1, 2])
                
        else:
            FT = np.empty((T,1))
            FT.fill(np.nan)
            
            for i in range(T):
                FT[i+1, 0] = GroupRtnVec[i,1] - GroupRtnVec[i,0] 
                
    elif gp == 3:     
        # for factor_test_all
        c_rank = ~np.isnan(Factorrank_ic)
        c_normal = ~np.isnan(Factornormal_ic)
        RankMean = np.mean(Factorrank_ic[c_rank])
        RankStd = np.std(Factorrank_ic[c_rank])
        NormalMean = np.mean(Factornormal_ic[c_normal])
        NormalStd = np.std(Factornormal_ic[c_normal])
        Sharpe = (LongShortNav[-1]**(365.2422/timeGap) - 1)\
            /((1e-6+np.std(LongShort))*np.sqrt(365.2422/timeStep))
        SharpeEx = (ExcessiveNav[-1]**(365.2422/timeGap) - 1)\
            /((1e-6+np.std(Excessive))*np.sqrt(365.2422/timeStep))
        WR_rank = len(Factorrank_ic[Factorrank_ic>0])/(T-1)*100.0
        WR_normal = len(Factornormal_ic[Factornormal_ic>0])/(T-1)*100.0
        FT = [RankMean, NormalMean, RankStd, NormalStd, 
              RankMean/RankStd,NormalMean/NormalStd,
              WR_rank, 
              WR_normal,
              LongShortDif*100.0, Sharpe, 
              ExcessiveDif*100.0, SharpeEx]

        FT = np.array(FT)

    if DispFlag == 1 and (gp == 3 or gp == 0):
        print('Factor name: %s'%name)
        print('Average rank IC: %.4f, normal IC: %.4f.'%(RankMean,NormalMean))
        print('Std Dev rank IC: %0.4f, normal IC: %0.4f.'%(RankStd, NormalStd))
        print('Rank IR: %0.4f, normal IR: %0.4f.'%(RankMean/RankStd, NormalMean/NormalStd))
        print('Win rate rank IC: %0.4f%%, normal IC: %0.4f%%.'\
            %(WR_rank, WR_normal))
        print('Long-short avg return: %0.4f%%.'%(LongShortDif*100.0))
        print('Long short Sharpe: %0.4f.'%Sharpe)
        print('Excessive avg return: %0.4f%%.'%(ExcessiveDif*100.0))
        print('Excessive Sharpe: %0.4f.'%SharpeEx)
            
        # plot the graph
    if PlotFlag and (gp == 3 or gp == 0):
        x2 = np.arange(1,NumOfGroup+1)
        if not plt_flag:
            p0 = plt.figure(figsize=(14,8),dpi=80)
            p0.suptitle(name, fontsize=20)
            p1= plt.subplot(221)
            p11 = p1.twinx()
            p1.bar(x1, np.squeeze(Factorrank_ic), width=timeStep, 
                    align='edge', color='b', linewidth=0)
            p1.plot(x1, np.mean(Factorrank_ic[~np.isnan(Factorrank_ic)])*np.ones([T,]),
                     color='r',linewidth=1.5)
            p11.plot(x1, np.cumsum(np.nan_to_num(Factorrank_ic)), 
                     color='g',linewidth=1.5)
 
            p1.set_xlim(x1[0], x1[-1])
            p1.set_ylim(-.75,.75)
            p1.set_ylabel('Rank IC')    
#                p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#                p11.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#                plt.setp(plt.xticks()[1], rotation=30, ha='right')
            
            
            p2=plt.subplot(222)
            p21 = p2.twinx()
            p2.bar(x1,np.squeeze(Factornormal_ic),width=timeStep, 
                    align='edge', color='b', linewidth=0)
            p2.plot(x1, np.mean(Factornormal_ic[~np.isnan(Factornormal_ic)])*np.ones([T,]),
                     color='r')
            p21.plot(x1, np.cumsum(np.nan_to_num(Factornormal_ic)), 
                     color='g',linewidth=1.5)
             
            p2.set_xlim(x1[0], x1[-1])
            p2.set_ylim(-.75,.75)
            p2.set_ylabel('Normal IC')
#                p2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#                plt.setp(plt.xticks()[1], rotation=30, ha='right');
            
            # plot 10 group return
            
            plt.subplot(223)
            plt.bar(x2,GroupReturn)
            plt.xlim(0, NumOfGroup+1)
            plt.text(int(NumOfGroup/2), 0, 'Rank corr: %d%%.'%mono,
                     fontsize=20) 
            plt.ylabel('Group return')
            
            p4=plt.subplot(224)
            
            p41 = p4.twinx()
            p4.plot(x1, LongShortNav, label=Legend4)
            p41.plot(x1, ExcessiveNav,color='r', label='ExcessiveNav')
            p4.set_xlim(x1[0], x1[-1]) 
            p4.set_ylabel('Long short NaV')
            p4.legend( loc='upper left')
            p41.legend( loc='upper right')
#            plt.plot(x1, LongShortNav)
#            plt.plot(x1, ExcessiveNav)
#            plt.xlim(x1[0], x1[-1])
#            plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
#            plt.ylabel('Long short NaV')
#            plt.legend((Legend4,'ExcessiveNav'))
            p4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(plt.xticks()[1], rotation=30, ha='right')
            
            denum = NumOfGroup*2.0
            p5 = plt.figure(figsize=(14,8),dpi=80)
            p5.suptitle(name)
            for i in range(NumOfGroup):
                if i == 0:
                    plt.plot(x1, GrpNav[:,i],
                             linewidth=3, color=[0, 0, (i+NumOfGroup)/denum])
                elif i == NumOfGroup-1:
                    plt.plot(x1, GrpNav[:,i], 
                             linewidth=3, color=[0, 0, (i+NumOfGroup)/denum])
                else:
                    plt.plot(x1, GrpNav[:,i], 
                             color=[0, 0, (i+NumOfGroup)/denum])
            
            ax = plt.gca()
            plt.xlim(x1[0], x1[-1]) 
            plt.ylabel('Grouped NaV')
            plt.legend(['Group %s'%(i+1) for i in range(NumOfGroup)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(plt.xticks()[1], rotation=30, ha='right')

            
        elif plt_flag == 1:

            p0 = plt.figure(figsize=(14,8),dpi=80)
            p0.suptitle(name, fontsize=20)
            p1= plt.subplot(221)
            p11 = p1.twinx()
            p1.bar(x1, np.squeeze(Factorrank_ic), width=timeStep, 
                    align='edge', color='b', linewidth=0)
            p1.plot(x1, np.mean(Factorrank_ic[~np.isnan(Factorrank_ic)])*np.ones([T,]),
                     color='r',linewidth=1.5)
            p11.plot(x1, np.cumsum(np.nan_to_num(Factorrank_ic)), 
                     color='g',linewidth=1.5) 
            p1.set_xlim(x1[0], x1[-1])
            p1.set_ylim(-.75,.75)
            p1.set_ylabel('Rank IC')    
#                p1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#                p11.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#                plt.setp(plt.xticks()[1], rotation=30, ha='right')
            
            p2=plt.subplot(222)
            p21 = p2.twinx()
            p2.bar(x1,np.squeeze(Factornormal_ic),width=timeStep, 
                    align='edge', color='b', linewidth=0)
            p2.plot(x1, np.mean(Factornormal_ic[~np.isnan(Factornormal_ic)])*np.ones([T,]),
                     color='r')
            p21.plot(x1, np.cumsum(np.nan_to_num(Factornormal_ic)), 
                     color='g',linewidth=1.5)
             
            p2.set_xlim(x1[0], x1[-1])
            p2.set_ylim(-.75,.75)
            p2.set_ylabel('Normal IC')

            # plot 10 group return
            plt.subplot(223)
            plt.bar(x2,GroupReturn)
            plt.xlim(0, NumOfGroup+1)
            plt.text(int(NumOfGroup/2), 0, 'Rank corr: %d%%.'%mono,
                     fontsize=20) 
            plt.ylabel('Group return')
            
            p4=plt.subplot(224)
            p41 = p4.twinx()
            p4.plot(x1, LongShortNav, label=Legend4)
            p41.plot(x1, ExcessiveNav,color='r', label='ExcessiveNav')
            p4.set_xlim(x1[0], x1[-1]) 
            p4.set_ylabel('Long short NaV')
            p4.legend( loc='upper left')
            p41.legend( loc='upper right')
            
            p5 = plt.figure(figsize=(14,8),dpi=80)
            p5.suptitle(name,fontsize=20)
            denum = NumOfGroup*2.0
            for i in range(NumOfGroup):
                if i == 0:
                    plt.plot(x1, GrpNav[:,i], 
                             linewidth=3, color=[0, 0, (i+NumOfGroup)/denum])
                elif i == NumOfGroup-1:
                    plt.plot(x1, GrpNav[:,i], 
                             linewidth=3, color=[0, 0, (i+NumOfGroup)/denum])
                else:
                    plt.plot(x1, GrpNav[:,i], 
                             color=[0, 0, (i+NumOfGroup)/denum])
                    
            plt.grid(color='r', linestyle='--', linewidth=0.5,alpha=0.3)
            plt.ylabel('Grouped NaV')
            plt.legend(['Group %s'%(i+1) for i in range(NumOfGroup)])
          
        if save_path is not None:
            p0.savefig('%s/FT_%s.png'%(save_path, name), dpi=120)
            p5.savefig('%s/FT_%s_nav.png'%(save_path, name), dpi=120)
            plt.close('all')
        else:
            plt.show()        
        # outputs

    return FT



# [EOF]
