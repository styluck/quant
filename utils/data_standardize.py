# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 16:16:57 2025

@author: lich5
"""
import numpy as np
from scipy.stats import boxcox

def _my_boxcox(r_data, return_lambda=False):
    """对非缺失正值做 Box-Cox 变换；可选返回估计的 lambda。"""
    row_data = r_data.to_numpy(dtype=float, copy=True)
    nonempty = np.where(~np.isnan(row_data))[0]
    a, lam = boxcox(row_data[nonempty])
    row_data[nonempty] = a
    if return_lambda:
        return row_data, lam
    return row_data

def preprocess(data):
    """逐期截面 MAD 缩尾与 Z-score；输入为日期×股票宽表。"""
    median = data.median(axis=1)
    mad = data.sub(median, axis=0).abs().median(axis=1)
    lower = (median - 3 * 1.4826 * mad).where(mad > 0)
    upper = (median + 3 * 1.4826 * mad).where(mad > 0)
    data = data.clip(lower=lower, upper=upper, axis=0)
    return data.sub(data.mean(axis=1), axis=0).div(
        data.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
