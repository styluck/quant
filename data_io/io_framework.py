# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 10:49:59 2025

@author: lich5
将该文件放于代码目录下的data_io文件夹中使用。
"""
# io.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional 
import pandas as pd

def _data_dir():
    return Path(__file__).resolve().parent.parent / "data"

def _read_one(field: str, market: str) -> Optional[pd.DataFrame]:
    """
    读取单个 CSV：例如 field='close', market='sh' -> 读取 ./close_sh.csv

    约定的 CSV 格式：
      - 第1行：列名（第1列建议为 'date' 或空；第2列开始为资产名称）
      - 第1列：日期字符串（YYYY-MM-DD 等可被 pandas 解析的格式） 

    返回：
      pandas.DataFrame，index 为 DatetimeIndex，列为各资产名；若文件不存在返回 None
    """
    if market is None:
        filename = f"{field}.csv"
    else:
        filename = f"{field}_{market}.csv"
        
    csv_file = os.path.join(_data_dir(), filename)
    # csv_path = Path(f"{field}_{market}.csv")
    if not os.path.exists(csv_file):
        return None

    # 读取：第一行是表头，第一列作为索引
    df = pd.read_csv(
        csv_file,
        header=0,
        index_col=0,
        encoding="utf-8",
    )

    # 解析日期为 DatetimeIndex，丢弃解析失败的行
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # 统一将数据转为 float（无法解析的置为 NaN）
    df = df.apply(pd.to_numeric, errors="coerce")

    # 按时间排序
    df = df.sort_index()

    return df

def load_benchmark(startdate: str, enddate: str) -> pd.DataFrame:
    
    both = _read_one('benchmark', None)
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)
    both = both.loc[(both.index >= start) & (both.index <= end)]
    
    return both
    
def load_data(field: str, startdate: str, enddate: str) -> pd.DataFrame:
    """
    载入指定 field 的市场数据。

    输入参数
    --------
    field : str
        字段名（例如：'open' 或 'close'），会读取 `field_sh.csv`
    startdate : str
        开始日期（含），如 '2020-01-01'
    enddate : str
        结束日期（含），如 '2024-12-31'

    输出
    ----
    pandas.DataFrame
        index 为日期；列名为资产名。
    """
    one = _read_one(field, 'sh')
    if one is None or one.empty:
        raise FileNotFoundError(
            f"No CSV found for field='{field}'. Expected file like '{field}_sh.csv'."
        )

    # 截取日期区间（左右闭区间）
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)
    one = one.loc[(one.index >= start) & (one.index <= end)]

    return one


def write_data(df: pd.DataFrame, field: str) -> None:
    """
    将指定 field 的 DataFrame 写回 CSV。

    CSV 格式
    --------
    - 第1行写列名（第1列为索引名 'date'）
    - 第1列为日期字符串（按 YYYY-MM-DD 导出）
    - 第2列起为资产数据（float）
    """ 

    if df.index.name is None:
        df = df.copy()
        df.index.name = "date"
        
    outfile = f"{field}.csv"
    df.to_csv(outfile, date_format="%Y-%m-%d", encoding="utf-8")
        
#%% ------------------------- main -------------------------      
if __name__=='__main__': 
    
    # 载入单个字段（如 'close'），并合并 sh/sz 两市
    df_close = load_data("close", "2020-01-01", "2024-12-31")
    print(df_close.columns)  # MultiIndex: level0 为 'sh'/'sz'，level1 为资产名
    
    # 载入所有字段（目录下凡是 *_sh.csv 或 *_sz.csv 的都会被识别为字段）
    # all_data = load_all("2020-01-01", "2024-12-31")
    # all_data 形如：{'open': df_open, 'close': df_close, ...}
    
    # 将最近一次缓存的 close 写回到 close_sh.csv 和 close_sz.csv
    write_data(df_close, "close")

    
    
# [EOF]
