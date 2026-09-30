"""Lecture 1 homework: download Shanghai A-share daily data from BaoStock.

The script saves one long-format CSV per stock and adds the daily backward
adjustment factor. Existing files containing ``backAdjustFactor`` are skipped
by default, so an interrupted download can be resumed safely.
"""

from pathlib import Path

import baostock as bs
import pandas as pd


# 要求的行情起止日期。
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# 复权因子从尽可能早的日期开始查询，避免遗漏样本开始前发生的分红送转。
FACTOR_START_DATE = "1990-01-01"

# 日线接口返回的字段；adjustflag="3" 时价格保持原始口径。
FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "turn,tradestatus,pctChg,isST"
)

# False 表示已有且包含后复权因子的文件不重复下载。
OVERWRITE = False

# 使用相对脚本位置构造路径，避免依赖个人电脑上的绝对路径。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
STOCK_DIR = DATA_DIR  # Keep every CSV in the package's single data folder.
BASIC_FILE = DATA_DIR / "sse_stock_basic.csv"


def to_frame(result):
    """将 BaoStock 查询结果逐行转换为 DataFrame。"""
    # 接口调用失败时立即报告错误，避免保存不完整文件。
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)

    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def get_shanghai_stock_list():
    """获取沪市 A 股列表，并保留上市和退市证券的基本信息。"""
    basic = to_frame(bs.query_stock_basic())

    # type="1" 表示股票；sh.6 是 BaoStock 的沪市 A 股代码格式。
    mask = (basic["type"] == "1") & basic["code"].str.startswith("sh.6")
    stocks = basic.loc[mask].sort_values("code").reset_index(drop=True)

    # 保存 ipoDate、outDate、status 等字段，后续可判断证券何时可投资。
    stocks.to_csv(BASIC_FILE, index=False, encoding="utf-8-sig")
    return stocks

# 请你自己补充获取深市 A 股列表的函数。

def download_one_stock(code):
    """下载一只股票的原始日线与后复权因子，并保存为长表。"""
    output_file = STOCK_DIR / f"{code}.csv"

    # 断点续传：只有现有文件已经含复权因子时才跳过。
    if output_file.exists() and output_file.stat().st_size > 0 and not OVERWRITE:
        existing_columns = pd.read_csv(output_file, nrows=0).columns
        if "backAdjustFactor" in existing_columns:
            return "skipped"

    # 第一步：下载不复权日线行情。
    price_result = bs.query_history_k_data_plus(
        code,
        FIELDS,
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="d",
        adjustflag="3",  # 3 表示不复权，保留真实历史成交价格。
    )
    stock = to_frame(price_result).reindex(columns=FIELDS.split(","))

    # 第二步：复权因子必须通过独立接口查询。
    factor_result = bs.query_adjust_factor(
        code=code,
        start_date=FACTOR_START_DATE,
        end_date=END_DATE,
    )
    factor = to_frame(factor_result)

    # 第三步：把除权除息日上的因子扩展到每个交易日。
    stock["date"] = pd.to_datetime(stock["date"])
    if factor.empty:
        # 样本内没有分红送转时，后复权因子等于 1。
        stock["backAdjustFactor"] = 1.0
    else:
        factor["dividOperateDate"] = pd.to_datetime(
            factor["dividOperateDate"], errors="coerce"
        )
        factor["backAdjustFactor"] = pd.to_numeric(
            factor["backAdjustFactor"], errors="coerce"
        )
        factor = (
            factor[["dividOperateDate", "backAdjustFactor"]]
            .dropna()
            .sort_values("dividOperateDate")
            .drop_duplicates("dividOperateDate", keep="last")
        )

        # 对每个交易日取当日或此前最近一次生效的后复权因子。
        stock = pd.merge_asof(
            stock.sort_values("date"),
            factor,
            left_on="date",
            right_on="dividOperateDate",
            direction="backward",
        ).drop(columns="dividOperateDate")
        stock["backAdjustFactor"] = stock["backAdjustFactor"].fillna(1.0)

    # 后复权价格可在后续研究中按 close * backAdjustFactor 计算。
    stock["date"] = stock["date"].dt.strftime("%Y-%m-%d")
    stock.to_csv(output_file, index=False, encoding="utf-8-sig")
    return f"{len(stock):,} rows"
 

if __name__ == "__main__":
    # 创建统一数据目录，并建立 BaoStock 会话。
    STOCK_DIR.mkdir(parents=True, exist_ok=True)

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

    try:
        # 先取得完整股票列表，再逐只下载，便于显示进度与失败代码。
        stocks = get_shanghai_stock_list()[:100]  # 仅下载前 100 只股票作为示例。
        total = len(stocks)
        print(f"Found {total:,} Shanghai A-share securities.")

        for number, code in enumerate(stocks["code"], start=1):
            try:
                status = download_one_stock(code)
                print(f"[{number:>4}/{total}] {code}: {status}")
            except Exception as error:
                # 单只股票失败时继续批量任务，稍后可重新运行脚本补齐。
                print(f"[{number:>4}/{total}] {code}: FAILED - {error}")
    finally:
        # 无论中途是否发生异常，都释放 BaoStock 会话。
        bs.logout()

    print(f"Stock files: {STOCK_DIR}")
    print(f"Security metadata: {BASIC_FILE}")
