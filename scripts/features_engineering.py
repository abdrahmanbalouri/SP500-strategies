import os

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "preprocessed.pkl")
TEST_DATE = pd.Timestamp("2017-01-01")

FEATURES = [
    "bb_upper", "bb_mid", "bb_lower", "bb_pct", "bb_width",
    "rsi", "macd", "macd_signal", "macd_hist",
]


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["close"]
    # target on D = sign(return(D+1, D+2))
    g["fwd_return"] = c.shift(-2) / c.shift(-1) - 1
    g["target"] = np.sign(g["fwd_return"])


    target_end_date = g["date"].shift(-2)
    crosses_test_boundary = (g["date"] < TEST_DATE) & (target_end_date >= TEST_DATE)
    g.loc[crosses_test_boundary, ["fwd_return", "target"]] = np.nan

    bb = BollingerBands(close=c, window=20, window_dev=2)
    g["bb_upper"] = bb.bollinger_hband()
    g["bb_mid"] = bb.bollinger_mavg()
    g["bb_lower"] = bb.bollinger_lband()
    g["bb_pct"] = bb.bollinger_pband()
    g["bb_width"] = bb.bollinger_wband()

    g["rsi"] = RSIIndicator(close=c, window=14).rsi()

    m = MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
    g["macd"] = m.macd()
    g["macd_signal"] = m.macd_signal()
    g["macd_hist"] = m.macd_diff()
    return g


def build_dataset(use_cache=True) -> pd.DataFrame:
    if use_cache and os.path.exists(CACHE):
        return pd.read_pickle(CACHE)

    raw = pd.read_csv(os.path.join(DATA, "all_stocks_5yr.csv"), parse_dates=["date"])
    raw = raw.sort_values(["Name", "date"])
    out = (
        pd.concat([_per_ticker(g) for _, g in raw.groupby("Name", sort=False)], ignore_index=True)
        .set_index(["date", "Name"])
        .rename_axis(["date", "ticker"])
        .loc[:, FEATURES + ["target", "fwd_return"]]
        .sort_index()
    )
    out.to_pickle(CACHE)
    return out


def split_train_test(df: pd.DataFrame):
    d = df.index.get_level_values("date")
    return df.loc[d < TEST_DATE], df.loc[d >= TEST_DATE]


if __name__ == "__main__":
    data = build_dataset(use_cache=False)
    tr, te = split_train_test(data)
    print(data.shape, "train", len(tr), "test", len(te))
    print(tr.index.get_level_values("date").min(), "->", te.index.get_level_values("date").max())
