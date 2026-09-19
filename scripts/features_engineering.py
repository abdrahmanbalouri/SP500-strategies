import os

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

TEST_DATE = pd.Timestamp("2017-01-01")

FEATURES = [
    "bb_pct",
    "rsi",
    "macd",
    "momentum_60d",
]


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["close"]

    g["fwd_return"] = c.shift(-2) / c.shift(-1) - 1
    g["target"] = np.sign(g["fwd_return"])

    target_end_date = g["date"].shift(-2)
    crosses_test_boundary = (
        (g["date"] < TEST_DATE)
        & (target_end_date >= TEST_DATE)
    )

    g.loc[
        crosses_test_boundary,
        ["fwd_return", "target"]
    ] = np.nan

    g["bb_pct"] = BollingerBands(close=c).bollinger_pband()
    g["rsi"] = RSIIndicator(close=c).rsi()
    g["macd"] = MACD(close=c).macd()
    g["momentum_60d"] = c.pct_change(60, fill_method=None)

    return g


def build_dataset() -> pd.DataFrame:
    raw = pd.read_csv(
        os.path.join(DATA, "all_stocks_5yr.csv"),
        parse_dates=["date"]
    )

    raw = raw.sort_values(["Name", "date"])

    return (
        pd.concat(
            [_per_ticker(g) for _, g in raw.groupby("Name", sort=False)],
            ignore_index=True
        )
        .set_index(["date", "Name"])
        .rename_axis(["date", "ticker"])
        .loc[:, FEATURES + ["target", "fwd_return"]]
        .sort_index()
    )


def split_train_test(df: pd.DataFrame):
    d = df.index.get_level_values("date")
    return df.loc[d < TEST_DATE], df.loc[d >= TEST_DATE]


if __name__ == "__main__":
    data = build_dataset()

    tr, te = split_train_test(data)

    print(data.shape, "train", len(tr), "test", len(te))
    print(
        tr.index.get_level_values("date").min(),
        "->",
        te.index.get_level_values("date").max()
    )
