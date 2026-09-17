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
]


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["close"]

    # Target
    g["fwd_return"] = c.shift(-2) / c.shift(-1) - 1
    g["target"] = np.sign(g["fwd_return"])

    # Prevent data leakage
    target_end_date = g["date"].shift(-2)
    crosses_test_boundary = (
        (g["date"] < TEST_DATE)
        & (target_end_date >= TEST_DATE)
    )

    g.loc[
        crosses_test_boundary,
        ["fwd_return", "target"]
    ] = np.nan

    # Features
    bb = BollingerBands(close=c)
    g["bb_pct"] = bb.bollinger_pband()

    rsi = RSIIndicator(close=c)
    g["rsi"] = rsi.rsi()

    m = MACD(close=c)
    g["macd"] = m.macd()

    return g


def build_dataset() -> pd.DataFrame:
    raw = pd.read_csv(
        os.path.join(DATA, "all_stocks_5yr.csv"),
        parse_dates=["date"]
    )

    raw = raw.sort_values(["Name", "date"])

    out = (
        pd.concat(
            [_per_ticker(g) for _, g in raw.groupby("Name", sort=False)],
            ignore_index=True
        )
        .set_index(["date", "Name"])
        .rename_axis(["date", "ticker"])
        .loc[:, FEATURES + ["target", "fwd_return"]]
        .sort_index()
    )

    return out


def split_train_test(df: pd.DataFrame):
    d = df.index.get_level_values("date")

    train = df.loc[d < TEST_DATE]
    test = df.loc[d >= TEST_DATE]

    return train, test


if __name__ == "__main__":
    data = build_dataset()

    tr, te = split_train_test(data)

    print(data.shape, "train", len(tr), "test", len(te))
    print(
        tr.index.get_level_values("date").min(),
        "->",
        te.index.get_level_values("date").max()
    )