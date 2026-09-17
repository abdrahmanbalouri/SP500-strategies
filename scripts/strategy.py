"""Long-only strategy: sklearn signal × returns; empyrical for drawdown."""
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from empyrical import max_drawdown
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(__file__))
from features_engineering import FEATURES, TEST_DATE, build_dataset, split_train_test
from gridsearch import make_date_folds, prepare_training_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "strategy")
MODEL_DIR = os.path.join(ROOT, "results", "selected-model")


def to_weights(signal):
    long = (signal > 0.5).astype(float)
    return (long / long.groupby(level="date").transform("sum").replace(0, np.nan)).rename("weight")


def sp500_cum(dates):
    close = (
        pd.read_csv(os.path.join(ROOT, "data", "HistoricalPrices.csv"), parse_dates=["Date"])
        .rename(columns=lambda c: c.strip())
        .set_index("Date")
        .sort_index()["Close"]
    )
    return (1 + close.pct_change().reindex(dates).fillna(0)).cumprod() - 1


def fold_lengths(folds):
    return "\n".join(
        f"- fold {i}: train {len(tr)} days "
        f"({pd.Timestamp(tr[0]).date()} → {pd.Timestamp(tr[-1]).date()}), "
        f"validation {len(va)} days "
        f"({pd.Timestamp(va[0]).date()} → {pd.Timestamp(va[-1]).date()})"
        for i, (tr, va) in enumerate(folds, 1)
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = build_dataset()
    train, test = split_train_test(data)
    _, _, dates = prepare_training_data(train)
    folds = make_date_folds(dates)
    pipe = joblib.load(os.path.join(MODEL_DIR, "selected_model.pkl"))

    oof = pd.read_csv(
        os.path.join(MODEL_DIR, "ml_signal.csv"), parse_dates=["date"]
    ).set_index(["date", "ticker"])["ml_signal"]

    Xtr, ytr, _ = prepare_training_data(train)
    Xte = test.dropna(subset=["fwd_return"])[FEATURES]
    test_sig = pd.Series(
        clone(pipe).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
        index=Xte.index,
        name="ml_signal",
    )
    signal = pd.concat([oof, test_sig]).sort_index()

    strategy = to_weights(signal)
    aligned = data[["fwd_return"]].join(strategy, how="inner").dropna()
    pnl = (aligned["weight"] * aligned["fwd_return"]).groupby(level="date").sum().sort_index()
    cum = pnl.cumsum()
    sp = sp500_cum(pnl.index)

    def metrics(mask):
        if not mask.any():
            return {"PnL": 0.0, "MaxDrawdown": 0.0}
        r = pnl[mask]
        return {"PnL": float(r.sum()), "MaxDrawdown": float(max_drawdown(r))}

    results = pd.DataFrame(
        {"train": metrics(cum.index < TEST_DATE), "test": metrics(cum.index >= TEST_DATE)}
    ).T
    results.to_csv(os.path.join(OUT, "results.csv"))

    # same y-scale for strategy & SP500
    both = pd.DataFrame({"Strategy PnL": cum, "SP500 PnL": sp})
    ax = both.plot(figsize=(11, 5), title="Strategy vs SP500", color=["steelblue", "gray"])
    ax.axvline(TEST_DATE, color="red", ls="--", label="Train / Test")
    ax.set(xlabel="Date", ylabel="Cumulative PnL")
    ax.legend(loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUT, "strategy.png"), dpi=140)
    plt.close()

    report = f"""# Strategy report

## Features
Bollinger, RSI(14), MACD — **`ta`**.
Target on day D: `sign(return(D+1, D+2))`.

## Pipeline (sklearn)
- Imputer / Scaler / LogisticRegression via `make_pipeline` + `GridSearchCV`

## Cross-validation
Expanding Time Series Split, 10 folds (`TimeSeriesSplit` date-level + `GridSearchCV`).

Fold lengths:
{fold_lengths(folds)}

## Strategy
Long-only (`ml_signal > 0.5`), $1/day. PnL = weight × fwd_return.
Drawdown: **`empyrical.max_drawdown`**.

![PnL](strategy.png)

## Metrics

| set | PnL | Max drawdown |
|-----|-----|--------------|
| train | {results.loc['train', 'PnL']:.4f} | {results.loc['train', 'MaxDrawdown']:.4f} |
| test | {results.loc['test', 'PnL']:.4f} | {results.loc['test', 'MaxDrawdown']:.4f} |
"""
    open(os.path.join(OUT, "report.md"), "w").write(report)
    print(results)
    print("saved strategy.png, results.csv, report.md")
