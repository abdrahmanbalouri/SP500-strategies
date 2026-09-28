"""Long-only strategy: sklearn signal multiplied by forward returns."""
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features_engineering import FEATURES, TEST_DATE, build_dataset, split_train_test
from gridsearch import make_date_folds, prepare_training_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "strategy")
MODEL_DIR = os.path.join(ROOT, "results", "selected-model")

MOMENTUM_QUANTILE = 0.90


def to_weights(signal, momentum):
    selected_data = pd.concat(
        [signal.rename("signal"), momentum.rename("momentum")], axis=1
    ).dropna()

    cutoff = selected_data["momentum"].groupby(level="date").transform(
        lambda values: values.quantile(MOMENTUM_QUANTILE)
    )

    long = (
        (selected_data["signal"] > 0.5)
        & (selected_data["momentum"] >= cutoff)
    ).astype(float)

    return (
        long
        / long.groupby(level="date").transform("sum").replace(0, np.nan)
    ).rename("weight")


def sp500_forward_returns(dates):
    close = (
        pd.read_csv(
            os.path.join(ROOT, "data", "HistoricalData.csv"),
            parse_dates=["Date"],
            date_format="%m/%d/%y",
        )
        .rename(columns=lambda c: c.strip())
        .set_index("Date")
        .sort_index()["Close"]
    )

    forward_return = close.shift(-2) / close.shift(-1) - 1

    return forward_return.reindex(dates).fillna(0)


def max_drawdown(cum_r):
    return float((cum_r.cummax() - cum_r).max())


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

    dates = prepare_training_data(train)[2]

    folds = make_date_folds(dates)

    pipe = joblib.load(
        os.path.join(MODEL_DIR, "selected_model.pkl")
    )

    oof = pd.read_csv(
        os.path.join(MODEL_DIR, "ml_signal.csv"),
        parse_dates=["date"],
    ).set_index(["date", "ticker"])["ml_signal"]

    Xte = test.dropna(subset=["fwd_return"])[FEATURES]

    test_sig = pd.Series(
        pipe.predict_proba(Xte)[:, 1],
        index=Xte.index,
        name="ml_signal",
    )

    signal = pd.concat([oof, test_sig]).sort_index()

    strategy = to_weights(
        signal,
        data["momentum_60d"],
    )

    aligned = (
        data[["fwd_return"]]
        .join(strategy, how="inner")
        .dropna()
    )

    pnl = (
        aligned["weight"] * aligned["fwd_return"]
    ).groupby(level="date").sum().sort_index()

    cum = pnl.cumsum()

    benchmark_pnl = sp500_forward_returns(pnl.index)
    benchmark_cum = benchmark_pnl.cumsum()


    # =========================
    # Metrics
    # =========================

    def metrics(mask):
        r = pnl[mask]
        benchmark = benchmark_pnl[mask]

        strategy_total = float(r.sum())
        benchmark_total = float(benchmark.sum())

        strategy_cum = r.cumsum()

        return {
            "PnL": strategy_total,
            "SP500PnL": benchmark_total,
            "MaxDrawdown": max_drawdown(strategy_cum),
        }


    results = pd.DataFrame(
        {
            "train": metrics(cum.index < TEST_DATE),
            "test": metrics(cum.index >= TEST_DATE),
        }
    ).T

    results.to_csv(
        os.path.join(OUT, "results.csv")
    )


    # =========================
    # Plot
    # =========================

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Strategy
    ax1.plot(
        cum.index,
        cum,
        label="Strategy PnL",
        color="red"
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Strategy PnL")

    # S&P 500
    ax2 = ax1.twinx()

    ax2.plot(
        benchmark_cum.index,
        benchmark_cum,
        label="SP500 PnL",
    )

    ax2.set_ylabel("SP500 PnL")

    # Same Y scale
    limit = max(
        abs(cum.min()),
        abs(cum.max()),
        abs(benchmark_cum.min()),
        abs(benchmark_cum.max()),
    )

    ax1.set_ylim(-limit, limit)
    ax2.set_ylim(-limit, limit)

    # Train / Test boundary
    ax1.axvline(
        TEST_DATE,
        ls="--",
        label="Train / Test",
        color="black"
    )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.suptitle("Strategy vs SP500")

    fig.tight_layout()

    fig.savefig(
        os.path.join(OUT, "strategy.png"),
        dpi=140,
    )

    plt.close()


    # =========================
    # Report
    # =========================

    report = f"""# Strategy report

## Features
Bollinger %B, RSI(14), MACD, and 60-day momentum, computed independently for
each ticker from prices available through day D.
Target on day D: `sign(return(D+1, D+2))`.

## Pipeline (sklearn)
- Median imputation
- Standard scaling
- No dimension reduction
- Logistic regression selected with `GridSearchCV`

## Cross-validation
Expanding Time Series Split, 10 date-level folds with a two-trading-date purged
gap matching the target horizon (`TimeSeriesSplit` + `GridSearchCV`).

Fold lengths:
{fold_lengths(folds)}

## Strategy
Long-only: require `ml_signal > 0.5`, then retain stocks in the top 10% of
60-day momentum on that date. The 90th-percentile momentum threshold is a
predetermined strategy rule; it was not selected using test-set results. On
each date, $1 is divided equally among selected stocks; if none is selected,
$0 is invested. PnL = weight × the D+1→D+2 forward return. The S&P 500
benchmark uses the same timing and additive $1-per-day PnL convention.

![PnL](strategy.png)

## Metrics

| set | Strategy PnL | S&P 500 PnL | Maximum Drawdown |
|-----|--------------|-------------|------------------|
| train | {results.loc['train', 'PnL']:.4f} | {results.loc['train', 'SP500PnL']:.4f} | {results.loc['train', 'MaxDrawdown']:.4f} |
| test | {results.loc['test', 'PnL']:.4f} | {results.loc['test', 'SP500PnL']:.4f} | {results.loc['test', 'MaxDrawdown']:.4f} |

## Trust assessment
This result is encouraging but is not sufficient to trust the strategy with
real capital. The test set is short, the constituent dataset can contain
survivorship bias, and the backtest omits transaction costs and slippage. A
longer point-in-time universe and a cost-aware walk-forward test are required.
"""

    with open(
        os.path.join(OUT, "report.md"),
        "w",
        encoding="utf-8",
    ) as output:
        output.write(report)

    print(results)

    print(
        "saved strategy.png, results.csv, report.md"
    )