# Strategy report

## Features
Bollinger %B, RSI(14), and MACD, computed independently for each ticker from
prices available through day D.
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
- fold 1: train 508 days (2013-02-08 → 2015-02-13), validation 47 days (2015-02-19 → 2015-04-27)
- fold 2: train 555 days (2013-02-08 → 2015-04-23), validation 47 days (2015-04-28 → 2015-07-02)
- fold 3: train 602 days (2013-02-08 → 2015-06-30), validation 47 days (2015-07-06 → 2015-09-09)
- fold 4: train 649 days (2013-02-08 → 2015-09-04), validation 47 days (2015-09-10 → 2015-11-13)
- fold 5: train 696 days (2013-02-08 → 2015-11-11), validation 47 days (2015-11-16 → 2016-01-25)
- fold 6: train 743 days (2013-02-08 → 2016-01-21), validation 47 days (2016-01-26 → 2016-04-01)
- fold 7: train 790 days (2013-02-08 → 2016-03-30), validation 47 days (2016-04-04 → 2016-06-08)
- fold 8: train 837 days (2013-02-08 → 2016-06-06), validation 47 days (2016-06-09 → 2016-08-15)
- fold 9: train 884 days (2013-02-08 → 2016-08-11), validation 47 days (2016-08-16 → 2016-10-20)
- fold 10: train 931 days (2013-02-08 → 2016-10-18), validation 47 days (2016-10-21 → 2016-12-28)

## Strategy
Long-only (`ml_signal > 0.5`). On each date, $1 is divided equally among all
selected stocks; if none is selected, $0 is invested. PnL = weight × the
D+1→D+2 forward return. The S&P 500 benchmark uses the same forward-return
timing and additive $1-per-day PnL convention.

![PnL](strategy.png)

## Metrics

| set | Strategy PnL | S&P 500 PnL | Excess PnL | Strategy max drawdown |
|-----|--------------|-------------|------------|-----------------------|

the S&P 500 under this common PnL convention.
