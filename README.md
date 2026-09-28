# SP500 Strategies

A machine-learning cross-sectional equity strategy on the S&P 500 universe. It
combines a **Logistic Regression probability signal** with a **fixed
momentum screen**, then backtests the resulting long-only portfolio against the
S&P 500 index.

The project is deliberately built around **leakage-free** time-series
validation: every prediction that enters a performance number comes from a
model that never saw that date during training.

---

## Table of contents

1. [What the project does](#1-what-the-project-does)
2. [Repository layout](#2-repository-layout)
3. [Data](#3-data)
4. [The prediction problem](#4-the-prediction-problem)
5. [Leakage control](#5-leakage-control)
6. [`scripts/features_engineering.py`](#6-scriptsfeatures_engineeringpy)
7. [`scripts/gridsearch.py`](#7-scriptsgridsearchpy)
8. [`scripts/model_selection.py`](#8-scriptsmodel_selectionpy)
9. [`scripts/create_signal.py`](#9-scriptscreate_signalpy)
10. [`scripts/strategy.py`](#10-scriptsstrategypy)
11. [Current results](#11-current-results)
12. [Output inventory](#12-output-inventory)
13. [How to run](#13-how-to-run)
14. [Design decisions and caveats](#14-design-decisions-and-caveats)
15. [Possible improvements](#15-possible-improvements)

---

## 1. What the project does

Five scripts run in order. Each one writes its artefacts to `results/` and is
independent of the others (they recompute what they need from the raw CSV).

```
data/all_stocks_5yr.csv  (505 tickers, OHLCV, 2013-02-08 → 2018-02-07)
            │
            ▼
  features_engineering.py ──► 4 technical features + 2-day forward return
            │
            ▼
  gridsearch.py ──────────► purged expanding-window CV → best pipeline (.pkl)
            │
            ▼
  model_selection.py ─────► per-fold train/validation metrics, feature
            │               importance, AUC chart
            ▼
  create_signal.py ───────► out-of-fold ml_signal for 2015-02-19 → 2016-12-28
            │
            ▼
  strategy.py ─────────────► + test-set signal (2017+)
                              → equal-weight long-only portfolio
                              → PnL, max drawdown, chart, report
```

---

## 2. Repository layout

```
SP500-strategies/
├── data/
│   ├── all_stocks_5yr.csv        # 619,040 rows of stock OHLCV
│   └── HistoricalData.csv        # 2,580 rows of S&P 500 index OHLC
├── scripts/
│   ├── features_engineering.py   # dataset construction
│   ├── gridsearch.py             # CV + hyper-parameter search
│   ├── model_selection.py        # fold-level diagnostics
│   ├── create_signal.py          # out-of-fold predictions
│   └── strategy.py               # portfolio construction + backtest
├── results/
│   ├── cross-validation/         # CV artefacts
│   ├── selected-model/           # fitted pipeline + signal
│   └── strategy/                 # PnL table, chart, report
├── requirements.txt
└── README.md
```

---

## 3. Data

### `data/all_stocks_5yr.csv`

| column | type | meaning |
|--------|------|---------|
| `date` | date | trading day |
| `open` / `high` / `low` / `close` | float | daily OHLC, split/dividend adjusted |
| `volume` | int | share volume |
| `Name` | str | ticker symbol |

- **619,040 rows**, **505 tickers**, **1,259 distinct trading dates**.
- Range: **2013-02-08 → 2018-02-07**.
- Panel structure: a long table, one row per `(ticker, date)`.

### `data/HistoricalData.csv`

Benchmark index history (`Date, Open, High, Low, Close`), **2012-07-31 →
2022-10-28**, covering both the train and test windows. Dates are in
**`%m/%d/%y`** format and the header cells contain spaces, which is why
`strategy.py` passes `date_format=` and `rename(columns=lambda c: c.strip())`.

---

## 4. The prediction problem

For every ticker and every trading day `D`, the model sees only information
available **through the close of `D`** and predicts the direction of the return
from the close of `D+1` to the close of `D+2`.

### Features (`FEATURES`)

| feature | library / formula | window | interpretation |
|---------|-------------------|--------|----------------|
| `bb_pct` | `ta.volatility.BollingerBands(close).bollinger_pband()` | 20 d, 2σ | Bollinger **%B** = `(close − lower) / (upper − lower)`. 0 = at the lower band, 1 = at the upper band, and it may fall outside `[0, 1]`. |
| `rsi` | `ta.momentum.RSIIndicator(close).rsi()` | 14 d | Relative Strength Index, `0–100`. Above 70 conventionally overbought. |
| `macd` | `ta.trend.MACD(close).macd()` | 12/26/9 d | The **MACD line** (fast EMA − slow EMA), *not* the histogram. It is a price-like level, so it is scale-dependent. |
| `momentum_60d` | `close.pct_change(60, fill_method=None)` | 60 d | Raw 60-day return (e.g. `0.18` = +18%), **not** annualised. |

`fill_method=None` is explicit so that pandas does not forward-fill inside a
60-day window across a missing day.

### Target

```python
g["fwd_return"] = c.shift(-2) / c.shift(-1) - 1     # return(D+1 close → D+2 close)
g["target"]     = np.sign(g["fwd_return"])          # +1 / 0 / -1
```

The label is a **three-class** sign. The modelling step converts it to binary by
dropping the `0` class (days where the price did not move at all between the two
closes) — see [`prepare_training_data`](#prepare_training_data).

### Train / test split

`TEST_DATE = 2017-01-01`:

- **train** = 2013-02-08 → 2016-12-30 (used for CV, model selection, and for
  producing the out-of-fold signal)
- **test** = 2017-01-01 → 2018-02-05 (held out; only scored at the very end)

---

## 5. Leakage control

Five independent mechanisms keep information from the future out of the model.

**1 — Purge labels that straddle the train/test boundary.**
In `features_engineering._per_ticker`, any row dated before the cutoff whose
*label* ends on or after the cutoff is blanked:

```python
target_end_date = g["date"].shift(-2)
crosses_test_boundary = (g["date"] < TEST_DATE) & (target_end_date >= TEST_DATE)
g.loc[crosses_test_boundary, ["fwd_return", "target"]] = np.nan
```

Without this, the last two training days would carry a label computed from test
period prices.

**2 — A `gap` equal to the target horizon inside every CV fold.**
`TimeSeriesSplit(..., gap=TARGET_HORIZON)` with `TARGET_HORIZON = 2` drops the
two dates immediately after each training window, so a training label can never
overlap its own validation label.

**3 — Date-level folds, not row-level folds.**
A naive `KFold` on a panel would put ticker A on Monday in train and ticker B on
Monday in validation — the model would effectively see the same day's prices
twice. `gridsearch.make_date_folds` builds the split on *unique dates* and
`make_index_folds` maps it back to rows, so a whole date is either train or
validation.

**4 — No shuffling.**
`TimeSeriesSplit` is inherently chronological; the training window only ever
grows ("expanding window").

**5 — Out-of-fold predictions for the training period.**
`create_signal.py` never fits a model once on everything. For each fold it fits
on that fold's training rows and predicts that fold's validation rows, so each
signal value is produced by a model blind to that date. The same fold structure
is re-derived in `model_selection.py`, `create_signal.py`, and `strategy.py`,
so all three agree on which rows belong where.

---

## 6. `scripts/features_engineering.py`

Builds the modelling table. This is the only file that touches raw prices.

### Constants

```python
ROOT     = <repo root>            # two dirname() calls up from scripts/
DATA     = ROOT/data
TEST_DATE = pd.Timestamp("2017-01-01")
FEATURES = ["bb_pct", "rsi", "macd", "momentum_60d"]
```

`ROOT` is resolved from `__file__` so every path works regardless of the shell's
working directory.

### `_per_ticker(g)` — lines 23-46

Applied to one ticker's rows at a time. This is what makes the indicators
correct: RSI, MACD, and Bollinger bands are all **stateful** (EMA, rolling
quantiles), so they must never be computed on a cross-ticker concatenation.

1. `sort_values("date")` — guarantees the shift/rolling order.
2. Compute `fwd_return` and `target`.
3. Null out boundary-crossing labels (leakage mechanism 1).
4. Compute the four features from `close`.

### `build_dataset()` — lines 49-66

```python
raw = pd.read_csv(...); raw = raw.sort_values(["Name", "date"])
return (pd.concat([_per_ticker(g) for _, g in raw.groupby("Name", sort=False)])
          .set_index(["date", "Name"])        # MultiIndex (date, ticker)
          .rename_axis(["date", "ticker"])
          .loc[:, FEATURES + ["target", "fwd_return"]]
          .sort_index())
```

- `groupby("Name")` → one independent indicator history per ticker.
- The result is a `MultiIndex` on `(date, ticker)`, which makes
  `groupby(level="date")` a natural cross-section later on.
- Only the five needed columns survive; `open/high/low/volume` are dropped.
- `sort_index()` at the end re-establishes date order because the
  `Name`-grouped concatenation produces ticker-major ordering.

Warm-up NaNs are expected and left in place: the first ~20 rows per ticker have
`NaN` RSI/MACD/Bollinger values, and the first 60 lack `momentum_60d`. The
`SimpleImputer` in the pipeline handles them.

### `split_train_test(df)` — lines 69-71

```python
d = df.index.get_level_values("date")
return df.loc[d < TEST_DATE], df.loc[d >= TEST_DATE]
```

A straight date-mask on the index level — no random split anywhere in the
project.

### Running it

`python scripts/features_engineering.py` prints the shape and the date range as
a sanity check. It writes no file, so the `.gitignore` entry for
`data/preprocessed.pkl` is stale (no script produces it).

---

## 7. `scripts/gridsearch.py`

Chooses the hyper-parameter and serialises the winning pipeline.

```python
matplotlib.use("Agg")   # headless backend, set before pyplot import
```

### `make_date_folds(dates, n_splits=10)` — lines 27-42

```python
unique_dates        = sorted unique trading dates
two_year_mark       = unique_dates[0] + pd.DateOffset(years=2)
minimum_train_size  = first index with date > two_year_mark, plus 1
test_size           = (len(unique_dates) - minimum_train_size) // n_splits
splitter            = TimeSeriesSplit(n_splits=10, test_size=test_size, gap=2)
```

- The **2-year burn-in** guarantees fold 1 is already trained on ~2 years of
  history, so the earliest fold is not fit on a handful of days.
- `test_size` is computed by hand so the folds exactly consume the post-burn-in
  window without overlapping the end of the sample.
- No `max_train_size` is passed → the training window is **expanding**.
- `gap=2` is the purge (leakage mechanism 2).
- The return value is `[(train_dates, validation_dates), ...]` — dates, not row
  positions, so the split is defined at the cross-section level.

The resulting geometry is printed into `report.md`:

```
fold 1 : train 508 days (2013-02-08 → 2015-02-13), validation 47 days (2015-02-19 → 2015-04-27)
...
fold 10: train 931 days (2013-02-08 → 2016-10-18), validation 47 days (2016-10-21 → 2016-12-28)
```

### `make_index_folds(dates, date_folds)` — lines 45-53

```python
np.flatnonzero(dates.isin(train_dates)), np.flatnonzero(dates.isin(validation_dates))
```

Turns the date-level plan into row positions so `GridSearchCV` can use it as
`cv=`. Because it selects by date, all ~500 tickers of a date land on the same
side of the split.

### `prepare_training_data(train)` — lines 56-63

```python
usable = train.dropna(subset=["target"])      # drop warm-up / purged rows
usable = usable[usable["target"] != 0]        # drop flat days
X      = usable[FEATURES]
y      = (usable["target"] > 0).astype(int)   # sign -> {0, 1}
dates  = X.index.get_level_values("date")
```

Dropping the `0` class turns a 3-class sign problem into a clean binary
up/down classification and drops the rows where the two closes were identical.
The returned `dates` are reused downstream to rebuild the exact same folds.

### `make_pipeline()` — lines 66-73

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LogisticRegression(max_iter=1000, random_state=42)),
])
```

- **Median imputation** — robust to the indicator warm-up NaNs, unlike
  `mean` which would be dragged by the extreme Bollinger/MACD tails.
- **Standard scaling** — mandatory for regularised logistic regression: the
  penalty in the loss is applied to coefficients, so unscaled features
  (`macd` in the hundreds, `rsi` in 0-100) would be penalised arbitrarily.
- The step is named `model`, which is why the search key is `model__C`.
- `random_state=42` makes `lbfgs` reproducible.

### `plot_folds(folds, output_path)` — lines 76-105

One horizontal bar per fold: blue = training window, orange = validation
window, y-axis inverted so fold 1 is on top. Written to
`results/cross-validation/Time_series_split.png`.

### `save_model_description(search, folds, output_path)` — lines 108-135

Writes a plain-text methodology sheet — CV scheme, purge size, fold count, the
date span of the first training fold and last validation fold, an explicit note
that the test set was not used for selection, the pipeline steps, the parameter
grid, the winning parameters, the mean validation AUC, and the repr of the
fitted pipeline. This is the audit trail for the model choice.

### `main()` — lines 138-158

```python
dataset = build_dataset()
train, _ = split_train_test(dataset)      # test discarded here on purpose
X, y, dates = prepare_training_data(train)
folds      = make_date_folds(dates)
index_folds = make_index_folds(dates, folds)
plot_folds(folds, CV_DIR / "Time_series_split.png")

search = GridSearchCV(
    estimator=make_pipeline(),
    param_grid={"model__C": [0.1, 1.0, 10.0]},
    scoring="roc_auc",
    cv=index_folds,
)
search.fit(X, y)
joblib.dump(search.best_estimator_, MODEL_DIR / "selected_model.pkl")
```

Notes:

- `scoring="roc_auc"` — ranking quality, which is what a thresholded probability
  signal actually needs. Accuracy would be dominated by class imbalance.
- Only `C` (inverse regularisation strength) is searched; the preprocessing
  steps are fixed, which keeps the search space at 3 candidates and the runtime
  short.
- `best_estimator_` is the pipeline **already refit on the whole training set**,
  which is what the test period later uses.
- Current result: `C = 0.1`, mean validation AUC **0.5190**.

---

## 8. `scripts/model_selection.py`

Diagnostics for the *already selected* pipeline. It deliberately does **not**
re-run the search — it re-fits the chosen configuration fold by fold so the
train/validation gap can be inspected.

### `calculate_metrics(y_true, probability)` — lines 22-28

```python
prediction = (probability >= 0.5).astype(int)
{"AUC": roc_auc_score(...), "Accuracy": accuracy_score(...), "LogLoss": log_loss(...)}
```

Log loss is included because accuracy sits at ~0.50 for this signal; log loss
moves even when the hard prediction does not.

### The fold loop — lines 40-63

```python
model = clone(selected_model)                        # unfitted copy, keeps hyperparameters
model.fit(X.iloc[train_index], y.iloc[train_index])  # one model per fold
for set_name, index in (("train", train_index), ("validation", validation_index)):
    probability = model.predict_proba(X.iloc[index])[:, 1]
```

- `clone` is required: fitting the same object repeatedly would leak state
  between folds.
- `[:, 1]` takes the probability of the positive class.
- Both the train and the validation score of the *same* fold model are recorded,
  so `ml_metrics_train.csv` has 2 rows per fold. A large train/validation gap
  would signal overfitting.

### Feature importance — lines 54-63

```python
importance   = np.abs(model.named_steps["model"].coef_[0])
top_features = pd.Series(importance, index=FEATURES).nlargest(10)
```

For a linear model the standardised coefficient magnitude is the importance.
`.nlargest(10)` is a generic cap: the model only has **4** features, so each
fold contributes 4 rows (40 rows total) even though the file is called
`top_10_feature_importance.csv`.

### Output — lines 71-98

`metrics.pivot` → `ml_metrics_train.csv`; a grouped bar chart of AUC per fold
(train in light grey, validation in green) with a red dashed 0.5 random
baseline and a deliberately narrow `ylim(0.48, 0.56)`, because the values sit
right on top of 0.5 and would otherwise be unreadable.

**Observed:** validation AUC ranges 0.505 – 0.546, train AUC is flat at
0.512 – 0.518. The model is weak but honest — train and validation are close,
so there is no meaningful overfitting.

---

## 9. `scripts/create_signal.py`

Produces the **out-of-fold** signal for the training period.

```python
proba = np.concatenate([
    clone(pipe).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
    for tr, va in folds
])
val_idx = np.concatenate([va for _, va in folds])
signal  = pd.Series(proba, index=X.index[val_idx], name="ml_signal").sort_index()
signal.to_csv(MODEL_DIR / "ml_signal.csv")
```

Key points:

- `clone(pipe)` **discards the fitted coefficients inside the pickle**. The
  pipeline is used purely as a hyper-parameter container; each fold's model is
  fit from scratch. (The pickle's own fitted model is only used for the test
  period, in `strategy.py`.)
- `X.index[val_idx]` reattaches the `(date, ticker)` MultiIndex, so the CSV
  keeps its identity and can be joined back to the feature table.
- The validation blocks of a `TimeSeriesSplit` are disjoint, so no date is
  predicted twice and none is dropped: **230,848 rows**, covering
  **2015-02-19 → 2016-12-28**.
- Consequence: the first ~2 years of the training set have no signal, so the
  backtest PnL starts at **2015-02-19**, not at the start of the data.

CSV shape: `date,ticker,ml_signal`.

---

## 10. `scripts/strategy.py`

Turns signals into a portfolio and writes the report.

```python
MOMENTUM_QUANTILE = 0.90     # fixed in advance, not tuned on the test set
```

### `to_weights(signal, momentum)` — lines 24-41

```python
selected_data = pd.concat([signal.rename("signal"), momentum.rename("momentum")], axis=1).dropna()

cutoff = selected_data["momentum"].groupby(level="date").transform(
    lambda v: v.quantile(0.90))

long = ((selected_data["signal"] > 0.5) &
        (selected_data["momentum"] >= cutoff)).astype(float)

return long / long.groupby(level="date").transform("sum").replace(0, np.nan)
```

Step by step:

1. **Align and clean** — an inner join on `(date, ticker)` followed by
   `dropna()`; a name survives only if it has both a signal and a momentum value.
2. **Cross-sectional momentum cutoff** — for each date, the 90th percentile of
   `momentum_60d`. Note it is computed over *all* rows that have a signal, not
   just the ones the model likes, so the screen is a fixed daily rule rather
   than a signal-dependent threshold.
3. **Two filters** — the model must say "up" (`signal > 0.5`) **and** the name
   must be in the top decile of 60-day momentum.
4. **Equal weight** — divide by the number of selected names on that date, so
   exactly **$1 is invested per day**. `replace(0, np.nan)` means a day with no
   qualifying name produces `NaN` weights (fully in cash) instead of dividing
   by zero. In the current run every one of the 745 signal dates has at least
   one qualifying name, so this branch never actually fires.
5. A typical day holds **~46 names** (min 26, max 51) out of ~500
   constituents — the top decile screen intersected with the model gate.

### `sp500_forward_returns(dates)` — lines 44-58

```python
close         = read HistoricalData.csv (format "%m/%d/%y", strip header spaces)["Close"]
forward_return = close.shift(-2) / close.shift(-1) - 1
return forward_return.reindex(dates).fillna(0)
```

The benchmark is measured on **exactly the same timing convention** as the
strategy (D+1 close → D+2 close) and reindexed to the strategy's own dates, so
the two cumulative curves are directly comparable.

### `max_drawdown(cum_r)` — lines 61-62

```python
float((cum_r.cummax() - cum_r).max())
```

The largest peak-to-trough drop of the cumulative PnL curve. Because the PnL is
an **additive $1/day** series rather than a compounded return, this is a
maximum drawdown in dollars of cumulative profit, not a percentage drawdown.

### `fold_lengths(folds)` — lines 65-72

Renders the fold geometry as a Markdown bullet list injected into `report.md`.

### `main()` — lines 75-279

```python
data             = build_dataset()
train, test      = split_train_test(data)
dates            = prepare_training_data(train)[2]     # rebuild the fold dates
folds            = make_date_folds(dates)
pipe             = joblib.load("selected_model.pkl")

oof     = pd.read_csv("ml_signal.csv", parse_dates=["date"]) \
              .set_index(["date","ticker"])["ml_signal"]        # 2015-02-19 → 2016-12-28

Xte     = test.dropna(subset=["fwd_return"])[FEATURES]         # 2017-01-03 → 2018-02-05
test_sig = pd.Series(pipe.predict_proba(Xte)[:, 1], index=Xte.index, name="ml_signal")

signal  = pd.concat([oof, test_sig]).sort_index()               # full 2015-2018 signal
strategy = to_weights(signal, data["momentum_60d"])
```

Then the performance engine:

```python
aligned = data[["fwd_return"]].join(strategy, how="inner").dropna()
pnl     = (aligned["weight"] * aligned["fwd_return"]).groupby(level="date").sum().sort_index()
cum     = pnl.cumsum()

benchmark_pnl = sp500_forward_returns(pnl.index)               # same dates
benchmark_cum = benchmark_pnl.cumsum()
```

- `join(how="inner")` restricts to rows that have both a weight and a known
  forward return.
- `dropna()` also removes the fully-cash dates (the `NaN`-weight branch of
  `to_weights`).
- The product `weight × forward_return` is the dollar PnL of each name;
  summing per date gives the daily portfolio PnL of a $1/day book.
- The benchmark is reindexed to `pnl.index`, so it covers only the days the
  strategy was actually invested.

### Metrics — lines 130-155

```python
def metrics(mask):
    r = pnl[mask]; benchmark = benchmark_pnl[mask]
    return {"PnL": r.sum(),
            "SP500PnL": benchmark.sum(),
            "MaxDrawdown": max_drawdown(r.cumsum())}

results = pd.DataFrame({"train": metrics(cum.index < TEST_DATE),
                        "test":  metrics(cum.index >= TEST_DATE)}).T
```

The same two statistics are reported for both windows and written to
`results/strategy/results.csv`. Drawdown is reported for the strategy only.

### Plot — lines 162-217

Twin-axis chart: strategy cumulative PnL in red on the left axis, S&P 500
cumulative PnL on the right. Both axes are forced to the same symmetric range
(`±max(|series|)`) — otherwise the twin axes would visually mislead. A dashed
vertical line marks 2017-01-01. Saved as `results/strategy/strategy.png`.

### Report — lines 224-273

An f-string writes `results/strategy/report.md` containing the feature
description, the pipeline, the fold table, the strategy rules (including the
explicit statement that the 90th-percentile threshold was fixed in advance), the
metrics table, the chart, and a **trust assessment** paragraph that names the
survivorship-bias, short-test-set, and zero-transaction-cost limitations.

---

## 11. Current results

From `results/strategy/results.csv` (additive $1/day PnL):

| set | window | Strategy PnL | S&P 500 PnL | Max drawdown |
|-----|--------|-------------:|-------------:|-------------:|
| train | 2015-02-19 → 2016-12-28 | **0.1631** | 0.0782 | 0.1937 |
| test  | 2017-01-03 → 2018-02-05 | **0.1888** | 0.1702 | 0.1026 |

Total cumulative PnL across the full 745 invested days: **0.3519**.

Reading these numbers honestly:

- The test window beats the index (0.1888 vs 0.1702), but by a margin small
  enough to be inside the noise of a ~1-year sample.
- The model itself is close to a coin flip: mean validation AUC **0.519**,
  per-fold validation AUC between 0.505 and 0.546.
- **98% of the out-of-fold signals are above 0.5**, so the `signal > 0.5` gate
  rejects almost nothing. Essentially all of the outperformance comes from the
  60-day momentum screen, not from the machine learning.
- The test window is only 275 trading days long. One good year is not evidence
  of an edge.

---

## 12. Output inventory

| file | written by | contents |
|------|-----------|----------|
| `results/cross-validation/Time_series_split.png` | `gridsearch.py` | fold geometry diagram |
| `results/cross-validation/ml_metrics_train.csv` | `model_selection.py` | AUC / Accuracy / LogLoss per fold × set |
| `results/cross-validation/top_10_feature_importance.csv` | `model_selection.py` | 4 rows per fold, `|coef|` |
| `results/cross-validation/metric_train.png` | `model_selection.py` | grouped AUC bars with 0.5 baseline |
| `results/selected-model/selected_model.pkl` | `gridsearch.py` | fitted `Pipeline` (joblib) |
| `results/selected-model/selected_model.txt` | `gridsearch.py` | methodology / audit sheet |
| `results/selected-model/ml_signal.csv` | `create_signal.py` | 230,848 out-of-fold probabilities |
| `results/strategy/results.csv` | `strategy.py` | PnL, benchmark, drawdown per window |
| `results/strategy/strategy.png` | `strategy.py` | cumulative PnL vs index |
| `results/strategy/report.md` | `strategy.py` | the written report |

`features_engineering.py` writes nothing; the dataset is rebuilt in-memory by
each downstream script (≈5 s each, dominated by the per-ticker indicator loop
over 505 tickers).

---

## 13. How to run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/features_engineering.py   # sanity check on the dataset
python scripts/gridsearch.py             # writes selected_model.pkl / .txt + fold plot
python scripts/model_selection.py        # writes the CV metrics + charts
python scripts/create_signal.py          # writes ml_signal.csv
python scripts/strategy.py               # writes results.csv / strategy.png / report.md
```

The scripts import each other by bare module name (`from features_engineering
import ...`), which works because Python adds the script's own directory to
`sys.path` when it is run as `python scripts/<file>.py`. `create_signal.py` and
`strategy.py` additionally do `sys.path.insert(0, os.path.dirname(__file__))`
explicitly so they also work under `python -m`.

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `ta`.

---

## 14. Design decisions and caveats

### Deliberate choices

- **Two-day horizon** matches a practical rebalance cadence and keeps the purge
  trivial to reason about (`gap = 2`).
- **Expanding window** rather than a rolling window: more training data early,
  and it matches how the model would actually be deployed.
- **No dimension reduction** — with 4 features PCA/selection would only add
  variance. The step is documented in the audit sheet as a deliberate omission.
- **Median imputation + standard scaling inside the `Pipeline`** so both are
  fit on training folds only; no preprocessing is fitted on the full dataset.
- **The 90% momentum quantile is fixed in advance** and is *not* a tuned
  parameter. This is stated in the report so the result cannot be read as
  parameter search on the test set.
- **Signals are probabilities, not 0/1 labels.** The portfolio threshold
  `> 0.5` keeps the ranking information around for diagnostics.

### Known weaknesses

1. **Survivorship bias.** The 505 tickers are today's constituents; delisted and
   merged names are absent. Real performance would be worse.
2. **No transaction costs, slippage, or spreads.** The strategy rebalances its
   entire book every single day, which is the single most unrealistic part of
   the backtest.
3. **The ML gate is nearly a no-op.** With 98% of probabilities above 0.5, the
   logistic regression is contributing little beyond a tilt. A more honest
   presentation would either calibrate a quantile-based top-K selection or drop
   the model.
4. **`momentum_60d` is not annualised** while `macd` and `bb_pct` are
   price-scale quantities. For a cross-sectional percentile screen this is
   harmless, but as a model input it mixes units — `StandardScaler` removes the
   global mean/scale but not the heavy right tail of raw returns.
5. **The benchmark is sampled on the strategy's invested days only.** If the
   strategy sits in cash on a day, the index is not charged for that day either.
   That comparison is generous to the benchmark in general and definitely not
   like a passive buy-and-hold.
6. **PnL is additive dollars, not a return series.** `$1/day` summed over 745
   days is not a portfolio return, and `MaxDrawdown` is a dollar drawdown, not a
   percentage one. There is no Sharpe, volatility, or turnover figure anywhere.
7. **Only the first fold is a true 2-year burn-in; the last training fold sees
   931 days.** Fold-to-fold AUC differences are not a like-for-like comparison.
8. **The test period is 275 trading days.** That is far too short to distinguish
   skill from luck.
9. `top_10_feature_importance.csv` contains 4 features per fold, not 10 — the
   `nlargest(10)` cap is never binding.
10. The `.gitignore` entry for `data/preprocessed.pkl` is stale; no script
    writes that file.

---

## 15. Possible improvements

- **Add costs**: subtract a bps round-trip × turnover per day, and report the
  strategy's break-even cost.
- **Vectorised backtest** with position carry-over (hold until rebalance) rather
  than a full daily turnover, plus a proper compounding equity curve, Sharpe,
  and turnover ratio.
- **Point-in-time universe** with delisted names, or an index-membership
  history, to remove survivorship bias.
- **Better signal**: cross-sectional z-scores/rankings per date, sector
  neutralisation, and a gradient-boosted or ensemble model to see whether the
  AUC is genuinely limited by the four features or by the model class.
- **Calibrated selection**: pick the top-K names by predicted probability instead
  of a fixed 0.5 threshold, and sweep K only on training folds.
- **Walk-forward evaluation** on the test window itself (refit every quarter)
  to test stability rather than a single 2017-2018 block.
- **Long/short variant** to check whether the model's information is merely
  directional or genuinely asymmetric.
