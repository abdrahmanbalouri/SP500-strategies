"""Blocking Time Series CV + sklearn GridSearchCV → selected_model.pkl/.txt"""
import math
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features_engineering import FEATURES, build_dataset, split_train_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_DIR = os.path.join(ROOT, "results", "cross-validation")
MODEL_DIR = os.path.join(ROOT, "results", "selected-model")
N_SPLITS, MIN_TRAIN_DAYS = 10, 730
PARAM_GRID = {"logisticregression__C": [0.1, 1.0, 10.0]}


def by_dates(df, dates):
    return df.loc[df.index.get_level_values("date").isin(dates)]


def to_xy(frame):
    frame = frame.dropna(subset=["target"])
    frame = frame[frame["target"] != 0]
    return frame[FEATURES], (frame["target"] > 0).astype(int)


def make_pipe():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )


def _date_folds(dates, n_splits=N_SPLITS, min_days=MIN_TRAIN_DAYS):
    dates = np.sort(pd.unique(dates))
    days = (dates - dates[0]) / np.timedelta64(1, "D")
    first_end = dates[int(np.argmax(days >= min_days))]
    rest = dates[dates > first_end]
    size = math.ceil(len(rest) / n_splits)
    return [
        (dates[dates < val[0]], val)
        for k in range(n_splits)
        if len(val := rest[k * size : (k + 1) * size])
    ]


class BlockingTimeSeriesSplit(BaseCrossValidator):
    """sklearn CV: expanding train + non-overlapping val blocks, split by date."""

    def __init__(self, n_splits=N_SPLITS, min_train_days=MIN_TRAIN_DAYS):
        self.n_splits = n_splits
        self.min_train_days = min_train_days

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        groups = np.asarray(groups)
        for tr_d, va_d in _date_folds(groups, self.n_splits, self.min_train_days):
            yield np.where(np.isin(groups, tr_d))[0], np.where(np.isin(groups, va_d))[0]


def plot_folds(folds, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (tr, va) in enumerate(folds, 1):
        ax.hlines(i, tr[0], tr[-1], colors="steelblue", lw=14, label="Train" if i == 1 else None)
        ax.hlines(i, va[0], va[-1], colors="darkorange", lw=14, label="Validation" if i == 1 else None)
    ax.set_yticks(range(1, len(folds) + 1), [f"fold {i}" for i in range(1, len(folds) + 1)])
    ax.invert_yaxis()
    ax.set(xlabel="Date", title="Blocking Time Series Split")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close()


if __name__ == "__main__":
    os.makedirs(CV_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train, _ = split_train_test(build_dataset())
    X, y = to_xy(train)
    groups = train.loc[X.index].index.get_level_values("date")
    folds = _date_folds(groups.unique())
    plot_folds(folds, os.path.join(CV_DIR, "blocking_time_series_split.png"))

    cv = BlockingTimeSeriesSplit()
    gs = GridSearchCV(
        make_pipe(),
        PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    gs.fit(X, y, groups=groups)

    best_name = f"logreg_C={gs.best_params_['logisticregression__C']}"
    idx = gs.best_index_
    fold_aucs = [float(gs.cv_results_[f"split{i}_test_score"][idx]) for i in range(gs.n_splits_)]
    summary = pd.DataFrame({best_name: fold_aucs}, index=[f"fold_{i}" for i in range(1, len(fold_aucs) + 1)])
    summary.loc["mean"] = summary.mean()
    print(summary, "\nbest:", best_name, "mean AUC:", gs.best_score_)

    joblib.dump(gs.best_estimator_, os.path.join(MODEL_DIR, "selected_model.pkl"))
    joblib.dump(folds, os.path.join(MODEL_DIR, "folds.pkl"))
    with open(os.path.join(MODEL_DIR, "selected_model.txt"), "w") as f:
        f.write(
            "Methodology\n"
            "- CV: BlockingTimeSeriesSplit (sklearn BaseCrossValidator), 10 folds\n"
            "- Search: sklearn GridSearchCV scoring=roc_auc\n"
            "- First train fold > 2 years; last val before 2017 test\n"
            "- Test set never used for training or model selection\n\n"
            "Pipeline (sklearn make_pipeline)\n"
            "1. SimpleImputer(strategy='median')\n"
            "2. StandardScaler\n"
            "3. Dimension reduction: none\n"
            "4. LogisticRegression(max_iter=1000, random_state=42)\n\n"
            f"Selected: {best_name}\nObject: {gs.best_estimator_}\n"
            f"Mean val AUC: {gs.best_score_:.4f}\n"
            f"Fold AUCs: {[round(a, 4) for a in fold_aucs]}\n"
            f"Grid: {PARAM_GRID}\n"
        )
    print("saved selected_model.pkl / .txt")
