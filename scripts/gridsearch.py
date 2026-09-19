from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features_engineering import FEATURES, build_dataset, split_train_test


ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "results" / "cross-validation"
MODEL_DIR = ROOT / "results" / "selected-model"

N_SPLITS = 10
TARGET_HORIZON = 2


def make_date_folds(dates, n_splits=N_SPLITS):
    unique_dates = pd.DatetimeIndex(pd.unique(dates)).sort_values()
    two_year_mark = unique_dates[0] + pd.DateOffset(years=2)
    after_two_years = np.flatnonzero(unique_dates > two_year_mark)
    minimum_train_size = after_two_years[0] + 1
    remaining_dates = len(unique_dates) - minimum_train_size
    test_size = remaining_dates // n_splits
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        gap=TARGET_HORIZON,
    )
    return [
        (unique_dates[train_indices], unique_dates[validation_indices])
        for train_indices, validation_indices in splitter.split(unique_dates)
    ]


def make_index_folds(dates, date_folds):
    dates = pd.DatetimeIndex(dates)
    return [
        (
            np.flatnonzero(dates.isin(train_dates)),
            np.flatnonzero(dates.isin(validation_dates)),
        )
        for train_dates, validation_dates in date_folds
    ]


def prepare_training_data(train):
    usable = train.dropna(subset=["target"])
    usable = usable[usable["target"] != 0]

    X = usable[FEATURES]
    y = (usable["target"] > 0).astype(int)
    dates = X.index.get_level_values("date")
    return X, y, dates


def make_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def plot_folds(folds, output_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    for fold_number, (train_dates, validation_dates) in enumerate(folds, start=1):
        ax.plot(
            [train_dates[0], train_dates[-1]],
            [fold_number, fold_number],
            linewidth=8,
            color="steelblue",
            solid_capstyle="butt",
            label="Train" if fold_number == 1 else None,
        )
        ax.plot(
            [validation_dates[0], validation_dates[-1]],
            [fold_number, fold_number],
            linewidth=8,
            color="darkorange",
            solid_capstyle="butt",
            label="Validation" if fold_number == 1 else None,
        )

    ax.set_title("Expanding Time Series Cross-Validation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fold")
    ax.set_yticks(range(1, len(folds) + 1))
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    # plt.close(fig)


def save_model_description(search, folds, output_path):
    first_train = folds[0][0]
    last_validation = folds[-1][1]

    lines = [
        "Selected model methodology",
        "==========================",
        "",
        "Cross-validation: expanding time-series split by date",
        f"Purged gap: {TARGET_HORIZON} trading dates (target horizon)",
        f"Number of folds: {len(folds)}",
        f"First training fold: {first_train[0].date()} to {first_train[-1].date()}",
        f"Last validation fold: {last_validation[0].date()} to {last_validation[-1].date()}",
        "Test data from 2017 onward was not used for model selection.",
        "",
        "Pipeline:",
        "1. Median imputation",
        "2. Standard scaling",
        "3. No dimension reduction",
        "4. Logistic regression",
        "",
        f"Parameter grid: {{'model__C': [0.1, 1.0, 10.0]}}",
        f"Best parameters: {search.best_params_}",
        f"Best mean validation AUC: {search.best_score_:.6f}",
        "",
        f"Selected pipeline: {search.best_estimator_}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataset = build_dataset()
    train, _ = split_train_test(dataset)
    X, y, dates = prepare_training_data(train)

    folds = make_date_folds(dates)
    index_folds = make_index_folds(dates, folds)
    plot_folds(folds, CV_DIR / "Time_series_split.png")

    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=index_folds,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)

    joblib.dump(search.best_estimator_, MODEL_DIR / "selected_model.pkl")
    save_model_description(search, folds, MODEL_DIR / "selected_model.txt")

    print("Best parameters:", search.best_params_)


if __name__ == "__main__":
    main()
