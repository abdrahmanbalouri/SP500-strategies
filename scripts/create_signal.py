import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(__file__))
from features_engineering import build_dataset, split_train_test
from gridsearch import make_date_folds, make_index_folds, prepare_training_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "selected-model")


if __name__ == "__main__":
    train, _ = split_train_test(build_dataset())
    X, y, dates = prepare_training_data(train)
    pipe = joblib.load(os.path.join(MODEL_DIR, "selected_model.pkl"))
    folds = make_index_folds(dates, make_date_folds(dates))

    # train on each fold's train → predict that fold's validation (no one-shot fit)
    proba = np.concatenate(
        [
            clone(pipe).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
            for tr, va in folds
        ]
    )

    val_idx = np.concatenate([va for _, va in folds])
    signal = pd.Series(proba, index=X.index[val_idx], name="ml_signal").sort_index()

    path = os.path.join(MODEL_DIR, "ml_signal.csv")
    signal.to_csv(path)
    print(signal.head(), "\nrows:", len(signal))
    print("saved", path)
