"""OOF ML signal via sklearn cross_val_predict (fold-by-fold under the hood)."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, os.path.dirname(__file__))
from features_engineering import build_dataset, split_train_test
from gridsearch import BlockingTimeSeriesSplit, to_xy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "results", "selected-model")


if __name__ == "__main__":
    train, _ = split_train_test(build_dataset())
    X, y = to_xy(train)
    groups = X.index.get_level_values("date")
    pipe = joblib.load(os.path.join(MODEL_DIR, "selected_model.pkl"))
    cv = BlockingTimeSeriesSplit()

    # train on each fold's train → predict that fold's validation (no one-shot fit)
    proba = cross_val_predict(
        clone(pipe), X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=-1
    )[:, 1]

    val_idx = np.concatenate([va for _, va in cv.split(X, y, groups)])
    signal = pd.Series(proba[val_idx], index=X.index[val_idx], name="ml_signal").sort_index()

    path = os.path.join(MODEL_DIR, "ml_signal.csv")
    signal.to_csv(path)
    print(signal.head(), "\nrows:", len(signal))
    print("saved", path)
