"""ML metrics + feature importance via sklearn cross_validate."""
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

sys.path.insert(0, os.path.dirname(__file__))
from features_engineering import FEATURES, build_dataset, split_train_test
from gridsearch import BlockingTimeSeriesSplit, to_xy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_DIR = os.path.join(ROOT, "results", "cross-validation")
MODEL_DIR = os.path.join(ROOT, "results", "selected-model")

SCORING = {"AUC": "roc_auc", "Accuracy": "accuracy", "LogLoss": "neg_log_loss"}


if __name__ == "__main__":
    os.makedirs(CV_DIR, exist_ok=True)
    train, _ = split_train_test(build_dataset())
    X, y = to_xy(train)
    groups = X.index.get_level_values("date")
    pipe = joblib.load(os.path.join(MODEL_DIR, "selected_model.pkl"))

    out = cross_validate(
        clone(pipe),
        X,
        y,
        cv=BlockingTimeSeriesSplit(),
        groups=groups,
        scoring=SCORING,
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1,
    )

    rows, tops, auc = [], [], {"train": [], "validation": []}
    n = len(out["estimator"])
    for i in range(n):
        fold = i + 1
        train_m = {
            "AUC": out["train_AUC"][i],
            "Accuracy": out["train_Accuracy"][i],
            "LogLoss": -out["train_LogLoss"][i],
        }
        val_m = {
            "AUC": out["test_AUC"][i],
            "Accuracy": out["test_Accuracy"][i],
            "LogLoss": -out["test_LogLoss"][i],
        }
        rows.append({"fold": fold, "set": "train", **train_m})
        rows.append({"fold": fold, "set": "validation", **val_m})
        auc["train"].append(train_m["AUC"])
        auc["validation"].append(val_m["AUC"])

        top = (
            pd.Series(np.abs(out["estimator"][i][-1].coef_.ravel()), index=FEATURES)
            .nlargest(10)
            .rename_axis("feature")
            .reset_index(name="importance")
        )
        top.insert(0, "fold", fold)
        tops.append(top)

    ml = pd.DataFrame(rows).set_index(["fold", "set"])
    extra = [
        {"fold": how, "set": s, **ml.xs(s, level="set").agg(how).to_dict()}
        for how in ("mean", "median")
        for s in ("train", "validation")
    ]
    pd.concat([ml.reset_index(), pd.DataFrame(extra)]).to_csv(
        os.path.join(CV_DIR, "ml_metrics_train.csv"), index=False
    )
    pd.concat(tops).to_csv(os.path.join(CV_DIR, "top_10_feature_importance.csv"), index=False)

    ax = pd.DataFrame(auc, index=range(1, n + 1)).plot(
        marker="o", figsize=(9, 5), title="AUC per fold (train set)"
    )
    ax.set(xlabel="Fold", ylabel="AUC")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(CV_DIR, "metric_train.png"), dpi=140)
    plt.close()
    print(ml)
    print("saved ml_metrics_train.csv, top_10_feature_importance.csv, metric_train.png")
