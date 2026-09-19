from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from features_engineering import FEATURES, build_dataset, split_train_test
from gridsearch import make_date_folds, make_index_folds, prepare_training_data


ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "results" / "cross-validation"
MODEL_PATH = ROOT / "results" / "selected-model" / "selected_model.pkl"


def calculate_metrics(y_true, probability):
    prediction = (probability >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y_true, probability),
        "Accuracy": accuracy_score(y_true, prediction),
        "LogLoss": log_loss(y_true, probability),
    }


def main():
    train, _ = split_train_test(build_dataset())
    X, y, dates = prepare_training_data(train)
    folds = make_index_folds(dates, make_date_folds(dates))
    selected_model = joblib.load(MODEL_PATH)

    metric_rows = []
    importance_rows = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        model = clone(selected_model)
        model.fit(X.iloc[train_index], y.iloc[train_index])

        for set_name, index in (("train", train_index), ("validation", validation_index)):
            probability = model.predict_proba(X.iloc[index])[:, 1]
            metric_rows.append(
                {
                    "fold": fold_number,
                    "set": set_name,
                    **calculate_metrics(y.iloc[index], probability),
                }
            )

        importance = np.abs(model.named_steps["model"].coef_[0])
        top_features = pd.Series(importance, index=FEATURES).nlargest(10)
        importance_rows.extend(
            {
                "fold": fold_number,
                "feature": feature,
                "importance": value,
            }
            for feature, value in top_features.items()
        )

    metrics = pd.DataFrame(metric_rows).set_index(["fold", "set"])
    metrics.to_csv(CV_DIR / "ml_metrics_train.csv")
    pd.DataFrame(importance_rows).to_csv(
        CV_DIR / "top_10_feature_importance.csv", index=False
    )

    auc = metrics["AUC"].unstack("set")
    ax = auc[["train", "validation"]].plot.bar(
        color=["#e8e8e8", "#84a95b"],
        edgecolor="#666666",
        figsize=(11, 6),
        width=0.75,
    )
    ax.set_title(
        "AUC on train and validation set\non all folds of the train set",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel("Fold", fontsize=13)
    ax.set_ylabel("AUC")
    ax.set_xticklabels(range(1, len(folds) + 1), rotation=0)
    ax.axhline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Random baseline",
    )
    ax.set_ylim(0.48, 0.56)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.figure.tight_layout()
    ax.figure.savefig(CV_DIR / "metric_train.png", dpi=150)
    plt.close(ax.figure)

    print(metrics)
    print("Saved model metrics, feature importance, and AUC plot in", CV_DIR)


if __name__ == "__main__":
    main()
