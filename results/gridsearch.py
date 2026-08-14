import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from features_engineering import load_historical_data

def run_gridsearch():
    data_dir = "../data" if os.path.exists("../data") else "data"
    df = load_historical_data(data_dir)
    
    train_df = df[df.index < '2017-01-01']
    features = ['rsi', 'bollinger_upper', 'bollinger_lower', 'macd']
    X = train_df[features]
    y = np.where(train_df['target'] > 0, 1, 0)
    
    tscv = TimeSeriesSplit(n_splits=10)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    metrics_list = []
    importances_list = []
    auc_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, preds)
        loss = log_loss(y_val, pipeline.predict_proba(X_val))
        try:
            auc = roc_auc_score(y_val, probs)
        except:
            auc = 0.5
            
        auc_per_fold.append(auc)
        metrics_list.append({'fold': fold, 'accuracy': acc, 'auc': auc, 'log_loss': loss})
        
        importances = pipeline.named_steps['model'].feature_importances_
        for f, imp in zip(features, importances):
            importances_list.append({'fold': fold, 'feature': f, 'importance': imp})
            
    os.makedirs("../results/cross-validation", exist_ok=True)
    os.makedirs("../results/selected-model", exist_ok=True)
    
    pd.DataFrame(metrics_list).to_csv("../results/cross-validation/ml_metrics_train.csv", index=False)
    pd.DataFrame(importances_list).to_csv("../results/cross-validation/top_10_feature_importance.csv", index=False)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(auc_per_fold)), auc_per_fold, marker='o', color='purple', label='AUC per Fold')
    plt.title('Validation AUC Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    plt.savefig("../results/cross-validation/metric_train.png")
    plt.close()

if __name__ == "__main__":
    run_gridsearch()