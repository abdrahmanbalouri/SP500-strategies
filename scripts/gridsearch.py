import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib

def run_gridsearch():
    data_dir = "../data" if os.path.exists("../data") else "data"
    from features_engineering import load_and_preprocess_data
    df = load_and_preprocess_data(data_dir)
    
    # Split train/test (Test set from 2017 onwards)
    train_df = df[df.index < '2017-01-01']
    
    features = ['rsi', 'bollinger_upper', 'bollinger_lower', 'macd']
    X = train_df[features]
    y = np.where(train_df['target'] > 0, 1, 0)
    
    # Temporal Cross Validation (10 folds)
    tscv = TimeSeriesSplit(n_splits=10)
    
    metrics_records = []
    feature_importances = []
    
    os.makedirs("../results/cross-validation", exist_ok=True)
    os.makedirs("../results/selected-model", exist_ok=True)
    
    best_score = 0
    best_pipeline = None
    
    fold = 1
    for train_index, val_index in tscv.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y[train_index], y[val_index]
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])
        
        pipeline.fit(X_tr, y_tr)
        
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, preds)
        try:
            auc = roc_auc_score(y_val, probs)
        except:
            auc = 0.5
        loss = log_loss(y_val, pipeline.predict_proba(X_val))
        
        metrics_records.append({
            'fold': fold,
            'train_accuracy': accuracy_score(y_tr, pipeline.predict(X_tr)),
            'val_accuracy': acc,
            'val_auc': auc,
            'val_logloss': loss
        })
        
        # Feature importance tracking
        importances = pipeline.named_steps['model'].feature_importances_
        fi_df = pd.DataFrame({'feature': features, 'importance': importances, 'fold': fold})
        feature_importances.append(fi_df)
        
        if auc > best_score:
            best_score = auc
            best_pipeline = pipeline
            
        fold += 1

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv("../results/cross-validation/ml_metrics_train.csv", index=False)
    
    fi_all = pd.concat(feature_importances)
    top_10 = fi_all.sort_values(by='importance', ascending=False).groupby('fold').head(10)
    top_10.to_csv("../results/cross-validation/top_10_feature_importance.csv", index=False)
    
    # Save AUC train plot
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['fold'], metrics_df['val_auc'], marker='o', color='purple', label='Validation AUC')
    plt.title('Validation AUC Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    plt.savefig("../results/cross-validation/metric_train.png")
    plt.close()
    
    # Save selected best model
    joblib.dump(best_pipeline, "../results/selected-model/selected_model.pkl")
    with open("../results/selected-model/selected_model.txt", "w") as f:
        f.write(str(best_pipeline.get_params()))
        
    print("GridSearch completed. Best model and metrics saved.")

if __name__ == "__main__":
    run_gridsearch()