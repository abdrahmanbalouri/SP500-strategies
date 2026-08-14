import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from features_engineering import load_historical_data

def generate_signals():
    data_dir = "../data" if os.path.exists("../data") else "data"
    df = load_historical_data(data_dir)
    
    features = ['rsi', 'bollinger_upper', 'bollinger_lower', 'macd']
    X = df[features]
    y = np.where(df['target'] > 0, 1, 0)
    
    tscv = TimeSeriesSplit(n_splits=10)
    model_path = "../results/selected-model/selected_model.pkl"
    
    if not os.path.exists(model_path):
        return
        
    pipeline = joblib.load(model_path)
    signal_records = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y[train_idx]
        
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_val)[:, 1]
        
        val_dates = X.index[val_idx]
        for date, prob in zip(val_dates, probs):
            signal_records.append({'date': date, 'ticker': 'SP500', 'ml_signal': prob})
            
    signal_df = pd.DataFrame(signal_records)
    os.makedirs("../results/selected-model", exist_ok=True)
    signal_df.to_csv("../results/selected-model/ml_signal.csv", index=False)

if __name__ == "__main__":
    generate_signals()