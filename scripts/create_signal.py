import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit

def generate_signals():
    data_dir = "../data" if os.path.exists("../data") else "data"
    from features_engineering import load_and_preprocess_data
    df = load_and_preprocess_data(data_dir)
    
    features = ['rsi', 'bollinger_upper', 'bollinger_lower', 'macd']
    X = df[features]
    
    tscv = TimeSeriesSplit(n_splits=10)
    model_path = "../results/selected-model/selected_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Run gridsearch.py first to create the model.")
        
    model = joblib.load(model_path)
    signal_records = []
    
    for train_index, val_index in tscv.split(X):
        X_val = X.iloc[val_index]
        val_dates = df.index[val_index]
        
        probs = model.predict_proba(X_val)[:, 1]
        
        for date, prob in zip(val_dates, probs):
            signal_records.append({
                'Date': date,
                'Ticker': 'SP500',
                'Signal': prob
            })
            
    signal_df = pd.DataFrame(signal_records)
    os.makedirs("../results/selected-model", exist_ok=True)
    signal_df.to_csv("../results/selected-model/ml_signal.csv", index=False)
    print("ml_signal.csv generated successfully.")

if __name__ == "__main__":
    generate_signals()