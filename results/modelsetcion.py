import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from features_engineering import load_historical_data

def select_model():
    data_dir = "../data" if os.path.exists("../data") else "data"
    df = load_historical_data(data_dir)
    
    train_df = df[df.index < '2017-01-01']
    features = ['rsi', 'bollinger_upper', 'bollinger_lower', 'macd']
    X = train_df[features]
    y = np.where(train_df['target'] > 0, 1, 0)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    os.makedirs("../results/selected-model", exist_ok=True)
    joblib.dump(pipeline, "../results/selected-model/selected_model.pkl")
    with open("../results/selected-model/selected_model.txt", "w") as f:
        f.write(str(pipeline.get_params()))

if __name__ == "__main__":
    select_model()