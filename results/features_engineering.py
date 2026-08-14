import os
import pandas as pd
import numpy as np

def load_historical_data(data_path):
    file_path = os.path.join(data_path, 'HistoricalData.csv')
        
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).set_index(date_col)
    
    price_col = 'close/last' if 'close/last' in df.columns else ('close' if 'close' in df.columns else df.columns[1])
    df['close'] = df[price_col].astype(str).str.replace('$', '').astype(float)
    
    return compute_features_and_target(df)

def load_all_stocks_data(data_path):
    file_path = os.path.join(data_path, 'all_stocks_5yr.csv')
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'name']).set_index('date')
    df['close'] = df['close'].astype(float)
    
    processed_dfs = []
    for name, group in df.groupby('name'):
        group_feat = compute_features_and_target(group.copy())
        group_feat['name'] = name
        processed_dfs.append(group_feat)
        
    return pd.concat(processed_dfs)

def compute_features_and_target(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = ma + (std * 2)
    df['bollinger_lower'] = ma - (std * 2)
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    
    price_d1 = df['close'].shift(-1)
    price_d2 = df['close'].shift(-2)
    ret_d1_d2 = (price_d2 - price_d1) / price_d1
    df['target'] = np.sign(ret_d1_d2)
    
    df = df.dropna()
    return df