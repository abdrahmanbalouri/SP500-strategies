import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features_engineering import load_historical_data

def run_strategy():
    data_dir = "../data" if os.path.exists("../data") else "data"
    df = load_historical_data(data_dir)
    
    signal_path = "../results/selected-model/ml_signal.csv"
    if not os.path.exists(signal_path):
        return
        
    signals = pd.read_csv(signal_path)
    signals['date'] = pd.to_datetime(signals['date'])
    signals = signals.set_index('date')
    
    df['return'] = df['close'].pct_change().shift(-1)
    merged = df.join(signals, how='inner')
    
    merged['position'] = np.where(merged['ml_signal'] > 0.5, 1.0, 0.0)
    merged['strategy_return'] = merged['position'] * merged['return']
    merged['market_return'] = merged['return']
    
    merged['strategy_pnl'] = (1 + merged['strategy_return'].fillna(0)).cumprod()
    merged['market_pnl'] = (1 + merged['market_return'].fillna(0)).cumprod()
    
    os.makedirs("../results/strategy", exist_ok=True)
    merged[['strategy_pnl', 'market_pnl']].to_csv("../results/strategy/results.csv")
    
    plt.figure(figsize=(10, 5))
    plt.plot(merged.index, merged['strategy_pnl'], label='Strategy PnL', color='blue')
    plt.plot(merged.index, merged['market_pnl'], label='S&P 500 PnL', color='orange')
    plt.axvline(pd.to_datetime('2017-01-01'), color='red', linestyle='--', label='Train/Test Split')
    plt.title('Strategy vs S&P 500 Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.grid(True)
    plt.savefig("../results/strategy/strategy.png")
    plt.close()

if __name__ == "__main__":
    run_strategy()