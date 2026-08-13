import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_strategy():
    data_dir = "../data" if os.path.exists("../data") else "data"
    from features_engineering import load_and_preprocess_data
    df = load_and_preprocess_data(data_dir)
    
    signal_path = "../results/selected-model/ml_signal.csv"
    if not os.path.exists(signal_path):
        raise FileNotFoundError("Run create_signal.py first.")
        
    signals = pd.read_csv(signal_path, parse_dates=['Date'], index_col='Date')
    
    # Merge market data with ML signals
    merged = df.join(signals['Signal'], how='inner')
    
    # Binary long strategy logic: Signal > 0.5 -> Invest ($1), else 0
    merged['position'] = np.where(merged['Signal'] > 0.5, 1, 0)
    
    # Calculate actual return between D+1 and D+2 mapped correctly
    price_d1 = merged['close'].shift(-1)
    price_d2 = merged['close'].shift(-2)
    actual_return = (price_d2 - price_d1) / price_d1
    
    merged['strategy_return'] = merged['position'] * actual_return
    merged['benchmark_return'] = actual_return
    
    # Cumulative Portfolio Values (PnL)
    merged['strategy_pnl'] = (1 + merged['strategy_return'].fillna(0)).cumprod()
    merged['benchmark_pnl'] = (1 + merged['benchmark_return'].fillna(0)).cumprod()
    
    os.makedirs("../results/strategy", exist_ok=True)
    merged[['strategy_pnl', 'benchmark_pnl']].to_csv("../results/strategy/results.csv")
    
    # Plotting Strategy vs Benchmark PnL
    plt.figure(figsize=(12, 6))
    plt.plot(merged.index, merged['strategy_pnl'], label='ML Strategy PnL', color='blue', linewidth=1.5)
    plt.plot(merged.index, merged['benchmark_pnl'], label='S&P 500 Benchmark PnL', color='orange', linewidth=1.5)
    plt.axvline(pd.to_datetime('2017-01-01'), color='red', linestyle='--', label='Train/Test Split (2017)')
    plt.title('Quantitative Strategy vs S&P 500 Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.savefig("../results/strategy/strategy.png")
    plt.close()
    
    print("Backtest strategy executed and strategy.png saved.")

if __name__ == "__main__":
    run_strategy()