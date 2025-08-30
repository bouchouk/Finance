# ---------- utils/data_processor.py ----------
import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'configs', 'env_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def preprocess_data(df):
    """Enhanced feature engineering without time features"""
    # Ensure datetime index
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Core features
    df['return'] = df['close'].pct_change().fillna(0)
    df['ema'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Relative distances with clipping
    for level in ['strike_high', 'strike_low', 'opt_strike_high', 'opt_strike_low']: 
        if level in df.columns:
            dist = (df[level] - df['close']) / df['close']
            df[f'{level}_dist'] = np.clip(dist, -0.1, 0.1)  # Clip extreme moves
            
    df['ema_dist'] = (df['ema'] - df['close']) / df['close']
    
    # Volatility features
    df['atr'] = df['high'] - df['low']
    df['volatility'] = df['atr'].rolling(5).std().fillna(0)
    
    # Volume processing
    if 'volume' in df.columns:
        scaler = RobustScaler(quantile_range=(5, 95))
        df['volume'] = scaler.fit_transform(df[['volume']])
    
    # Feature selection (no time features)
    features = config['data']['features']
    df = df[features + ['date_time', 'close']].copy()
    
    # Final cleaning
    return df.dropna()

def create_sequences(data, window_size):
    """Create temporal sequences with shuffling"""
    sequences = []
    close_prices = []
    
    for i in tqdm(range(window_size, len(data)), desc="Creating sequences"):
        seq = data[i-window_size:i, :-1]  # Exclude close from features
        sequences.append(seq)
        close_prices.append(data[i-1, -1])  # Last close in sequence
    
    sequences = np.array(sequences)
    close_prices = np.array(close_prices)
    
    # Shuffle to prevent temporal bias
    # indices = np.arange(len(sequences))
    # np.random.shuffle(indices)
    return sequences, close_prices

def train_test_split(sequences, close_prices, timestamps, test_size=0.2):
    """Time-based train-test split to avoid look-ahead bias"""
    # Sort by timestamp to maintain chronological order
    # sorted_indices = np.argsort(timestamps)
    # sorted_sequences = sequences[sorted_indices]
    # sorted_close_prices = close_prices[sorted_indices]
    # sorted_timestamps = timestamps[sorted_indices]
    
    # Split based on time
    split_idx = int(len(sequences) * (1 - test_size))
    
    train_sequences = sequences[:split_idx]
    train_close_prices = close_prices[:split_idx]
    train_timestamps = timestamps[:split_idx]
    
    test_sequences = sequences[split_idx:]
    test_close_prices = close_prices[split_idx:]
    test_timestamps = timestamps[split_idx:]
    
    return (train_sequences, train_close_prices, train_timestamps), \
           (test_sequences, test_close_prices, test_timestamps)