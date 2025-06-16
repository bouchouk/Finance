import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from envs.trading_env import TradingEnv
import time
import os
import matplotlib.pyplot as plt

# Custom callback for tracking training progress
class TrainingTrackerCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.training_start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.trade_counts = []
        
    def _on_training_start(self):
        self.training_start_time = time.time()
        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Get current environment info
            env = self.training_env.envs[0]
            elapsed = time.time() - self.training_start_time
            fps = int(self.n_calls / elapsed) if elapsed > 0 else 0
            
            # Print training progress
            print(f"\nStep: {self.num_timesteps:,} | "
                  f"FPS: {fps} | "
                  f"Elapsed: {self._format_time(elapsed)}")
                  
            # Print GPU memory usage if available
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated(0) / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU Mem: {mem_used:.2f}/{mem_total:.2f} GB")
        
        return True
        
    def _on_rollout_end(self):
        # Collect episode stats
        for env in self.training_env.envs:
            if hasattr(env, 'get_episode_rewards'):
                self.episode_rewards.append(env.get_episode_rewards())
                self.episode_lengths.append(env.get_episode_lengths())
        
        # Print summary every 10 rollouts
        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % 10 == 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            mean_length = np.mean(self.episode_lengths[-10:])
            print(f"Rollout {len(self.episode_rewards)} | "
                  f"Avg Reward: {mean_reward:,.2f} | "
                  f"Avg Length: {mean_length:,.0f} steps")
    
    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def compute_metrics(equity_curve: np.ndarray):
    # daily returns
    returns = equity_curve[1:] / equity_curve[:-1] - 1
    # Sharpe (annualized)
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    # max drawdown
    cum = np.maximum.accumulate(equity_curve)
    drawdown = (cum - equity_curve) / cum
    max_dd = np.max(drawdown)
    total_return = equity_curve[-1] / equity_curve[0] - 1
    return total_return, sharpe, max_dd, returns

def plot_equity_curve(equity, returns, title="Equity Curve"):
    plt.figure(figsize=(15, 10))
    
    # Equity curve
    plt.subplot(2, 1, 1)
    plt.plot(equity, label='Net Worth')
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.legend()
    
    # Daily returns
    plt.subplot(2, 1, 2)
    plt.bar(range(len(returns)), returns, color=np.where(returns >= 0, 'g', 'r'))
    plt.title('Daily Returns')
    plt.xlabel('Day')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.close()

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # GPU detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*50)
    print(f"Initializing Trading Agent - Using {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("="*50 + "\n")
    
    # USER SETTINGS
    csv_path = 'data/Final_model_data_SPY(2).csv'
    feature_cols = ['Open','High','Low','close','opt_high','opt_low','gamma_high','gamma_low','gamma_low_gex','gamma_high_gex','diff_high','diff_low']
    risk_pct = 0.01
    window_size = 6
    total_timesteps = 100_000

    # 1) Load data
    print("Loading data...")
    df = pd.read_csv(csv_path).reset_index(drop=True)
    print(f"Data loaded: {len(df)} rows, {len(feature_cols)} features")
    
    # 2) Train/test split and normalize
    split = int(len(df) * 0.8)
    train_df, test_df = df[:split], df[split:]
    print(f"Train period: {train_df.index[0]} to {train_df.index[-1]} | "
          f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    
    # Normalize using training data only
    for c in feature_cols:
        mn = train_df[c].min()
        mx = train_df[c].max()
        train_df[c] = (train_df[c] - mn) / (mx - mn + 1e-8)
        test_df[c] = (test_df[c] - mn) / (mx - mn + 1e-8)
    print("Data normalized using training set stats")

    # 3) Create environment factory
    def make_env(data_df):
        return TradingEnv(
            data_df,
            feature_cols,
            window_size=window_size,
            initial_balance=100_000,
            transaction_cost_pct=0.00,
            risk_pct=risk_pct,
            shares_per_lot=100,
            max_stop_pct=0.5,
            n_stop_bins=10
        )

    # 4) Create training environment
    print("\nCreating environments...")
    train_env = DummyVecEnv([lambda: make_env(train_df)])
    
    # 5) Initialize DQN model with GPU support
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[256, 256]  # Larger network for better learning
    )
    
    model = DQN(
        'MlpPolicy', 
        train_env,
        device=device,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=1024 if device == "cuda" else 32,  # Larger batches for GPU
        train_freq=4,
        target_update_interval=10_000,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        tensorboard_log="logs/dqn_trading",
        verbose=1
    )
    
    # 6) Train the model
    print("\n" + "="*50)
    print(f"Starting training for {total_timesteps:,} timesteps")
    print("="*50)
    
    tracker = TrainingTrackerCallback(check_freq=1000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=tracker,
        tb_log_name="dqn_trading",
        progress_bar=True
    )
    
    # 7) Save the trained model
    model.save("models/dqn_trading_model")
    
    # model = DQN.load("models/dqn_trading_model")

    print("\nTraining completed. Model saved to models/dqn_trading_model")
    
    # 8) Evaluate on test data
    print("\n" + "="*50)
    print("Starting evaluation on test data")
    print("="*50)
    
    test_env = make_env(test_df)
    obs, info = test_env.reset()
    
    terminated = False
    truncated = False
    equity_curve = [info['net_worth']]
    positions = []
    trade_count = 0
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        equity_curve.append(info['net_worth'])
        positions.append(info['position'])
        
        if info.get('trade_executed', False):
            trade_count += 1
        
        # Print every 100 steps
        if test_env.current_step % 100 == 0:
            print(f"Step {test_env.current_step}: "
                  f"Price: ${info['price']:.2f} | "
                  f"Position: {info['position']} | "
                  f"Net Worth: ${info['net_worth']:.2f}")

    # 9) Calculate and display performance metrics
    equity = np.array(equity_curve)
    total_ret, sharpe, max_dd, returns = compute_metrics(equity)
    
    print("\n" + "="*50)
    print("Backtesting Results")
    print("="*50)
    print(f"Initial Balance: ${equity_curve[0]:,.2f}")
    print(f"Final Balance: ${equity_curve[-1]:,.2f}")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    # print(f"Total Trades: {trade_count}")
    print(f"Long Positions: {positions.count('LONG')}")
    print(f"Short Positions: {positions.count('SHORT')}")
    print(f"Hold Positions: {positions.count('FLAT')}")
    
    # 10) Plot results
    plot_equity_curve(equity, returns, "Test Period Performance")
    print("\nEquity curve plot saved to results/Test_Period_Performance.png")

if __name__ == '__main__':
    main()