import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Gymnasium environment for trading with learned stop-loss discretized into bins
    and fixed risk per trade. Uses discrete actions for compatibility with DQN.

    Observation: flattened window of past rows (OHLC + extra features).
    Actions: Discrete(3 * n_stop_bins) where:
      dir_idx = action // n_stop_bins
        0 = HOLD, 1 = LONG, 2 = SHORT
      stop_bin = action % n_stop_bins
        stop_pct = (stop_bin + 1) / n_stop_bins * max_stop_pct
    Reward: change in net worth each step.
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        window_size: int = 10,
        initial_balance: float = 100_000,
        transaction_cost_pct: float = 0.001,
        risk_pct: float = 0.01,
        shares_per_lot: int = 100,
        max_stop_pct: float = 0.5,
        n_stop_bins: int = 10,
        render_mode: str = None
    ):
        super().__init__()
        # Data and parameters
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_pct = risk_pct
        self.shares_per_lot = shares_per_lot
        self.max_stop_pct = max_stop_pct
        self.n_stop_bins = n_stop_bins
        self.render_mode = render_mode

        # Action space: discrete combined direction and stop-bin
        self.action_space = spaces.Discrete(3 * self.n_stop_bins)

        # Observation space: flattened window_size × len(feature_cols)
        obs_dim = window_size * len(feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize in reset
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Internal state
        self.current_step = self.window_size - 1
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.entry_price = None
        self.previous_net_worth = float(self.initial_balance)
        self.trades_executed = 0
        
        info = {'status': 'Reset', 'net_worth': self.initial_balance}
        return self._get_observation(), info

    def _get_observation(self):
        start = self.current_step - (self.window_size - 1)
        end = self.current_step + 1
        block = self.df.loc[start:end - 1, self.feature_cols].values
        return block.flatten().astype(np.float32)

    def step(self, action: int):
        # Decode discrete action
        dir_idx = action // self.n_stop_bins
        stop_bin = action % self.n_stop_bins
        stop_pct = (stop_bin + 1) / self.n_stop_bins * self.max_stop_pct

        # Current market price
        price = float(self.df['close'].iloc[self.current_step])
        equity = self.balance + self.shares_held * price
        trade_executed = False

        # Execute action
        if dir_idx == 1:  # LONG
            # If short, cover first
            if self.shares_held < 0:
                cover_qty = -self.shares_held
                cost = cover_qty * price * (1 + self.transaction_cost_pct)
                self.balance -= cost
                self.shares_held = 0.0
                self.entry_price = None
                trade_executed = True
            # If flat, open long
            if self.shares_held == 0:
                stop_price = price * (1 - stop_pct)
                loss_per_share = price - stop_price
                if loss_per_share > 0:
                    risk_amount = self.risk_pct * equity
                    raw_shares = risk_amount / loss_per_share
                    lots = int(raw_shares // self.shares_per_lot)
                    qty = lots * self.shares_per_lot
                    cost = qty * price * (1 + self.transaction_cost_pct)
                    if qty > 0 and cost <= self.balance:
                        self.balance -= cost
                        self.shares_held = qty
                        self.entry_price = price
                        trade_executed = True

        elif dir_idx == 2:  # SHORT
            # If long, liquidate first
            if self.shares_held > 0:
                proceeds = self.shares_held * price * (1 - self.transaction_cost_pct)
                self.balance += proceeds
                self.shares_held = 0.0
                self.entry_price = None
                trade_executed = True
            # If flat, open short
            if self.shares_held == 0:
                stop_price = price * (1 + stop_pct)
                loss_per_share = stop_price - price
                if loss_per_share > 0:
                    risk_amount = self.risk_pct * equity
                    raw_shares = risk_amount / loss_per_share
                    lots = int(raw_shares // self.shares_per_lot)
                    qty = lots * self.shares_per_lot
                    proceeds = qty * price * (1 - self.transaction_cost_pct)
                    if qty > 0:
                        self.balance += proceeds
                        self.shares_held = -qty
                        self.entry_price = price
                        trade_executed = True
        
        if trade_executed:
            self.trades_executed += 1

        # Advance time
        self.current_step += 1
        
        # Termination conditions
        terminated = False
        truncated = self.current_step >= len(self.df) - 1  # End of data
        
        # Compute current net worth
        next_price = float(self.df['close'].iloc[min(self.current_step, len(self.df)-1)])
        net_worth = self.balance + self.shares_held * next_price
        
        # Bankruptcy condition
        if net_worth <= self.initial_balance * 0.1:  # 90% loss
            terminated = True

        # Compute reward
        reward = net_worth - self.previous_net_worth
        self.previous_net_worth = net_worth

        # Prepare info dictionary
        info = {
            'net_worth': net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position': 'LONG' if self.shares_held > 0 else 
                       'SHORT' if self.shares_held < 0 else 'FLAT',
            'price': next_price,
            'step': self.current_step,
            'trades': self.trades_executed,
            'terminated': terminated,
            'truncated': truncated
        }
        
        # Get next observation if not done
        if terminated or truncated:
            obs = None
        else:
            obs = self._get_observation()

        # Auto-render if needed
        if self.render_mode == 'human':
            self.render()
            
        return obs, reward, terminated, truncated, info

    def render(self):
        # Current price might be at current_step - 1 after step
        idx = min(self.current_step, len(self.df) - 1)
        price = float(self.df['close'].iloc[idx])
        net_worth = self.balance + self.shares_held * price
        pos = 'LONG' if self.shares_held > 0 else 'SHORT' if self.shares_held < 0 else 'FLAT'
        
        print(f"Step {self.current_step} | Price:{price:.2f} | Bal:{self.balance:.2f} | "
              f"Pos:{pos} {abs(self.shares_held):.0f} | NW:{net_worth:.2f}")

    def close(self):
        pass