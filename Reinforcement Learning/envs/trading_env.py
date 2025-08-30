# ---------- envs/trading_env.py ----------
import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, sequences, close_prices, timestamps, window_size, max_steps=500):
        super().__init__()
        self.sequences    = sequences
        self.close_prices = close_prices
        self.timestamps   = timestamps
        self.window_size  = window_size
        self.max_steps    = max_steps

        # Discrete actions: 0=close,1=hold,2=enter long,3=enter short,4=scale long,5=scale short,6=partial close long,7=partial close short
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, sequences.shape[2] + 6),
            dtype=np.float32,
        )

        # Financial parameters
        self.initial_balance    = 10_000.0
        self.cash               = self.initial_balance
        self.portfolio          = self.initial_balance
        self.max_position_size  = 100_000.0  # max notional dollars per position

        self.reset()

    def reset(self):
        # Start at a random contiguous slice
        self.start_idx    = np.random.randint(0, len(self.sequences) - self.max_steps)
        self.current_idx  = self.start_idx
        self.episode_step = 0

        # Position state: 'size' is share count
        self.position = {
            'type':        0,    # 0=flat, 1=long, -1=short
            'size':        0.0,  # shares
            'entry_price': 0.0,  # price per share
            'duration':    0,    # steps held
            'scale_count': 0,    # number of adds
        }

        # Reset P&L
        self.cash      = self.initial_balance
        self.portfolio = self.initial_balance
        return self._get_state()

    def _get_state(self):
        market     = self.sequences[self.current_idx]
        pos        = self.position
        price_diff = ((self.get_current_price() - pos['entry_price']) / pos['entry_price']) \
                        if pos['entry_price'] > 0 else 0.0
        pos_feat   = np.array([
            pos['type'],
            pos['size']/350,  #max sz is max_position_size / min price in the data (300)
            pos['duration'] / 100.0,
            price_diff,
            pos['scale_count'] / 3.0,
            self.cash / self.portfolio if self.portfolio > 0 else 0.0,
        ], dtype=np.float32)
        pos_matrix = np.tile(pos_feat, (self.window_size, 1))
        return np.concatenate([market, pos_matrix], axis=1)

    def get_current_price(self):
        return self.close_prices[self.current_idx]

    def get_current_time(self):
        return self.timestamps[self.current_idx]

    def _get_position_size(self, pct):
        # Determine notional = pct of available cash (capped), then convert to shares
        notional = min(pct * self.cash, self.max_position_size)
        shares   = notional / self.get_current_price()
        return shares

    def _execute_action(self, action):
        price     = self.get_current_price()
        prev_port = self.portfolio
        print('selfposition : ',self.position['type'],' action: ',action)

        # 0: CLOSE_POSITION
        if action == 0 and self.position['type'] != 0:
            shares = self.position['size']
            entry  = self.position['entry_price']
            # Realize P&L
            pnl    = shares * (price - entry) * self.position['type']
            # Close: for longs receive shares*price; for shorts pay shares*price
            self.cash += shares * entry if self.position['type'] == 1 else -shares * entry
            self.cash += pnl
            # Reset
            self.position = {'type':0,'size':0.0,'entry_price':0.0,'duration':0,'scale_count':0}

        # 1: HOLD_NO_POSITION (or hold existing)
        elif action == 1:
            pass

        # 2: ENTER_LONG
        elif action == 2 and self.position['type'] == 0:
            shares = self._get_position_size(0.5)
            cost   = shares * price
            self.cash      -= cost
            self.position  = {'type':1,'size':shares,'entry_price':price,'duration':1,'scale_count':1}

        # 3: ENTER_SHORT
        elif action == 3 and self.position['type'] == 0:
            shares = self._get_position_size(0.5)
            proceeds = shares * price
            self.cash      += proceeds
            self.position  = {'type':-1,'size':shares,'entry_price':price,'duration':1,'scale_count':1}

        # 4: SCALE_IN_LONG
        elif action == 4 and self.position['type'] == 1:
            add_shares = self._get_position_size(0.5)
            old_shares = self.position['size']
            new_shares = min(old_shares + add_shares, self.max_position_size / price)
            cost       = (new_shares - old_shares) * price
            self.cash  -= cost
            # Weighted entry by shares
            entry_old  = self.position['entry_price']
            entry_new  = (entry_old * old_shares + price * (new_shares - old_shares)) / new_shares
            self.position.update(size=new_shares, entry_price=entry_new,
                                 scale_count=self.position['scale_count'] + 1)

        # 5: SCALE_IN_SHORT
        elif action == 5 and self.position['type'] == -1:
            add_shares = self._get_position_size(0.5)
            old_shares = self.position['size']
            new_shares = min(old_shares + add_shares, self.max_position_size / price)
            proceeds   = (new_shares - old_shares) * price
            self.cash  += proceeds
            entry_old  = self.position['entry_price']
            entry_new  = (entry_old * old_shares + price * (new_shares - old_shares)) / new_shares
            self.position.update(size=new_shares, entry_price=entry_new,
                                 scale_count=self.position['scale_count'] + 1)

        # 6: PARTIAL_CLOSE_LONG
        elif action == 6 and self.position['type'] == 1 and self.position['size'] > 0:
            close_shares = 0.2 * self.position['size']
            self.cash   += close_shares * price
            self.position['size'] -= close_shares

        # 7: PARTIAL_CLOSE_SHORT
        elif action == 7 and self.position['type'] == -1 and self.position['size'] > 0:
            close_shares = 0.2 * self.position['size']
            self.cash   -= close_shares * price
            self.position['size'] -= close_shares

        # --- update portfolio: cash + position mark-to-market ---
        pos = self.position
        self.portfolio = self.cash + pos['type'] * pos['size'] * price
    
        # --- reward: log-return ---
        if prev_port > 0:
            reward = np.log(self.portfolio / prev_port)
        else:
            reward = 0.0

        return reward

    def step(self, action):
        # 1. Execute the user’s action and get reward
        reward = self._execute_action(action)

        # 2. If you still have a position, bump its duration
        if self.position['type'] != 0:
            self.position['duration'] += 1

        # 3. Advance episode counters
        self.episode_step += 1
        self.current_idx  += 1

        # 4. Check for natural episode termination
        done = (
            self.episode_step >= self.max_steps or
            self.current_idx  >= len(self.sequences)
        )

        # --- Forced liquidation on extreme moves ---
        if self.position['type'] != 0:
            change = (self.get_current_price() - self.position['entry_price']) \
                    / self.position['entry_price']
            signed_return = change * self.position['type']
            # stop‐loss: down 0.5%   or take‐profit: up 3%
            if signed_return < -0.005 or signed_return > 0.10:
                # force close, you could also capture its reward if you like
                self._execute_action(0)

        # 5. Return next state, the original reward, done flag, and empty info
        return self._get_state(), reward, done, {}

