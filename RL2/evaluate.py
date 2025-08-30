import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_processor import load_config, preprocess_data, create_sequences, train_test_split
from agents.dqn_agent import DQNAgent
from envs.trading_env import TradingEnv

def evaluate():
    # Load configuration
    config = load_config()

    # Load and preprocess data
    df = pd.read_csv(config['data']['path'])
    data = preprocess_data(df)

    # Prepare sequences and timestamps
    feature_cols = [c for c in data.columns if c not in ['date_time', 'close']]
    arr = np.column_stack((data[feature_cols].values, data['close'].values))
    seqs, prices = create_sequences(arr, config['model']['window_size'])
    times = data['date_time'].values[config['model']['window_size']:]

    # Split into train/test (last test_size fraction) without shuffling
    _, (test_s, test_p, test_t) = train_test_split(seqs, prices, times, test_size=0.25)

    # Initialize environment for full test set
    max_steps = len(test_s) - 1
    env = TradingEnv(
        sequences=test_s,
        close_prices=test_p,
        timestamps=test_t,
        window_size=config['model']['window_size'],
        max_steps=max_steps
    )
    state = env.reset()

    # Load trained agent
    state_shape = (config['model']['window_size'], test_s.shape[2] + 6)
    agent = DQNAgent(state_shape, env.action_space.n, config)
    agent.load(os.path.join('results', 'dqn_final.h5'))
    agent.epsilon = 0.0

    # Containers for logging
    trade_open         = None
    trade_returns      = []
    trade_drawdowns    = []
    trade_records      = []
    portfolio_log      = []
    timestamps_log     = []
    actions_log        = []
    shares_log         = []
    entry_portfolio_log= []
    price_log          = []  # current price at each step
    entry_price_log    = []  # price at trade entry for open trades
    trade_type_log     = []  # 'long' or 'short' for open trades

    # Loop through test period
    while True:
        price     = env.get_current_price()
        timestamp = env.get_current_time()

        action = agent.act(state)
        next_state, _, done, _ = env.step(action)

        # Log step-level details
        portfolio_log.append(env.portfolio)
        timestamps_log.append(timestamp)
        actions_log.append(action)
        shares_log.append(getattr(env, 'size', np.nan))
        entry_portfolio_log.append(trade_open['entry_portfolio'] if trade_open else np.nan)
        price_log.append(price)
        entry_price_log.append(trade_open['entry_price'] if trade_open else np.nan)
        trade_type_log.append(trade_open['type'] if trade_open else None)

        # If a trade is open, record price path
        if trade_open is not None:
            trade_open['prices'].append(price)

        # On open: action 2 (long) or 3 (short)
        if action in (2, 3) and trade_open is None:
            trade_open = {
                'entry_price': price,
                'entry_time': timestamp,
                'type': 'long' if action == 2 else 'short',
                'prices': [price],
                'entry_portfolio': env.portfolio,
            }

        # On full close: action 0 when a position exists
        if action == 0 and trade_open is not None:
            exit_price = price
            t = trade_open

            # compute return
            if t['type'] == 'long':
                ret = (exit_price - t['entry_price']) / t['entry_price']
            else:
                ret = (t['entry_price'] - exit_price) / t['entry_price']

            # compute this trade's max drawdown
            prices = t['prices']
            if t['type'] == 'long':
                trough = min(prices)
                dd = (t['entry_price'] - trough) / t['entry_price']
            else:
                peak = max(prices)
                dd = (peak - t['entry_price']) / t['entry_price']

            t['drawdown'] = dd
            trade_drawdowns.append(dd)

            # assign trade number
            t['trade_num'] = len(trade_records) + 1

            # finalize record
            t.update({
                'exit_price': exit_price,
                'exit_time': timestamp,
                'return': ret,
            })
            trade_returns.append(ret)
            trade_records.append(t)
            trade_open = None

        state = next_state
        if done:
            break

    # Compute metrics
    returns_arr = np.array(trade_returns)
    sharpe_ratio = (np.mean(returns_arr) / np.std(returns_arr)) if returns_arr.size and np.std(returns_arr) != 0 else np.nan
    max_drawdown = max(trade_drawdowns) if trade_drawdowns else 0.0
    final_portfolio = portfolio_log[-1] if portfolio_log else env.initial_balance
    num_trades = len(trade_records)

    # Cumulative returns for plotting
    cum_returns = np.cumprod(1 + returns_arr)

    # Persist results
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(trade_records).to_csv(os.path.join('results', 'trade_details.csv'), index=False)
    pd.DataFrame({
        'timestamp': timestamps_log,
        'action': actions_log,
        'shares': shares_log,
        'portfolio': portfolio_log,
        'entry_portfolio': entry_portfolio_log,
        'entry_price': entry_price_log,
        'price': price_log,
        'trade_type': trade_type_log
    }).to_csv(os.path.join('results', 'step_details.csv'), index=False)
    pd.DataFrame([{  
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_portfolio': final_portfolio,
        'num_trades': num_trades
    }]).to_csv(os.path.join('results', 'evaluation_metrics.csv'), index=False)

    # Print summary
    print(f"Final Portfolio Value: {final_portfolio:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")

    # Plot portfolio over test period
    plt.figure()
    plt.plot(timestamps_log, portfolio_log)
    plt.title('Portfolio Value Over Test Period')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'portfolio_full_test.png'))
    plt.show()

    # Plot individual trade returns
    plt.figure()
    plt.bar(range(len(trade_returns)), returns_arr)
    plt.title('Individual Trade Returns')
    plt.xlabel('Trade Number')
    plt.ylabel('Return')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'trade_returns.png'))
    plt.show()

    # Plot cumulative returns with metrics
    plt.figure()
    plt.plot(cum_returns)
    plt.title(f'Cumulative Returns over Trades\nSharpe: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Return')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'cumulative_returns.png'))
    plt.show()

if __name__ == '__main__':
    evaluate()
