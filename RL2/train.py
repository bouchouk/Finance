import numpy as np
import tensorflow as tf
import random
import pandas as pd
from tqdm import tqdm
import os
from envs.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent
from utils.data_processor import preprocess_data, create_sequences, train_test_split, load_config

# Seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# GPU config omitted for brevity

def main():
    config = load_config()
    df = pd.read_csv(config['data']['path'])
    data = preprocess_data(df)
    print('data.columns :',data.columns)
    # Prepare sequences
    feature_cols = [c for c in data.columns if c not in ['date_time','close']]
    arr = np.column_stack((data[feature_cols].values, data['close'].values))
    seqs, prices = create_sequences(arr, config['model']['window_size'])
    times = data['date_time'].values[config['model']['window_size']:]

    (train_s, train_p, train_t), (test_s, test_p, test_t) = train_test_split(
        seqs, prices, times, test_size=config['training']['test_size']
    )

    env = TradingEnv(train_s, train_p, train_t, config['model']['window_size'],
                     max_steps=config['training']['max_episode_steps'])
    state_shape = (config['model']['window_size'], train_s.shape[2] + 6)
    agent = DQNAgent(state_shape, env.action_space.n, config)

    rewards, history = [], []
    pbar = tqdm(range(config['training']['episodes']), desc="Training Episodes")
    for ep in pbar:
        state = env.reset()
        total_r = 0.0
        done = False
        while not done:
            a = agent.act(state)
            s2, r, done, _ = env.step(a)
            agent.remember(state, a, r, s2, done)
            state = s2
            total_r += r
        agent.replay()
        rewards.append(total_r)
        history.append(env.portfolio)

        avg_r = np.nanmean(rewards[-100:])
        ret_pct = (env.portfolio - env.initial_balance) / env.initial_balance * 100
        pbar.set_postfix({'AvgR':f"{avg_r:.4f}", 'Epsilon':f"{agent.epsilon:.4f}",
                          'Port':f"${env.portfolio:.2f}", 'Return%':f"{ret_pct:.2f}%"})

        if ep>0 and ep%100==0:
            os.makedirs('results', exist_ok=True)
            agent.save(f"results/dqn_{ep}.h5")

    # Final save
    os.makedirs('results', exist_ok=True)
    agent.save('results/dqn_final.h5')
    pd.DataFrame({'ep':range(len(rewards)),'reward':rewards,'port':history}).to_csv('results/history.csv', index=False)

if __name__ == "__main__":
    main()

