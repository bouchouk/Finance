# Liquidity Key Levels RL Trader 🚀

A research project exploring a reinforcement-learning (RL) trading agent that “sees” both price action **and** options-derived liquidity key levels. The agent learns to trade SPY using a Deep Q-Learning (DQN) policy with risk-aware position management.

> **Disclaimer:** This project is for research and educational purposes only. **Not financial advice.** 📎  
> **Data availability:** You **won’t find all raw datasets in this repo** because of their large size. We store them in external storage / Git LFS. Use the collection scripts with your own API keys to rebuild the key tables. 📦

---

## Chapter 1 — Data Collection 📊

We assembled three complementary datasets:

- **Options data (Alpha Vantage API):**  
  Collected and consolidated into a single large file. Focus on open interest (OI), expirations, and strikes to infer weekly and monthly liquidity levels.

- **Price data (SPY):**  
  Gathered first from **TFT Broker**, then augmented/validated with **Barchart** data. After cleaning and deduplication, we organized everything into a unified price time series.

- **News articles (CNBC):**  
  Scraped and stored for later **market sentiment analysis** to contextualize regime shifts and high-volatility periods.

---

## Chapter 2 — Data Processing 🧹

We processed options and price data **in sync** to build model-ready features:

- **Weekly options key levels:**  
  Identified using **open interest concentration** around weekly and **monthly expiration** dates.

- **Expected Move (EM):**  
  Computed and aligned with the trading horizon to set volatility-aware targets/bounds.

- **Gamma Exposure (GEX) key levels:**  
  Derived to capture dealer-flow dynamics and likely support/resistance zones.

- **Final feature table:**  
  All signals—price OHLCV, OI clusters, EM, GEX levels, and optional sentiment tags—were merged into a **single, tidy table** indexed by timestamp, ready for exploration and modeling.

---

## Chapter 3 — Reinforcement Learning Model 🧠

After experimenting with several RL environments, we converged on a DQN-based agent that performed best in our tests:

- **Action space:** `Buy`, `Sell`, `Hold`, plus **position adjustments** (add, trim, or fully exit).  
- **State representation:** Windowed price features + options-based key levels (weekly/monthly OI, EM, GEX) and optional sentiment cues.  
- **Risk management:** Smart sizing and safeguards (e.g., volatility-aware limits, stop/exit logic) to control drawdowns and tail risk.  
- **Why DQN?** DQN’s deep neural network approximator adapts better to this **high-dimensional, non-linear** setting than a simple Q-table.

---

## Quick Start 🛠️

```bash
# 1) Set up environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2) Collect or place raw data
# - Alpha Vantage key in your .env
# - Price data from TFT/Barchart
# - CNBC articles (optional)

# 3) Process features
python scripts/build_features.py  # outputs a single model-ready parquet/csv

# 4) Train & evaluate
python scripts/train_dqn.py
python scripts/backtest.py
