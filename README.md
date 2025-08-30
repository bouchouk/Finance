# Liquidity Key Levels RL Trader 🚀

A research project exploring a reinforcement-learning (RL) trading agent that “sees” both price action **and** options-derived liquidity key levels. The agent learns to trade SPY using a Deep Q-Learning (DQN) policy with risk-aware position management.

> **Disclaimer:** This project is for research and educational purposes only. **Not financial advice.** 📎  
> **Data availability:** You **won’t find all datasets in this repo** because of their large size. Use the collection scripts with your own API keys to rebuild the key tables. 📦

---

## Chapter 1 — Data Collection 📊

We assembled three complementary datasets:

- **Options data (Alpha Vantage API):**  
  Collected and consolidated into a single large file. Focus on open interest (OI), expirations, and strikes to infer weekly liquidity levels.

- **Price data (SPY):**  
  Gathered first from **TFT Broker**, then with **Barchart** data. After cleaning and deduplication, we organized everything into a unified price time series.

- **News articles (CNBC):**  
  Scraped and stored for later **market sentiment analysis** to contextualize regime shifts and high-volatility periods.

---

## Chapter 2 — Data Processing 🧹

We processed options and price data **in sync** to build model-ready features:

- **Weekly options key levels (statistical + analytical) 🧠📐:**  
  Identified using a **smart statistical edge on contract open interest (OI)** **combined with analytical analyses** each **week** using the **monthly expiration** dates to confirm meaningful support/resistance zones.

- **Weekly Expected Move (EM):**  
  Computed and aligned with the trading horizon to set volatility-aware targets/bounds.

- **Gamma Exposure (GEX) key levels:**  
  Derived to capture dealer-flow dynamics and likely support/resistance zones.

- **Final feature table:**  
  All signals—price OHLCV, OI levels, EM and GEX levels were merged into a **single table** indexed by timestamp, ready for exploration by the model.

---

## Chapter 3 — Reinforcement Learning Model 🧠

After experimenting with several RL environments, we converged on a DQN-based agent that performed best in our tests:

- **Action space:** `Buy`, `Sell`, `Hold`, plus **position adjustments** (add, trim, or fully exit).   
- **Risk management:** Smart sizing and safeguards to control drawdowns and tail risk.  
- **Why DQN?** DQN’s deep neural network approximator adapts better to this **high-dimensional, non-linear** setting than a simple Q-table.

---

