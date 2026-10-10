# SmartStock AI: Daily Stock Recommendation & Quantum Intelligence Platform

An intelligent web application that analyzes daily real-world stock market data (NSE Nifty 50 & Global), live financial news headlines, and macro sentiment, delivering daily **actionable stock recommendations** and **Quantum QAOA-optimized portfolios**.

---

## 🌟 Key Features

1. **Daily AI Stock Picks**:
   - Analyzes real-time price trends, moving averages, volume surges, and RSI/MACD momentum.
   - Categorizes stocks into **Strong Buy**, **Buy**, **Hold / Neutral**, and **Avoid** with Confidence Scores (0-100%).
   - Calculates dynamic **Target Price 1 & 2** and **Stop Loss** levels based on volatility (ATR).

2. **Live Financial News & Sentiment Analyzer**:
   - Ingests real-time financial headlines via Google News RSS & financial feeds.
   - Domain-specific financial NLP lexicon analyzing bullish vs. bearish market sentiment and catalysts.

3. **Quantum QAOA Daily Portfolio Builder**:
   - Seamlessly converts the top recommended stocks into a Markowitz QUBO / Ising Hamiltonian.
   - Solves for the optimal diversified daily asset allocation using the **Quantum Approximate Optimization Algorithm (QAOA)** with **PennyLane**.

4. **Interactive Stock Deep-Dive Visualizer**:
   - 45-day price trend and volume charts.
   - Indicator health gauges (Technical Score, Sentiment Score, Fundamental Health).
   - Instant search for any stock on the National Stock Exchange (NSE).

---

## 🚀 Quickstart & Running the Web Server

### 1. Install Dependencies
```bash
pip install flask yfinance pennylane pandas numpy scipy plotly feedparser
```

### 2. Launch the Web Application
```bash
python app.py
```

### 3. Open in Browser
Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 📂 Project Architecture

```
smart_stock_ai_web/
│
├── app.py                      # Flask Web Application & REST API Server
├── technical_engine.py         # RSI, MACD, Moving Averages, Bollinger Bands Calculator
├── news_sentiment_engine.py    # Real-time News Ingestion & Sentiment Polarity Engine
├── fundamental_engine.py       # P/E, Market Cap, Valuation & Financial Health Engine
├── recommender_engine.py       # Multi-Factor Scoring, Target Price & Signal Generator
├── quantum_portfolio_engine.py # QAOA Quantum Portfolio Optimizer
├── templates/
│   └── index.html              # Modern, Responsive Dark-Themed Web Dashboard
└── README.md                   # Full documentation
```
