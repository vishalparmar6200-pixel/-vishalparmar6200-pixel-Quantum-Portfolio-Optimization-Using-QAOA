"""
Flask Web Application: Smart Stock AI & Quantum Daily Recommendation Platform
=============================================================================
Serves the interactive web interface, real-time market data endpoints,
news sentiment feeds, and quantum portfolio optimization engine.
"""

import sys
import os
try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

from flask import Flask, render_template, jsonify, request
from recommender_engine import StockRecommenderEngine
from news_sentiment_engine import NewsSentimentEngine
from quantum_portfolio_engine import QuantumPortfolioEngine

app = Flask(__name__, template_folder="templates", static_folder="static")

# Cache daily picks in memory for fast instant loading
CACHED_PICKS = None
CACHED_MACRO_NEWS = None

DEFAULT_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "TATAMOTORS.NS", "SUNPHARMA.NS",
    "MARUTI.NS", "TATASTEEL.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"
]


@app.route("/")
def home():
    """Renders the main web dashboard."""
    return render_template("index.html")


@app.route("/api/daily-picks")
def get_daily_picks():
    """
    Returns AI multi-factor daily stock recommendations.
    """
    global CACHED_PICKS
    refresh = request.args.get("refresh", "false").lower() == "true"
    
    if CACHED_PICKS is None or refresh:
        print(">> Generating fresh daily AI stock recommendations...", flush=True)
        CACHED_PICKS = StockRecommenderEngine.get_daily_market_picks(DEFAULT_UNIVERSE)
        
    return jsonify({
        "status": "success",
        "count": len(CACHED_PICKS),
        "data": CACHED_PICKS
    })


@app.route("/api/analyze/<symbol>")
def analyze_stock(symbol):
    """
    Generates on-demand deep-dive analysis for any searched stock symbol.
    """
    clean_sym = symbol.strip().upper()
    try:
        data = StockRecommenderEngine.analyze_single_stock(clean_sym, lookback_period="6mo")
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/macro-news")
def get_macro_news():
    """
    Returns real-time financial and macro market news headlines.
    """
    global CACHED_MACRO_NEWS
    refresh = request.args.get("refresh", "false").lower() == "true"
    
    if CACHED_MACRO_NEWS is None or refresh:
        CACHED_MACRO_NEWS = NewsSentimentEngine.fetch_market_macro_news()
        
    return jsonify({
        "status": "success",
        "news": CACHED_MACRO_NEWS
    })


@app.route("/api/quantum-portfolio", methods=["POST"])
def build_quantum_portfolio():
    """
    Runs QAOA on the selected basket of top recommended stocks.
    """
    req_json = request.get_json() or {}
    tickers = req_json.get("tickers", ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "TATAMOTORS"])
    budget = int(req_json.get("budget", 2))
    risk_q = float(req_json.get("risk_q", 0.5))

    if budget >= len(tickers):
        budget = max(1, len(tickers) - 1)

    try:
        result = QuantumPortfolioEngine.optimize_ai_portfolio(tickers, budget=budget, q_risk=risk_q, p_layers=2)
        return jsonify({"status": "success", "portfolio": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Launch local web server on port 5000
    print("=" * 70)
    print(" SMART STOCK AI & QUANTUM RECOMMENDATION WEB SERVER")
    print("=" * 70)
    print(" Dashboard URL: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)
