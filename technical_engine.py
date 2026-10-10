"""
Technical Indicators & Quantitative Analysis Engine
===================================================
Calculates RSI, MACD, Moving Averages (SMA 20/50/200), Bollinger Bands,
Volume Anomalies, and generates a normalized Technical Health Score (0-100).
"""

import numpy as np
import pandas as pd

class TechnicalAnalysisEngine:
    @staticmethod
    def compute_rsi(prices, period=14):
        """Computes Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_macd(prices, fast=12, slow=26, signal=9):
        """Computes MACD Line, Signal Line, and Histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def compute_bollinger_bands(prices, period=20, num_std=2):
        """Computes Bollinger Bands (Upper, Middle SMA, Lower)."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    @classmethod
    def analyze_stock(cls, df):
        """
        Runs comprehensive technical analysis on historical price/volume DataFrame.
        Expected columns: Open, High, Low, Close, Volume
        """
        if len(df) < 30:
            return None

        close = df['Close']
        volume = df['Volume'] if 'Volume' in df else pd.Series(np.ones(len(df)), index=df.index)

        # 1. Moving Averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(min(50, len(close))).mean().iloc[-1]
        sma_200 = close.rolling(min(200, len(close))).mean().iloc[-1] if len(close) >= 100 else sma_50
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2] if len(close) >= 2 else current_price
        pct_change_1d = ((current_price - prev_price) / prev_price) * 100

        # 2. RSI (14)
        rsi_series = cls.compute_rsi(close)
        current_rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

        # 3. MACD
        macd_line, signal_line, hist = cls.compute_macd(close)
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_hist = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2]) if len(hist) >= 2 else current_hist

        # 4. Bollinger Bands
        upper_bb, mid_bb, lower_bb = cls.compute_bollinger_bands(close)
        c_upper = float(upper_bb.iloc[-1])
        c_lower = float(lower_bb.iloc[-1])
        c_mid = float(mid_bb.iloc[-1])
        bb_width = ((c_upper - c_lower) / c_mid) * 100

        # 5. Volume Surge
        avg_vol_20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        current_vol = volume.iloc[-1]
        vol_ratio = float(current_vol / (avg_vol_20 + 1e-9))

        # 6. Technical Score Calculation (0 to 100)
        score = 50.0  # Base neutral score

        # RSI scoring
        if 40 <= current_rsi <= 60:
            score += 5.0
        elif 30 <= current_rsi < 40:
            score += 15.0  # Approaching oversold bounce
        elif current_rsi < 30:
            score += 20.0  # Deep oversold / potential reversal
        elif 60 < current_rsi <= 70:
            score += 10.0  # Strong bullish momentum
        elif current_rsi > 70:
            score -= 10.0  # Overbought warning

        # Trend & Moving Average scoring
        if current_price > sma_20:
            score += 8.0
        if current_price > sma_50:
            score += 10.0
        if current_price > sma_200:
            score += 7.0
        if sma_20 > sma_50:
            score += 5.0  # Short term golden cross

        # MACD momentum scoring
        if current_macd > current_signal:
            score += 10.0
            if current_hist > prev_hist:
                score += 5.0  # Expanding bullish momentum
        else:
            score -= 8.0

        # Volume confirmation
        if vol_ratio > 1.2 and current_price > prev_price:
            score += 10.0  # Bullish volume breakout
        elif vol_ratio > 1.5 and current_price < prev_price:
            score -= 10.0  # High volume selloff

        # Normalize score between 10 and 98
        final_score = max(10.0, min(98.0, score))

        # Trend label
        if final_score >= 70:
            trend = "Bullish"
        elif final_score <= 40:
            trend = "Bearish"
        else:
            trend = "Neutral"

        return {
            "current_price": float(current_price),
            "prev_price": float(prev_price),
            "pct_change_1d": float(pct_change_1d),
            "rsi": round(current_rsi, 2),
            "macd": round(current_macd, 2),
            "macd_signal": round(current_signal, 2),
            "macd_hist": round(current_hist, 2),
            "sma_20": round(float(sma_20), 2),
            "sma_50": round(float(sma_50), 2),
            "sma_200": round(float(sma_200), 2),
            "bb_upper": round(c_upper, 2),
            "bb_lower": round(c_lower, 2),
            "vol_ratio": round(vol_ratio, 2),
            "tech_score": round(final_score, 1),
            "trend": trend
        }
