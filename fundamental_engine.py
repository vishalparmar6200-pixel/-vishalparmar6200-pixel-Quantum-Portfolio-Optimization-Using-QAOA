"""
Fundamental Analysis & Valuation Engine
=======================================
Extracts valuation metrics, 52-week price channels, P/E, Market Cap,
and computes a fundamental health rating.
"""

import yfinance as yf

class FundamentalEngine:
    @staticmethod
    def get_stock_fundamentals(ticker_symbol):
        """
        Retrieves fundamental metrics for a given ticker.
        """
        clean_name = ticker_symbol.replace(".NS", "").replace(".BO", "")
        result = {
            "symbol": clean_name,
            "full_ticker": ticker_symbol,
            "company_name": clean_name,
            "sector": "Large Cap",
            "market_cap": "N/A",
            "pe_ratio": 24.5,
            "pb_ratio": 3.2,
            "dividend_yield": 1.2,
            "week_52_high": 0.0,
            "week_52_low": 0.0,
            "fund_score": 70.0
        }

        try:
            t = yf.Ticker(ticker_symbol)
            info = t.info or {}
            
            result["company_name"] = info.get("shortName", info.get("longName", clean_name))
            result["sector"] = info.get("sector", "Equities")
            
            m_cap = info.get("marketCap")
            if m_cap:
                if m_cap >= 1e12:
                    result["market_cap"] = f"₹{m_cap/1e12:.2f} Lakh Cr"
                elif m_cap >= 1e7:
                    result["market_cap"] = f"₹{m_cap/1e7:.2f} Cr"
                else:
                    result["market_cap"] = f"${m_cap/1e9:.2f} B"

            result["pe_ratio"] = round(info.get("trailingPE", info.get("forwardPE", 25.0)), 2)
            result["pb_ratio"] = round(info.get("priceToBook", 3.0), 2)
            result["dividend_yield"] = round((info.get("dividendYield", 0.01) or 0.01) * 100, 2)
            result["week_52_high"] = round(info.get("fiftyTwoWeekHigh", 0.0), 2)
            result["week_52_low"] = round(info.get("fiftyTwoWeekLow", 0.0), 2)

            # Score fundamentals
            f_score = 50.0
            if 10.0 <= result["pe_ratio"] <= 35.0:
                f_score += 15.0
            if result["dividend_yield"] >= 1.0:
                f_score += 10.0
            if info.get("returnOnEquity", 0.1) > 0.15:
                f_score += 10.0
            
            result["fund_score"] = round(min(95.0, f_score), 1)

        except Exception:
            pass

        return result
