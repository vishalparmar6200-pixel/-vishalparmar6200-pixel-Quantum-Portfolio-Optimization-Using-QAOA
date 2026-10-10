"""
Quantum Portfolio Engine for AI Recommended Stock Baskets
=========================================================
Runs QAOA (Quantum Approximate Optimization Algorithm) on the top AI-picked stocks
to construct an optimal, risk-diversified daily investment portfolio.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pennylane as qml
from scipy.optimize import minimize

class QuantumPortfolioEngine:
    @classmethod
    def optimize_ai_portfolio(cls, candidate_tickers, budget=3, q_risk=0.5, p_layers=2):
        """
        Runs QAOA optimization on top AI recommended stocks.
        """
        full_tickers = [t if "." in t else f"{t}.NS" for t in candidate_tickers]
        n = len(full_tickers)

        try:
            raw = yf.download(full_tickers, period="6mo", progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw['Close']
            else:
                df = raw['Close'] if 'Close' in raw else raw

            df = df[full_tickers].dropna()
            returns = df.pct_change().dropna()
            mu = returns.mean().values * 252
            sigma = returns.cov().values * 252
        except Exception:
            # Fallback
            mu = np.linspace(0.28, 0.15, n)
            vol = np.linspace(0.22, 0.35, n)
            corr = np.full((n, n), 0.4)
            np.fill_diagonal(corr, 1.0)
            sigma = np.outer(vol, vol) * corr

        # QUBO Formulation: Cost(x) = q * x^T Sigma x - mu^T x + lambda * (sum(x) - B)^2
        penalty_lambda = 1.0
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    Q[i, i] += q_risk * sigma[i, i] - mu[i] + penalty_lambda * (1 - 2 * budget)
                else:
                    Q[i, j] += q_risk * sigma[i, j] + 2 * penalty_lambda

        # Map to Ising H_C = sum h_i Z_i + sum J_ij Z_i Z_j + offset
        h = np.zeros(n)
        J = np.zeros((n, n))
        offset = penalty_lambda * (budget ** 2)

        for i in range(n):
            offset += Q[i, i] / 2.0
            h[i] -= Q[i, i] / 2.0
            for j in range(i + 1, n):
                val = Q[i, j] + Q[j, i]
                offset += val / 4.0
                h[i] -= val / 4.0
                h[j] -= val / 4.0
                J[i, j] += val / 4.0

        # Build PennyLane Hamiltonian
        coeffs, obs = [], []
        for i in range(n):
            if abs(h[i]) > 1e-9:
                coeffs.append(float(h[i]))
                obs.append(qml.PauliZ(i))
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-9:
                    coeffs.append(float(J[i, j]))
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if abs(offset) > 1e-9:
            coeffs.append(float(offset))
            obs.append(qml.Identity(0))

        H_cost = qml.Hamiltonian(coeffs, obs)
        dev = qml.device("default.qubit", wires=n)

        def circuit(params):
            gammas, betas = params[0], params[1]
            for i in range(n):
                qml.Hadamard(wires=i)
            for l in range(p_layers):
                g, b = gammas[l], betas[l]
                for i in range(n):
                    if abs(h[i]) > 1e-9:
                        qml.RZ(2 * g * h[i], wires=i)
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(J[i, j]) > 1e-9:
                            qml.IsingZZ(2 * g * J[i, j], wires=[i, j])
                for i in range(n):
                    qml.RX(2 * b, wires=i)

        @qml.qnode(dev)
        def cost_node(params):
            circuit(params)
            return qml.expval(H_cost)

        @qml.qnode(dev)
        def prob_node(params):
            circuit(params)
            return qml.probs(wires=range(n))

        # Optimize QAOA
        np.random.seed(42)
        init_flat = np.random.uniform(0.1, 0.5, 2 * p_layers)

        def obj(p_flat):
            p_2d = np.array([p_flat[:p_layers], p_flat[p_layers:]])
            return float(cost_node(p_2d))

        res = minimize(obj, init_flat, method="COBYLA", options={"maxiter": 60})
        opt_2d = np.array([res.x[:p_layers], res.x[p_layers:]])
        probs = prob_node(opt_2d)

        # Parse valid states
        valid_candidates = []
        for i in range(2**n):
            bs = format(i, f"0{n}b")
            x = np.array([int(b) for b in bs])
            if np.sum(x) == budget:
                sel_idx = np.where(x == 1)[0]
                w = np.zeros(n)
                w[sel_idx] = 1.0 / budget
                p_ret = float(np.dot(w, mu))
                p_vol = float(np.sqrt(np.dot(w, np.dot(sigma, w))))
                p_sharpe = (p_ret - 0.065) / p_vol if p_vol > 0 else 0
                sel_names = [candidate_tickers[idx].replace(".NS", "") for idx in sel_idx]
                valid_candidates.append({
                    "bitstring": bs,
                    "stocks": sel_names,
                    "weights": {name: round(100.0/budget, 1) for name in sel_names},
                    "probability": round(float(probs[i]) * 100, 2),
                    "annual_return": round(p_ret * 100, 2),
                    "annual_volatility": round(p_vol * 100, 2),
                    "sharpe_ratio": round(p_sharpe, 2)
                })

        valid_candidates.sort(key=lambda item: item["probability"], reverse=True)
        best_portfolio = valid_candidates[0] if valid_candidates else None

        return {
            "best_portfolio": best_portfolio,
            "all_valid_options": valid_candidates[:5],
            "quantum_qubits": n,
            "qaoa_layers": p_layers
        }
