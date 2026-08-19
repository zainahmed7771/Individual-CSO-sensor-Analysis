from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

def tail_values(durations, threshold: float = 240.0) -> np.ndarray:
    x = np.asarray(durations, dtype=float)
    return x[np.isfinite(x) & (x > threshold)]

def fit_tail(durations, threshold: float = 240.0) -> dict:
    tail = tail_values(durations, threshold)
    if len(tail) < 10:
        raise ValueError("at least 10 events strictly above threshold are required")
    log_x = np.log(tail); log_u = math.log(threshold)
    def nll(par):
        log_lam, beta = float(par[0]), float(par[1])
        z = beta * (log_lam + log_x); zu = beta * (log_lam + log_u)
        if np.any(z > 700) or zu > 700: return np.finfo(float).max
        return -float(np.sum(math.log(beta) + log_lam + (beta-1)*(log_lam+log_x) - np.exp(z) + math.exp(zu)))
    result = minimize(nll, [math.log(1/np.median(tail)), .18], method="L-BFGS-B", bounds=[(-50,20),(.01,2.0)])
    return {"beta": float(result.x[1]), "lambda_per_minute": float(math.exp(result.x[0])), "n_tail": int(len(tail)), "success": bool(result.success)}
