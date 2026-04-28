"""Black-Scholes Option Pricing - Ground Truth Implementation"""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S_over_K, T, r, sigma):
    """
    Black-Scholes call option price.
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness = S/K (spot price / strike price)
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (annualized, decimal form e.g., 0.05 for 5%)
    sigma : float
        Volatility (annualized, decimal form e.g., 0.2 for 20%)
    
    Returns
    -------
    float or array
        Call option price normalized by strike (V/K)
    
    Notes
    -----
    Formula:
        C/K = S/K * N(d1) - exp(-r*T) * N(d2)
        where d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
              d2 = d1 - sigma*sqrt(T)
    """
    # Guard against T=0 edge case
    T = np.maximum(T, 1e-8)
    
    # Calculate d1 and d2
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Call price
    price = S_over_K * norm.cdf(d1) - np.exp(-r * T) * norm.cdf(d2)
    return price


def black_scholes_put(S_over_K, T, r, sigma):
    """
    Black-Scholes put option price.
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness = S/K
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns
    -------
    float or array
        Put option price normalized by strike (V/K)
    """
    T = np.maximum(T, 1e-8)
    
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = np.exp(-r * T) * norm.cdf(-d2) - S_over_K * norm.cdf(-d1)
    return price


def bs_delta_call(S_over_K, T, r, sigma):
    """
    Delta of a call option (dC/dS, normalized).
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns
    -------
    float or array
        Delta (between 0 and 1 for calls)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_delta_put(S_over_K, T, r, sigma):
    """
    Delta of a put option.
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns
    -------
    float or array
        Delta (between -1 and 0 for puts)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def bs_gamma(S_over_K, T, r, sigma):
    """
    Gamma of an option (d²C/dS²).
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns
    -------
    float or array
        Gamma (always positive, same for calls and puts)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S_over_K * sigma * np.sqrt(T))


def bs_vega(S_over_K, T, r, sigma):
    """
    Vega of an option (dC/dsigma).
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns
    -------
    float or array
        Vega (sensitivity to 1% change in volatility, normalized by K)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S_over_K * norm.pdf(d1) * np.sqrt(T) / 100


def bs_theta_call(S_over_K, T, r, sigma):
    """
    Theta of a call option (dC/dT).
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns
    -------
    float or array
        Theta (per day, normalized by K)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta = (-S_over_K * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * np.exp(-r * T) * norm.cdf(d2))
    
    return theta / 365


def bs_rho_call(S_over_K, T, r, sigma):
    """
    Rho of a call option (dC/dr).
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns
    -------
    float or array
        Rho (sensitivity to 1% change in interest rate, normalized by K)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return T * np.exp(-r * T) * norm.cdf(d2) / 100
