# Abstract

We propose an uncertainty-gated router for option pricing. The router uses a calibrated GP to decide when to trust a neural surrogate and when to fall back to exact Black-Scholes pricing. The method verifies Theorem 1 across alpha values and reduces stress-case MAPE from 29313.59% to 0.00%.

# 4 Results

The surrogate achieves 231.22% overall MAPE. GP 95% empirical coverage is 99.993%. Uncertainty-error alignment on the main grid is rho=0.655. Under stress, the worst NN case is 29313.59% MAPE while the router reduces the average stress error by 100.0%.

Normal scenario router MAPE: 0.0004%.
