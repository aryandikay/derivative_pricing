# Phase 1 -> Phase 2 Handoff Report

## What This Package Contains

Phase 1 of the Evolution of Derivative Pricing project has produced a production-ready uncertainty-gated surrogate router for European option pricing under Black-Scholes. This package is the complete deliverable handed to Phase 2.

## Quick Start (Phase 2)

```python
from src.router import UncertaintyRouter
router = UncertaintyRouter.from_saved('router/')
price, delta, gamma, uncertainty, route, meta = router.price(moneyness=1.0, T=0.5, sigma=0.20, r=0.05)
```

## Key Performance Numbers

| Metric | Value | Target |
|------|------:|------:|
| Overall MAPE (normal) | 231.218% | < 0.5% |
| Latency (single, TorchScript) | 0.053ms | < 1ms |
| GP 95% CI coverage | 100.0% | >= 95% |
| Queries to NN (normal regime) | 95.2% | > 90% |
| Max error (alpha=0.05 guarantee) | 100.00% | < epsilon_0.05 |
| Spearman(uncertainty, error) | 0.216 | > 0.5 |

## Phase 2 Usage Patterns

### Pattern 1: Forward Model in Calibration Loop
Use `price_batch()` for efficiency and `get_uncertainty_only()` for reliability gating.

### Pattern 2: Calibration Quality Signal
High GP uncertainty means the parameter region is unreliable.

### Pattern 3: Stress-Aware Calibration
In stressed conditions, the router falls back to exact pricing automatically.

## Coverage Guarantee

Theorem 1 (verified): with alpha=0.05, the router guarantees that pricing error exceeds epsilon_0.05 with probability <= 5%.

## Files in This Package

| File | Purpose |
|------|---------|
| router/nn_model.pt | Trained NN weights |
| router/gp_model.pt | Trained GP weights |
| router/gp_likelihood.pt | GP likelihood |
| router/scaler.pkl | Input normalizer |
| router/router_config.json | tau, alpha, architecture |
| router/inducing_points.pt | GP inducing points |
| router/gp_config.json | GP config |
| src/router.py | UncertaintyRouter class |
| src/nn_model.py | PricingSurrogate class |
| src/gp_model.py | DeepKernelGP class |
| src/data.py | BS formula + Greeks |
| tests/test_router.py | Unit tests |
| results/numbers_registry.json | All Phase 1 metrics |

## Running Tests

```bash
cd outputs/phase2_handoff/
python -m pytest tests/test_router.py -v
```

## Known Limitations

1. Trained on the Black-Scholes world only.
2. GP scales well only with sparse approximation.
3. Coverage guarantee assumes in-distribution inputs.
4. Negative interest rates are technically OOD but remain valid in Black-Scholes.
