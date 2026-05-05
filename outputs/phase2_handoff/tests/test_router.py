# tests/test_router.py
# Run with: python -m pytest tests/test_router.py -v
# All tests must pass before Phase 2 uses the router.

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import UncertaintyRouter
from src.data import black_scholes_call, bs_delta


class TestUncertaintyRouter:

    @pytest.fixture(scope='class')
    def router(self):
        return UncertaintyRouter.from_saved('router/')

    def test_loads_correctly(self, router):
        assert router is not None
        assert router.tau > 0
        assert router.alpha == 0.05

    def test_single_prediction_returns_correct_types(self, router):
        price, delta, gamma, unc, route, meta = router.price(1.0, 0.5, 0.20, 0.05)
        assert isinstance(price, float)
        assert isinstance(delta, float)
        assert isinstance(gamma, float)
        assert isinstance(unc, float)
        assert route in ['nn', 'exact']
        assert isinstance(meta, dict)

    def test_atm_option_accuracy(self, router):
        price, delta, gamma, unc, route, meta = router.price(1.0, 0.5, 0.20, 0.05)
        bs_true = black_scholes_call(1.0, 0.5, 0.05, 0.20)
        rel_error = abs(price - bs_true) / bs_true
        assert rel_error < 0.005, f"ATM error {rel_error:.4f} exceeds 0.5%"

    def test_no_arbitrage_constraints(self, router):
        test_inputs = [
            (0.7, 1.0, 0.2, 0.05),
            (1.0, 1.0, 0.2, 0.05),
            (1.3, 1.0, 0.2, 0.05),
        ]
        for m, T, sig, r in test_inputs:
            _, delta, gamma, _, _, _ = router.price(m, T, sig, r)
            assert 0 <= delta <= 1, f"Delta {delta} outside [0,1]"
            assert gamma >= 0, f"Gamma {gamma} is negative"

    def test_stress_input_routed_to_exact(self, router):
        _, _, _, unc, route, _ = router.price(0.7, 0.02, 1.5, 0.01)
        if route == 'nn':
            assert unc < router.tau
        assert route in ['nn', 'exact']

    def test_batch_matches_single(self, router):
        X = np.array([
            [1.0, 0.5, 0.2, 0.05],
            [0.9, 0.25, 0.3, 0.03],
            [1.1, 1.0, 0.15, 0.07],
        ])
        batch_prices, _, _, _, _ = router.price_batch(X)
        for i, x in enumerate(X):
            single_price, _, _, _, _, _ = router.price(*x)
            np.testing.assert_allclose(batch_prices[i], single_price, rtol=1e-4)

    def test_routing_statistics_tracked(self, router):
        router._reset_stats()
        for _ in range(100):
            router.price(1.0, 0.5, 0.2, 0.05)
        stats = router.routing_stats()
        assert stats['total_queries'] == 100
        assert stats['nn_queries'] + stats['exact_queries'] == 100

    def test_uncertainty_only_faster_than_full_price(self, router):
        import time
        n = 100
        start = time.perf_counter()
        for _ in range(n):
            router.get_uncertainty_only(1.0, 0.5, 0.2, 0.05)
        unc_time = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(n):
            router.price(1.0, 0.5, 0.2, 0.05)
        full_time = time.perf_counter() - start
        print(f'Uncertainty-only: {unc_time/n*1000:.3f}ms/call')
        print(f'Full price:       {full_time/n*1000:.3f}ms/call')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
