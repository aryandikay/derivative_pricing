from src.router import load_components, UncertaintyRouter
import numpy as np
comps = load_components()
router = UncertaintyRouter.from_saved('outputs/router_v1')
# pick samples
rng_idx = np.random.choice(len(comps['X_test']), size=100, replace=False)
sample_X = comps['X_test'][rng_idx]
prices, deltas, gammas, uncertainties, routes = router.price_batch(sample_X, batch_size=100)
print('prices finite:', np.all(np.isfinite(prices)))
print('prices >0:', np.all(prices>0))
print('min, max, nan count, neg count:', float(np.nanmin(prices)), float(np.nanmax(prices)), int(np.isnan(prices).sum()), int(np.sum(prices<=0)))
print('routes unique:', set(routes))
print('sample prices[:10]', prices[:10])
