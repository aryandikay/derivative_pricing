import numpy as np
from src.router import UncertaintyRouter

np.random.seed(42)

data = np.load('data/processed/test.npz')
X_test = data['X_test'] if 'X_test' in data else data['X_test_original']
y_test = data['y_test']
router = UncertaintyRouter.from_saved('outputs/router_v1/', device=None)
router.tau = 0.01

idx = np.random.choice(len(X_test), 1000, replace=False)
errors = []
uncertainties = []
for i in idx:
    x = X_test[i]
    true = y_test[i, 0]
    price, _, _, unc, route, _ = router.price(x[0], x[1], x[2], x[3])
    errors.append(abs(price - true))
    uncertainties.append(unc)

errors = np.array(errors)
uncertainties = np.array(uncertainties)
order = np.argsort(uncertainties)
errors_sorted = errors[order]
n = len(errors)
low = errors_sorted[:n//4]
mid = errors_sorted[n//4:n//2]
high = errors_sorted[-n//4:]

print(np.mean(low))
print(np.mean(mid))
print(np.mean(high))
