import torch

router_weights = torch.load('outputs/router_v1/nn_model.pt', 
                             map_location='cpu')
best_weights   = torch.load('models/nn/best_model.pt', 
                             map_location='cpu')

# Check if they are identical
all_same = all(
    torch.allclose(router_weights[k], best_weights[k]) 
    for k in router_weights.keys()
)
print(f"Weights identical: {all_same}")

# Check first layer weights as quick check
first_key = list(router_weights.keys())[0]
print(f"Router first layer mean: {router_weights[first_key].mean():.6f}")
print(f"Best   first layer mean: {best_weights[first_key].mean():.6f}")