import torch

weights = torch.load('models/nn/best_model.pt', map_location='cpu')

# Find the final layer
final_layer_key = [k for k in weights.keys() if 'weight' in k][-1]
output_size = weights[final_layer_key].shape[0]
print(f"Saved model output size: {output_size}")
# Should be 3 (price + delta + gamma)
# If it shows 1, this is the cause