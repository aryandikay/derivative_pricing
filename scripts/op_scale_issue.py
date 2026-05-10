import numpy as np
y_train = np.load('data/processed/y_train.npy')
print(f"Price range in training data: {y_train[:,0].min():.4f} to {y_train[:,0].max():.4f}")
print(f"Price mean: {y_train[:,0].mean():.4f}")
# If prices are between 0 and 1 in the training data, scaling is fine
# If they are raw BS prices (ranging 0.001 to 0.5+), no output scaling was done
# A prediction of 0.0000110 when true price is 0.069 suggests wrong weights, not scaling