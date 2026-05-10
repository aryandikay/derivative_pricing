import numpy as np
import torch
import joblib
from sklearn.ensemble import RandomForestRegressor
from src.nn_model import PricingSurrogate


def main():
    data = np.load('data/processed/train.npz')
    X = data['X_train_scaled']
    y = data['y_train'][:, 0]

    model = PricingSurrogate(hidden_dim=128, n_layers=4, activation='silu')
    nn_state = torch.load('models/nn/final_stable_model.pt', map_location='cpu')
    if isinstance(nn_state, dict) and 'model_state_dict' in nn_state:
        nn_state = nn_state['model_state_dict']
    model.load_state_dict(nn_state)
    model.eval()

    with torch.no_grad():
        pred = model(torch.FloatTensor(X))[:, 0].numpy()

    y_unc = np.abs(pred - y)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y_unc)

    joblib.dump(rf, 'models/uncertainty_model.pkl')
    print('Uncertainty model trained and saved to models/uncertainty_model.pkl')


if __name__ == '__main__':
    main()
