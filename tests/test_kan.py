import pytest
import numpy as np
import torch
import pandas as pd
from models.kan_model import kan_net

def make_synthetic_data(n=50, d=3):
    np.random.seed(42)
    X = np.random.randn(n, d)
    T = np.random.binomial(1, 0.5, n)
    y = X @ np.array([1.5, -2.0, 0.5]) + T * 2.0 + np.random.randn(n)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(d)])
    return df, np.reshape(y, (-1,1)), np.reshape(T, (-1,1))

# Helper to get dims for each model type
MODEL_CONFIGS = {
    'slearner':   lambda d, h: [d] + (h if isinstance(h, list) else [h]) + [1],
    'tarnet':     lambda d, h: [d] + (h if isinstance(h, list) else [h]) + [2],
    'dragonnet':  lambda d, h: [d] + (h if isinstance(h, list) else [h]) + [3],
    'tlearner':   lambda d, h: [d] + (h if isinstance(h, list) else [h]) + [2],  # tlearner uses two networks
}

MODEL_OUTPUTS = {'slearner': 1, 'tarnet': 2, 'dragonnet': 3, 'tlearner': 2}

@pytest.mark.parametrize("model_name,hidden_dims,mult_kan", [
    ('slearner', 5, False),
    ('slearner', [5, 5], True),
    ('tarnet', 5, False),
    ('tarnet', [5, 5], True),
    ('dragonnet', 5, False),
    ('dragonnet', [5, 5], True),
    ('tlearner', 5, False),
    ('tlearner', [5, 5], True),
])
def test_kan_model_structure_and_fit(model_name, hidden_dims, mult_kan):
    X, y, T = make_synthetic_data(n=30, d=3)
    d = X.shape[1]
    dims = MODEL_CONFIGS[model_name](d, hidden_dims)
    # For tlearner, instantiate as in metrics.py: two networks
    if model_name == 'tlearner':
        model = kan_net(
            model_name=model_name,
            dims=dims,
            grid=6,
            k=2,
            sparse_init=True,
            try_gpu=False
        )
        # tlearner: model.model is a list of two KANs
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        assert len(model.model.model) == 2
        y_pred0 = model.model.model[0](X_tensor)
        y_pred1 = model.model.model[1](X_tensor)
        assert y_pred0.shape == (X.shape[0], 1)
        assert y_pred1.shape == (X.shape[0], 1)
    else:
        model = kan_net(
            model_name=model_name,
            dims=dims,
            grid=6,
            k=2,
            sparse_init=True,
            try_gpu=False
        )
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_pred = model.model(X_tensor)
        assert y_pred.shape[0] == X.shape[0]
        assert y_pred.shape[1] == MODEL_OUTPUTS[model_name]
    # Fit test (if fit method exists)
    if hasattr(model, 'fit'):
        X = X.to_numpy()

        results = model.fit(X, y, T, X, y, T, steps=10, batch=10, lr=0.01, verbose=0, lamb=0.01, lamb_entropy=0.1)
        assert 'train_loss' in results
        assert len(results['train_loss']) > 0
        # assert results['train_loss'][-1] <= results['train_loss'][0]

    # check predict method
    if hasattr(model, 'predict'):
        preds = model.predict(X, T)

        if model_name == 'dragonnet':
            assert preds['ps_pred'].shape == (X.shape[0], )
        assert preds['y_pred_f'].shape == (X.shape[0],)
