import pytest
import numpy as np
import pandas as pd
from models.mlp_model import mlp_net
from utils.utils_results import get_dims_mlp

def make_synthetic_data(n=50, d=3):
    np.random.seed(42)
    X = np.random.randn(n, d)
    T = np.random.binomial(1, 0.5, n)
    y = X @ np.array([1.5, -2.0, 0.5]) + T * 2.0 + np.random.randn(n)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(d)])
    return df, y, T


@pytest.mark.parametrize("model_name,hidden_dims", [
    ('slearner', [5]),
    ('slearner', [5, 5]),
    ('tlearner', [5]),
    ('tlearner', [5, 5]),
    ('dragonnet', [[5, 5], [5], []]),
    ('tarnet', [[5, 5], [5]]),
])
def test_mlp_model_structure_and_fit(model_name, hidden_dims):
    X, y, T = make_synthetic_data(n=30, d=3)
    d = X.shape[1]
    dims = get_dims_mlp(model_name, d, hidden_dims)
    model = mlp_net(
        model_name=model_name,
        dims=dims,
        seed=42,
        try_gpu=False
    )
    # Fit test
    X = X.to_numpy()
    y = np.reshape(y, (-1, 1))
    T = np.reshape(T, (-1, 1))

    results = model.fit(X, y, T, X, y, T, early_stop=True, patience=5, steps=10, batch=10, lr=0.01, verbose=0)
    assert 'train_loss' in results
    assert len(results['train_loss']) > 0
    assert results['train_loss'][-1] <= results['train_loss'][0]
    # Predict test

    pred = model.predict(X, T)
    assert 'y_pred_0' in pred and 'y_pred_1' in pred
    assert pred['y_pred_0'].shape == (X.shape[0],)
    assert pred['y_pred_1'].shape == (X.shape[0],)
    if model_name == 'dragonnet':
        assert 'ps_pred' in pred
        assert pred['ps_pred'].shape == (X.shape[0],)
    if model_name == 'slearner':
        assert 'y_pred_f' in pred and 'y_pred_cf' in pred
        assert pred['y_pred_f'].shape == (X.shape[0],)
        assert pred['y_pred_cf'].shape == (X.shape[0],)
