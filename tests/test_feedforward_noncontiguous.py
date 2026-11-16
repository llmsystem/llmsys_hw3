import numpy as np
import pytest

import minitorch
from minitorch.transformer import FeedForward


class Identity(minitorch.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("permuted", [False, True])
def test_feedforward_preserves_values_and_gradients_for_tensor_layouts(permuted):
    # Isolate FeedForward's layout handling from the student-implemented layers.
    layer = FeedForward.__new__(FeedForward)
    minitorch.Module.__init__(layer)
    layer.linear_in = Identity()
    layer.linear_out = Identity()
    layer.dropout = Identity()

    values = np.linspace(-2, 2, 24, dtype=np.float32).reshape(2, 4, 3)
    backend = minitorch.TensorBackend(minitorch.FastOps)
    base = minitorch.tensor_from_numpy(values, backend=backend, requires_grad=True)
    x = base.permute(0, 2, 1) if permuted else base
    logical_values = values.transpose(0, 2, 1) if permuted else values

    result = layer(x)
    a = np.sqrt(2 / np.pi) * (logical_values + 0.044715 * logical_values ** 3)
    expected = 0.5 * logical_values * (1 + np.tanh(a))
    np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6, rtol=1e-5)
    assert result.shape == x.shape

    result.sum().backward()
    a = np.sqrt(2 / np.pi) * (values + 0.044715 * values ** 3)
    expected_grad = (
        0.5 * (1 + np.tanh(a))
        + 0.5 * values * (1 - np.tanh(a) ** 2)
        * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * values ** 2)
    )
    np.testing.assert_allclose(base.grad.to_numpy(), expected_grad, atol=1e-6, rtol=1e-5)
