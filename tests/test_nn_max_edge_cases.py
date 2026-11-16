import numpy as np
import pytest

import minitorch


def cpu_tensor(values, requires_grad=False):
    backend = minitorch.TensorBackend(minitorch.FastOps)
    return minitorch.tensor_from_numpy(
        np.asarray(values, dtype=np.float32),
        backend=backend,
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("dim", [0, 1])
def test_max_below_previous_initial_value(dim):
    values = np.array([[-2e9, -3e9, -4e9], [-5e9, -6e9, -7e9]], dtype=np.float32)
    result = minitorch.max(cpu_tensor(values), dim)
    np.testing.assert_array_equal(
        result.to_numpy(), np.max(values, axis=dim, keepdims=True)
    )


def test_softmax_with_extreme_negative_equal_logits():
    x = cpu_tensor([[-1e10, -1e10, -1e10]])
    result = minitorch.softmax(x, dim=1)
    np.testing.assert_allclose(result.to_numpy(), [[1 / 3, 1 / 3, 1 / 3]], atol=1e-6)


def test_max_backward_below_previous_initial_value():
    x = cpu_tensor([[-3e9, -2e9, -4e9]], requires_grad=True)
    minitorch.max(x, dim=1).sum().backward()
    np.testing.assert_array_equal(x.grad.to_numpy(), [[0, 1, 0]])
