import numpy as np
import pytest

import minitorch
from project import run_machine_translation


@pytest.fixture
def backend():
    return minitorch.TensorBackend(minitorch.FastOps)


def make_batch(weights, backend):
    shape = np.asarray(weights).shape
    return {
        "input_ids": minitorch.tensor_from_numpy(np.ones(shape), backend=backend),
        "labels": minitorch.tensor_from_numpy(np.ones(shape), backend=backend),
        "label_token_weights": minitorch.tensor_from_numpy(
            np.asarray(weights), backend=backend
        ),
    }


def model_must_not_run(**kwargs):
    pytest.fail("A batch with no target tokens must be rejected before model execution")


@pytest.mark.parametrize("shape", [(1, 3), (2, 3)])
def test_loss_rejects_empty_target_mask_before_model_execution(shape, backend):
    batch = make_batch(np.zeros(shape), backend)
    with pytest.raises(ValueError, match="no target tokens"):
        run_machine_translation.loss_fn(batch, model_must_not_run)


def test_loss_rejects_batch_whose_targets_are_truncated(backend):
    class Tokenizer:
        vocab = {"<pad>": 0}

        def __call__(self, text):
            return {"input_ids": [1, 2, 3, 4, 5] if text.endswith("<eos_de>") else [6, 7]}

    batch = run_machine_translation.collate_batch(
        examples=[{"de": "long source", "en": "target"}],
        src_key="de", tgt_key="en", tokenizer=Tokenizer(),
        model_max_length=4, backend=backend,
    )
    assert batch["label_token_weights"].sum().item() == 0
    with pytest.raises(ValueError, match="no target tokens"):
        run_machine_translation.loss_fn(batch, model_must_not_run)


@pytest.mark.parametrize("weights", [[1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0.5, 0]])
def test_loss_normalization_and_gradients_for_nonempty_masks(weights, backend, monkeypatch):
    batch = make_batch([weights], backend)
    losses_np = np.array([2, 4, 6, 8], dtype=np.float32)
    losses = minitorch.tensor_from_numpy(losses_np, backend=backend, requires_grad=True)
    logits = minitorch.zeros((1, 4, 3), backend=backend)
    # Loss implementation is a separate assignment; isolate normalization here.
    monkeypatch.setattr(minitorch.nn, "softmax_loss", lambda **kwargs: losses)

    result = run_machine_translation.loss_fn(batch, lambda idx: logits)
    weights_np = np.asarray(weights, dtype=np.float32)
    expected = (losses_np * weights_np).sum() / weights_np.sum()
    np.testing.assert_allclose(result.to_numpy(), [expected], atol=1e-6)

    result.backward()
    np.testing.assert_allclose(
        losses.grad.to_numpy(), weights_np / weights_np.sum(), atol=1e-6
    )
