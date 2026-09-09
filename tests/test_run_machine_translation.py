import numpy as np

from project import run_machine_translation


class FakeTokenizer:
    vocab = {"<eos_en>": 9}

    def __init__(self, expected_generated_ids=(4, 5), decoded_text="Hello"):
        self.expected_generated_ids = list(expected_generated_ids)
        self.decoded_text = decoded_text

    def __call__(self, text):
        assert text == "Hallo<eos_de>"
        return {"input_ids": [1, 2]}

    def decode(self, token_ids):
        assert token_ids == self.expected_generated_ids
        return self.decoded_text


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def to_numpy(self):
        return np.asarray(self.values)


class FakeOutput:
    def __init__(self, next_token_id):
        self.values = np.zeros((1, 1, 10))
        self.values[0, -1, next_token_id] = 1.0

    def to_numpy(self):
        return self.values


class FakeModel:
    def __init__(self, predictions=(4, 5, 9)):
        self.predictions = iter(predictions)
        self.eval_called = False
        self.inputs = []

    def eval(self):
        self.eval_called = True

    def forward(self, input_ids):
        self.inputs.append(input_ids.to_numpy().tolist())
        return FakeOutput(next(self.predictions))


def test_generate_uses_greedy_decoding_until_target_eos(monkeypatch):
    # Keep this a unit test: generation behavior should not depend on a real
    # MiniTorch backend or GPU kernels.
    monkeypatch.setattr(
        run_machine_translation.minitorch,
        "tensor",
        lambda values, backend: FakeTensor(values),
    )

    model = FakeModel()
    result = run_machine_translation.generate(
        model=model,
        examples=[{"de": "Hallo", "en": "Hello"}],
        src_key="de",
        tgt_key="en",
        tokenizer=FakeTokenizer(),
        model_max_length=10,
        backend=None,
        desc="test",
    )

    assert result == ["Hello"]
    assert model.eval_called
    assert model.inputs == [
        [[1, 2]],
        [[1, 2, 4]],
        [[1, 2, 4, 5]],
    ]


def test_generate_stops_when_first_prediction_is_target_eos(monkeypatch):
    monkeypatch.setattr(
        run_machine_translation.minitorch,
        "tensor",
        lambda values, backend: FakeTensor(values),
    )
    model = FakeModel(predictions=[9])

    result = run_machine_translation.generate(
        model=model,
        examples=[{"de": "Hallo", "en": ""}],
        src_key="de",
        tgt_key="en",
        tokenizer=FakeTokenizer(expected_generated_ids=[], decoded_text=""),
        model_max_length=10,
        backend=None,
        desc="test",
    )

    assert result == [""]
    assert model.inputs == [[[1, 2]]]


def test_generate_decodes_when_length_equals_maximum(monkeypatch):
    monkeypatch.setattr(
        run_machine_translation.minitorch,
        "tensor",
        lambda values, backend: FakeTensor(values),
    )
    model = FakeModel(predictions=[4, 5])

    result = run_machine_translation.generate(
        model=model,
        examples=[{"de": "Hallo", "en": "Hello"}],
        src_key="de",
        tgt_key="en",
        tokenizer=FakeTokenizer(expected_generated_ids=[4, 5], decoded_text="Hello"),
        model_max_length=3,
        backend=None,
        desc="test",
    )

    assert result == ["Hello"]
    assert model.inputs == [
        [[1, 2]],
        [[1, 2, 4]],
    ]
