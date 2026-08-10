import json
import random
from types import SimpleNamespace

import numpy as np
import torch

from src.absa.data.schema import AspectExample
from src.absa.labels import LABEL_TO_ID
from src.absa.training import atae_lstm as atae_training
from src.absa.training import distilbert as distilbert_training
from src.absa.training import target_gru as gru_training
from src.absa.training import target_lstm as target_training
from src.absa.training import text_cnn as cnn_training
from src.absa.training.atae_lstm import save_artifact as save_atae_artifact
from src.absa.training.atae_lstm import train_atae_lstm
from src.absa.training.common import (
    BestCheckpoint,
    seed_everything,
    training_diagnostic,
    validate_training_parameters,
)
from src.absa.training.target_lstm import save_artifact as save_target_artifact
from src.absa.training.target_lstm import train_target_lstm
from src.absa.training.target_gru import save_artifact as save_gru_artifact
from src.absa.training.target_gru import train_target_gru
from src.absa.training.text_cnn import save_artifact as save_cnn_artifact
from src.absa.training.text_cnn import train_text_cnn


def _row(sentence_id: str, review: str, aspect: str, label: str, source: str) -> AspectExample:
    start = review.index(aspect)
    return AspectExample(
        sentence_id=sentence_id,
        review_raw=review,
        aspect=aspect,
        aspect_from=start,
        aspect_to=start + len(aspect),
        label=label,
        source_split=source,
        offset_valid=True,
    )


def _tiny_rows() -> tuple[list[AspectExample], list[AspectExample]]:
    train = [
        _row("train-1", "great food", "food", "positive", "train"),
        _row("train-2", "slow service", "service", "negative", "train"),
        _row("train-3", "average menu", "menu", "neutral", "train"),
        _row("train-4", "clean restaurant", "restaurant", "positive", "train"),
        _row("train-5", "noisy room", "room", "negative", "train"),
        _row("train-6", "ordinary desserts", "desserts", "neutral", "train"),
    ]
    test = [
        _row("test-1", "excellent food", "food", "positive", "test"),
        _row("test-2", "poor service", "service", "negative", "test"),
        _row("test-3", "standard menu", "menu", "neutral", "test"),
    ]
    return train, test


def test_seed_everything_reproduces_python_numpy_torch_and_loader_generator() -> None:
    first_generator = seed_everything(17)
    first_python = random.random()  # noqa: S311 - deterministic non-cryptographic RNG test
    first = (first_python, np.random.random(), torch.rand(1), torch.rand(1, generator=first_generator))
    second_generator = seed_everything(17)
    second_python = random.random()  # noqa: S311 - deterministic non-cryptographic RNG test
    second = (second_python, np.random.random(), torch.rand(1), torch.rand(1, generator=second_generator))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])


def test_best_checkpoint_stops_and_restores_selected_state() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = BestCheckpoint(patience=2)
    model.weight.data.fill_(1)
    assert tracker.update(model, 0.8, epoch=1) is False
    model.weight.data.fill_(2)
    assert tracker.update(model, 0.7, epoch=2) is False
    model.weight.data.fill_(3)
    assert tracker.update(model, 0.6, epoch=3) is True
    tracker.restore(model)
    assert model.weight.item() == 1
    assert tracker.best_epoch == 1


class _RecordingBestCheckpoint(BestCheckpoint):
    latest = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.selected_state = None
        self.restore_calls = 0
        _RecordingBestCheckpoint.latest = self

    def update(self, model: torch.nn.Module, score: float, epoch: int) -> bool:
        previous_best = self.best_epoch
        should_stop = super().update(model, score, epoch)
        if self.best_epoch != previous_best:
            self.selected_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        return should_stop

    def restore(self, model: torch.nn.Module) -> None:
        super().restore(model)
        self.restore_calls += 1


def _controlled_metrics(scores: list[float]):
    remaining = iter(scores)
    calls = 0

    def compute(_labels, _predictions):
        nonlocal calls
        calls += 1
        tracker = _RecordingBestCheckpoint.latest
        if calls > 3:
            assert tracker is not None
            assert tracker.restore_calls == 1
        return {"macro_f1": next(remaining)}

    return compute


def _assert_state_matches_selected(model, selected_state) -> None:
    assert selected_state is not None
    assert model.state_dict().keys() == selected_state.keys()
    assert all(
        torch.equal(model.state_dict()[name].detach().cpu(), selected_state[name])
        for name in selected_state
    )


def test_training_diagnostic_recommends_multi_seed_only_for_material_overfitting() -> None:
    stable = [
        {"epoch": 1, "train_loss": 1.0, "development_macro_f1": 0.6},
        {"epoch": 2, "train_loss": 0.8, "development_macro_f1": 0.59},
    ]
    overfit = [
        {"epoch": 1, "train_loss": 1.0, "development_macro_f1": 0.7},
        {"epoch": 2, "train_loss": 0.7, "development_macro_f1": 0.6},
    ]
    assert training_diagnostic(stable)["multi_seed_recommended"] is False
    assert training_diagnostic(overfit)["multi_seed_recommended"] is True


def test_invalid_training_parameters_fail_before_a_run_starts() -> None:
    try:
        validate_training_parameters(
            epochs=0,
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0,
            max_length=80,
        )
    except ValueError as error:
        assert "epochs" in str(error)
    else:
        raise AssertionError("A run without epochs cannot select a checkpoint")

    try:
        validate_training_parameters(
            epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0,
            max_length=80,
            patience=0,
        )
    except ValueError as error:
        assert "patience" in str(error)
    else:
        raise AssertionError("Invalid patience must fail before training setup")


def test_neural_trainers_return_and_persist_reproducibility_metadata(tmp_path) -> None:
    train_rows, test_rows = _tiny_rows()
    trainers = (
        (train_target_lstm, save_target_artifact, "target_lstm.pt", "target_lstm_metrics.json"),
        (train_target_gru, save_gru_artifact, "target_gru.pt", "target_gru_metrics.json"),
        (train_text_cnn, save_cnn_artifact, "text_cnn.pt", "text_cnn_metrics.json"),
        (train_atae_lstm, save_atae_artifact, "atae_lstm.pt", "atae_lstm_metrics.json"),
    )

    for trainer, saver, checkpoint_name, metrics_name in trainers:
        model, vocab, result = trainer(
            train_rows,
            test_rows,
            epochs=2,
            batch_size=2,
            seed=9,
            patience=1,
            max_length=8,
        )
        assert result["selection_metric"] == "development_macro_f1"
        assert result["best_epoch"] in {1, 2}
        assert result["config"]["seed"] == 9
        assert result["config"]["weight_decay"] > 0
        assert len(result["history"]) in {1, 2}
        assert "train_loss" in result["history"][0]

        output_dir = tmp_path / result["config"]["model"]
        saver(model, vocab, result, output_dir)
        checkpoint = torch.load(output_dir / checkpoint_name, weights_only=True)
        persisted = json.loads((output_dir / metrics_name).read_text())
        assert checkpoint["best_epoch"] == result["best_epoch"]
        assert checkpoint["config"]["seed"] == 9
        assert persisted["history"] == result["history"]


def test_neural_trainers_restore_early_winner_before_evaluation_and_saving(
    monkeypatch,
    tmp_path,
) -> None:
    train_rows, test_rows = _tiny_rows()
    trainers = (
        (target_training, train_target_lstm, save_target_artifact, "target_lstm.pt"),
        (gru_training, train_target_gru, save_gru_artifact, "target_gru.pt"),
        (cnn_training, train_text_cnn, save_cnn_artifact, "text_cnn.pt"),
        (atae_training, train_atae_lstm, save_atae_artifact, "atae_lstm.pt"),
    )

    for module, trainer, saver, checkpoint_name in trainers:
        monkeypatch.setattr(module, "BestCheckpoint", _RecordingBestCheckpoint)
        monkeypatch.setattr(module, "compute_metrics", _controlled_metrics([0.9, 0.8, 0.7, 0.9, 0.5]))
        model, vocab, result = trainer(
            train_rows,
            test_rows,
            epochs=4,
            batch_size=2,
            seed=31,
            patience=2,
            max_length=8,
        )
        tracker = _RecordingBestCheckpoint.latest
        assert tracker is not None
        assert result["best_epoch"] == 1
        assert result["stopped_early"] is True
        assert tracker.restore_calls == 1
        _assert_state_matches_selected(model, tracker.selected_state)

        output_dir = tmp_path / result["config"]["model"]
        saver(model, vocab, result, output_dir)
        saved = torch.load(output_dir / checkpoint_name, weights_only=True)
        assert all(
            torch.equal(saved["state_dict"][name], tracker.selected_state[name])
            for name in tracker.selected_state
        )


class _TinyPairDataset(torch.utils.data.Dataset):
    def __init__(self, _tokenizer, rows, max_length: int) -> None:
        assert max_length == 16
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        label = LABEL_TO_ID[self.rows[index].label]
        return {
            "input_ids": torch.tensor([label, 1], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class _TinyDistilBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 3)
        self.config = SimpleNamespace(id2label={}, label2id={})

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.classifier(input_ids.float()))

    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")


class _TinyTokenizer:
    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}\n")


class _TinyDistilBertFactory:
    @staticmethod
    def from_pretrained_absa(_model_name: str) -> _TinyDistilBert:
        return _TinyDistilBert()


def test_distilbert_training_is_seeded_and_records_complete_run_config(monkeypatch, tmp_path) -> None:
    train_rows, test_rows = _tiny_rows()
    monkeypatch.setattr(distilbert_training, "AspectPairDataset", _TinyPairDataset)
    monkeypatch.setattr(distilbert_training, "ABSADistilBERT", _TinyDistilBertFactory)
    monkeypatch.setattr(
        distilbert_training.AutoTokenizer,
        "from_pretrained",
        lambda _model_name: _TinyTokenizer(),
    )

    kwargs = {
        "epochs": 2,
        "batch_size": 2,
        "seed": 23,
        "patience": 1,
        "max_length": 16,
        "model_name": "tiny-distilbert",
        "device": torch.device("cpu"),
    }
    first_model, _, first = distilbert_training.train_distilbert(train_rows, test_rows, **kwargs)
    second_model, _, second = distilbert_training.train_distilbert(train_rows, test_rows, **kwargs)

    assert first["history"] == second["history"]
    assert all(
        torch.equal(first_model.state_dict()[name], second_model.state_dict()[name])
        for name in first_model.state_dict()
    )
    assert first["selection_metric"] == "development_macro_f1"
    assert first_model.config.id2label == {
        0: "negative",
        1: "neutral",
        2: "positive",
    }
    assert first["config"] == {
        "model": "distilbert",
        "pretrained_model": "tiny-distilbert",
        "seed": 23,
        "device": "cpu",
        "epochs_requested": 2,
        "epochs_completed": len(first["history"]),
        "batch_size": 2,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "patience": 1,
        "max_length": 16,
        "configuration_selection": "development_macro_f1_only",
        "official_test_evaluated": True,
        "class_weighting": "none",
        "learning_rate_schedule": "constant",
        "warmup_ratio": 0.0,
    }
    output_dir = tmp_path / "distilbert"
    distilbert_training.save_artifact(first_model, _TinyTokenizer(), first, output_dir)
    run_record = json.loads((output_dir / "distilbert" / "training_run.json").read_text())
    assert run_record["config"] == first["config"]
    assert run_record["history"] == first["history"]


def test_distilbert_configuration_selection_skips_official_test(monkeypatch) -> None:
    train_rows, test_rows = _tiny_rows()
    monkeypatch.setattr(distilbert_training, "AspectPairDataset", _TinyPairDataset)
    monkeypatch.setattr(distilbert_training, "ABSADistilBERT", _TinyDistilBertFactory)
    monkeypatch.setattr(
        distilbert_training.AutoTokenizer,
        "from_pretrained",
        lambda _model_name: _TinyTokenizer(),
    )

    _model, _tokenizer, result = distilbert_training.train_distilbert(
        train_rows,
        test_rows,
        epochs=1,
        batch_size=2,
        max_length=16,
        model_name="tiny-distilbert",
        device=torch.device("cpu"),
        evaluate_official_test=False,
    )

    assert result["test"] == {
        "status": "not_evaluated_during_configuration_selection"
    }
    assert result["config"]["official_test_evaluated"] is False
    assert result["config"]["configuration_selection"] == (
        "development_macro_f1_only"
    )


def test_distilbert_restores_early_winner_before_evaluation_and_saving(monkeypatch, tmp_path) -> None:
    train_rows, test_rows = _tiny_rows()
    monkeypatch.setattr(distilbert_training, "AspectPairDataset", _TinyPairDataset)
    monkeypatch.setattr(distilbert_training, "ABSADistilBERT", _TinyDistilBertFactory)
    monkeypatch.setattr(distilbert_training, "BestCheckpoint", _RecordingBestCheckpoint)
    monkeypatch.setattr(
        distilbert_training.AutoTokenizer,
        "from_pretrained",
        lambda _model_name: _TinyTokenizer(),
    )
    monkeypatch.setattr(
        distilbert_training,
        "compute_metrics",
        _controlled_metrics([0.9, 0.8, 0.7, 0.9, 0.5]),
    )

    model, tokenizer, result = distilbert_training.train_distilbert(
        train_rows,
        test_rows,
        epochs=4,
        batch_size=2,
        seed=37,
        patience=2,
        max_length=16,
        model_name="tiny-distilbert",
        device=torch.device("cpu"),
    )
    tracker = _RecordingBestCheckpoint.latest
    assert tracker is not None
    assert result["best_epoch"] == 1
    assert result["stopped_early"] is True
    assert tracker.restore_calls == 1
    _assert_state_matches_selected(model, tracker.selected_state)

    output_dir = tmp_path / "distilbert-restored"
    distilbert_training.save_artifact(model, tokenizer, result, output_dir)
    saved_state = torch.load(output_dir / "distilbert" / "model.pt", weights_only=True)
    assert all(
        torch.equal(saved_state[name], tracker.selected_state[name])
        for name in tracker.selected_state
    )
