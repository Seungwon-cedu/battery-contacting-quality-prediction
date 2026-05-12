from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """Torch dataset for multivariate time-series classification."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        if sequences.ndim != 3:
            raise ValueError("Expected sequences with shape (samples, timesteps, features).")

        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class LSTMClassifier(nn.Module):
    """Two-layer LSTM classifier for multi-channel sensor traces."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.25,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.classifier(last_step)


@dataclass
class LSTMTrainingConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.25
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 25
    patience: int = 5
    bidirectional: bool = False
    random_state: int = 42
    device: str = "cpu"


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def oversample_sequences(
    sequences: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes by sampling minority classes with replacement."""

    rng = np.random.default_rng(random_state)
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = int(counts.max())

    sampled_sequences = []
    sampled_labels = []
    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        selected = rng.choice(indices, size=max_count, replace=True)
        sampled_sequences.append(sequences[selected])
        sampled_labels.append(labels[selected])

    return np.concatenate(sampled_sequences), np.concatenate(sampled_labels)


def jitter_sequences(
    sequences: np.ndarray,
    noise_std: float = 0.02,
    random_state: int = 42,
) -> np.ndarray:
    """Add small Gaussian noise to time-series traces for augmentation."""

    rng = np.random.default_rng(random_state)
    scale = np.std(sequences, axis=(0, 1), keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    noise = rng.normal(loc=0.0, scale=noise_std * scale, size=sequences.shape)
    return sequences + noise


def augment_training_data(
    sequences: np.ndarray,
    labels: np.ndarray,
    method: str | None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply train-fold-only augmentation for imbalanced sequence data."""

    if method is None or method == "none":
        return sequences, labels
    if method == "oversampling":
        return oversample_sequences(sequences, labels, random_state)
    if method == "jitter":
        jittered = jitter_sequences(sequences, random_state=random_state)
        return np.concatenate([sequences, jittered]), np.concatenate([labels, labels])
    if method == "oversampling_jitter":
        balanced_x, balanced_y = oversample_sequences(sequences, labels, random_state)
        jittered = jitter_sequences(balanced_x, random_state=random_state)
        return np.concatenate([balanced_x, jittered]), np.concatenate([balanced_y, balanced_y])

    raise ValueError("method must be one of: none, oversampling, jitter, oversampling_jitter")


def train_lstm_once(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: LSTMTrainingConfig | None = None,
) -> tuple[LSTMClassifier, dict[str, list[float]]]:
    """Train one LSTM model and keep the best validation-loss checkpoint."""

    config = config or LSTMTrainingConfig()
    set_torch_seed(config.random_state)
    device = torch.device(config.device)

    train_loader = DataLoader(
        SequenceDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(x_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = LSTMClassifier(
        input_size=x_train.shape[-1],
        hidden_size=config.hidden_size,
        num_classes=num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(config.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_loss += loss_fn(model(batch_x), batch_y).item() * len(batch_y)

        train_loss /= len(y_train)
        val_loss /= len(y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict_lstm(
    model: LSTMClassifier,
    sequences: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Predict class ids for a sequence batch."""

    loader = DataLoader(
        SequenceDataset(sequences, np.zeros(len(sequences), dtype=int)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def cross_validate_lstm(
    sequences: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    augmentation: str | None = "oversampling_jitter",
    config: LSTMTrainingConfig | None = None,
) -> dict[str, object]:
    """Run stratified cross-validation for the LSTM pipeline."""

    config = config or LSTMTrainingConfig()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)

    accuracies = []
    f1_scores = []
    histories = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(sequences, y_encoded), start=1):
        x_train, y_train = sequences[train_idx], y_encoded[train_idx]
        x_val, y_val = sequences[val_idx], y_encoded[val_idx]

        x_train_aug, y_train_aug = augment_training_data(
            x_train,
            y_train,
            augmentation,
            random_state=config.random_state + fold,
        )
        model, history = train_lstm_once(
            x_train_aug,
            y_train_aug,
            x_val,
            y_val,
            num_classes=len(label_encoder.classes_),
            config=config,
        )
        y_pred = predict_lstm(model, x_val, config.batch_size, config.device)

        accuracies.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        histories.append(history)

    return {
        "label_encoder": label_encoder,
        "accuracy": np.asarray(accuracies),
        "f1_macro": np.asarray(f1_scores),
        "histories": histories,
    }
