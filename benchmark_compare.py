"""Benchmark this project against the cloned AI-TRADING-BOT LSTM model."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from stock_model import (
    START_DATE,
    WINDOW_SIZE,
    build_lstm_model,
    download_stock_data,
)


BENCHMARK_REPO = Path("benchmark_repo/AI-TRADING-BOT")
OUTPUT_PATH = Path("benchmark_results.csv")
SYMBOL = "AAPL"
EPOCHS = 15
BATCH_SIZE = 32


def set_seed(seed: int = 42) -> None:
    """Make runs more reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def add_repo_to_path() -> None:
    """Expose the cloned repository for imports."""

    repo_path = BENCHMARK_REPO.resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Missing benchmark repo at {repo_path}")

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def create_shared_sequences(
    data: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, pd.Index, object]:
    """Build a common scaled dataset for both models."""

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(data[["Close"]])

    x_data, y_data = [], []
    for index in range(window_size, len(scaled_close)):
        x_data.append(scaled_close[index - window_size : index, 0])
        y_data.append(scaled_close[index, 0])

    x_array = np.array(x_data).reshape(-1, window_size, 1)
    y_array = np.array(y_data)
    dates = data.index[window_size:]
    return x_array, y_array, dates, scaler


def split_sequences(
    x_data: np.ndarray,
    y_data: np.ndarray,
    dates: pd.Index,
) -> tuple[np.ndarray, ...]:
    """Use the same 70/15/15 chronological split for both models."""

    train_end = int(len(x_data) * 0.7)
    val_end = train_end + int(len(x_data) * 0.15)

    x_train, y_train = x_data[:train_end], y_data[:train_end]
    x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
    x_test, y_test = x_data[val_end:], y_data[val_end:]
    test_dates = dates[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test, test_dates


def calculate_directional_accuracy(
    actual_values: np.ndarray, predicted_values: np.ndarray
) -> float:
    """Measure whether price direction is predicted correctly."""

    actual_direction = np.sign(np.diff(actual_values.reshape(-1)))
    predicted_direction = np.sign(np.diff(predicted_values.reshape(-1)))
    return float(np.mean(actual_direction == predicted_direction) * 100)


def calculate_metrics(
    actual_values: np.ndarray, predicted_values: np.ndarray
) -> dict[str, float]:
    """Calculate benchmark metrics."""

    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(math.sqrt(mean_squared_error(actual_values, predicted_values))),
        "R2": float(r2_score(actual_values, predicted_values)),
        "Directional_Accuracy": float(
            calculate_directional_accuracy(actual_values, predicted_values)
        ),
    }


def train_our_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    scaler,
) -> np.ndarray:
    """Train this project's LSTM model."""

    model = build_lstm_model(window_size=WINDOW_SIZE, num_features=x_train.shape[2])
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[callback],
    )
    predictions = model.predict(x_test, verbose=0)
    return scaler.inverse_transform(predictions)


def train_repo_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    scaler,
) -> np.ndarray:
    """Train the cloned repository's LSTM model on the same split."""

    add_repo_to_path()
    from lstm_model import StockPredictor

    predictor = StockPredictor((x_train.shape[1], x_train.shape[2]))
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
    predictor.model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[callback],
    )
    predictions = predictor.predict(x_test)
    return scaler.inverse_transform(predictions)


def main() -> None:
    """Run the benchmark and save a result table."""

    set_seed()
    print(f"Downloading {SYMBOL} data from {START_DATE} onward...")
    data = download_stock_data(symbol=SYMBOL, start_date=START_DATE)

    x_data, y_data, dates, scaler = create_shared_sequences(data)
    x_train, y_train, x_val, y_val, x_test, y_test, test_dates = split_sequences(
        x_data, y_data, dates
    )

    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

    print("Training this project's model...")
    our_predictions = train_our_model(x_train, y_train, x_val, y_val, x_test, scaler)
    our_metrics = calculate_metrics(actual_values, our_predictions)

    print("Training AI-TRADING-BOT model...")
    repo_predictions = train_repo_model(x_train, y_train, x_val, y_val, x_test, scaler)
    repo_metrics = calculate_metrics(actual_values, repo_predictions)

    results = pd.DataFrame(
        [
            {"Model": "Our Project", **our_metrics},
            {"Model": "AI-TRADING-BOT", **repo_metrics},
        ]
    )
    results.to_csv(OUTPUT_PATH, index=False)

    detailed_predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_dates),
            "Actual": actual_values.reshape(-1),
            "Our_Project_Predicted": our_predictions.reshape(-1),
            "AI_Trading_Bot_Predicted": repo_predictions.reshape(-1),
        }
    )
    detailed_predictions.to_csv("benchmark_predictions.csv", index=False)

    print("\nBenchmark Results")
    print(results.to_string(index=False))
    print(f"\nSaved metrics to {OUTPUT_PATH.resolve()}")
    print(f"Saved prediction comparison to {Path('benchmark_predictions.csv').resolve()}")


if __name__ == "__main__":
    main()
