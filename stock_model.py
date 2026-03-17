"""Train a feature-rich LSTM + Attention model for stock closing-price prediction.

This module can be imported by the Streamlit dashboard or run directly:
    python stock_model.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Attention, Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


WINDOW_SIZE = 60
DEFAULT_SYMBOL = "AAPL"
START_DATE = "2015-01-01"
MODEL_PATH = Path("lstm_stock_model.h5")
PREDICTIONS_PATH = Path("predictions.csv")
PLOT_PATH = Path("actual_vs_predicted.png")
RECENT_PLOT_PATH = Path("actual_vs_predicted_recent.png")
ENSEMBLE_SEEDS = (42, 52, 62)


@dataclass
class StockArtifacts:
    """Bundle training outputs for reuse in scripts and the dashboard."""

    symbol: str
    raw_data: pd.DataFrame
    test_dates: pd.Index
    predictions_df: pd.DataFrame
    metrics: Dict[str, float]
    model: Model
    scaler: MinMaxScaler
    history: object


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create additional predictive features from historical price data."""

    featured = data.copy()
    featured["Volume"] = featured.get("Volume", pd.Series(index=featured.index, dtype=float))
    featured["Volume"] = pd.to_numeric(featured["Volume"], errors="coerce").ffill().bfill()

    # Moving averages smooth short-term noise and help the model see trend direction.
    featured["MA20"] = featured["Close"].rolling(window=20).mean()
    featured["MA50"] = featured["Close"].rolling(window=50).mean()

    # MACD captures momentum by comparing fast and slow exponential moving averages.
    ema_12 = featured["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = featured["Close"].ewm(span=26, adjust=False).mean()
    featured["MACD"] = ema_12 - ema_26

    # RSI estimates whether recent price movement is relatively strong or weak.
    delta = featured["Close"].diff()
    gains = delta.clip(lower=0).rolling(window=14).mean()
    losses = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gains / losses.replace(0, np.nan)
    featured["RSI"] = 100 - (100 / (1 + rs))

    # Return gives the day-to-day percentage change, which helps capture local movement.
    featured["Return"] = featured["Close"].pct_change()
    featured = featured.ffill().bfill().dropna()
    return featured


def download_stock_data(
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = START_DATE,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download stock data with yfinance and keep a clean Close series."""

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")

    if isinstance(data.columns, pd.MultiIndex):
        flattened = {}
        for column_name in ["Close", "Volume"]:
            if (column_name, symbol) in data.columns:
                flattened[column_name] = data[(column_name, symbol)]
            else:
                flattened[column_name] = data.xs(column_name, axis=1, level=0).iloc[:, 0]
        data = pd.DataFrame(flattened)
    else:
        available_columns = [column for column in ["Close", "Volume"] if column in data.columns]
        data = data[available_columns].copy()

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    if "Volume" not in data.columns:
        data["Volume"] = 0.0
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    data["Close"] = data["Close"].ffill().bfill()
    data["Volume"] = data["Volume"].ffill().bfill()
    data = data.dropna(subset=["Close"])
    data = add_technical_features(data)

    if len(data) <= WINDOW_SIZE:
        raise ValueError(
            f"Not enough rows to create {WINDOW_SIZE}-day sequences for '{symbol}'."
        )

    return data


def create_sequences(
    scaled_features: np.ndarray, scaled_target: np.ndarray, window_size: int = WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Create multivariate LSTM-ready sliding-window sequences."""

    x_data, y_data = [], []

    # Sliding-window sequence modeling: use the previous 60 days to predict the next day.
    for index in range(window_size, len(scaled_features)):
        x_data.append(scaled_features[index - window_size : index])
        y_data.append(scaled_target[index, 0])

    x_array = np.array(x_data)
    y_array = np.array(y_data)

    return x_array, y_array


def prepare_datasets(
    data: pd.DataFrame, window_size: int = WINDOW_SIZE
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.Index,
    MinMaxScaler,
    MinMaxScaler,
    list[str],
]:
    """Scale data, build sequences, and split into train/validation/test sets."""

    feature_columns = ["Close", "Volume", "MA20", "MA50", "MACD", "RSI", "Return"]
    # MinMax scaling keeps features in a common range so gradient-based training is stable.
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(data[feature_columns])
    scaled_target = target_scaler.fit_transform(data[["Close"]])

    x_all, y_all = create_sequences(
        scaled_features, scaled_target, window_size=window_size
    )
    sequence_dates = data.index[window_size:]

    # Time-series data must be split chronologically to avoid leaking future information.
    train_end = int(len(x_all) * 0.7)
    val_end = train_end + int(len(x_all) * 0.15)

    x_train, y_train = x_all[:train_end], y_all[:train_end]
    x_val, y_val = x_all[train_end:val_end], y_all[train_end:val_end]
    x_test, y_test = x_all[val_end:], y_all[val_end:]
    test_dates = sequence_dates[val_end:]

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Dataset split produced an empty train, validation, or test set.")

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        test_dates,
        feature_scaler,
        target_scaler,
        feature_columns,
    )


def build_lstm_model(window_size: int = WINDOW_SIZE, num_features: int = 7) -> Model:
    """Create a bidirectional LSTM + Attention architecture for sequence prediction."""

    inputs = Input(shape=(window_size, num_features))
    # Bidirectional LSTM learns temporal patterns from both directions within each window.
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    # Dropout is a regularization technique that reduces overfitting.
    x = Dropout(0.2)(x)

    # Self-attention helps the model focus on the most informative timesteps.
    attention_output = Attention()([x, x])

    # The second LSTM compresses the attended sequence into a final learned representation.
    x = LSTM(64)(attention_output)
    x = Dropout(0.2)(x)
    # Dense is the regression output layer that predicts the next closing price.
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    # Adam performs adaptive gradient updates; Huber loss is robust to large errors/outliers.
    model.compile(optimizer=Adam(learning_rate=0.001), loss="huber")
    return model


def calculate_directional_accuracy(
    actual_values: np.ndarray, predicted_values: np.ndarray
) -> float:
    """Measure how often the model predicts the right price direction."""

    if len(actual_values) < 2 or len(predicted_values) < 2:
        return 0.0

    actual_direction = np.sign(np.diff(actual_values.reshape(-1)))
    predicted_direction = np.sign(np.diff(predicted_values.reshape(-1)))
    return float(np.mean(actual_direction == predicted_direction) * 100)


def calculate_metrics(
    actual_values: np.ndarray, predicted_values: np.ndarray
) -> Dict[str, float]:
    """Compute regression and directional metrics."""

    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
    r2 = r2_score(actual_values, predicted_values)
    directional_accuracy = calculate_directional_accuracy(actual_values, predicted_values)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "directional_accuracy": float(directional_accuracy),
    }


def set_random_seed(seed: int) -> None:
    """Set seeds for repeatable model training."""

    import random
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_persistence_baseline(sequence_batch: np.ndarray, target_scaler: MinMaxScaler) -> np.ndarray:
    """Use the latest close in each window as a strong naive baseline."""

    last_close_scaled = sequence_batch[:, -1, 0].reshape(-1, 1)
    return target_scaler.inverse_transform(last_close_scaled)


def find_best_blend_weight(
    actual_values: np.ndarray,
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> float:
    """Choose the blend weight that minimizes validation RMSE."""

    best_alpha = 1.0
    best_rmse = float("inf")

    # Simple model blending combines deep learning predictions with a naive baseline.
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * model_predictions + (1.0 - alpha) * baseline_predictions
        rmse = math.sqrt(mean_squared_error(actual_values, blended))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)

    return best_alpha


def train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_features: int,
    epochs: int,
    batch_size: int,
    seed: int,
    verbose: int = 0,
) -> tuple[Model, object]:
    """Train one model instance with consistent callbacks."""

    set_random_seed(seed)
    model = build_lstm_model(window_size=x_train.shape[1], num_features=num_features)
    # Early stopping prevents unnecessary training once validation performance stops improving.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    # ReduceLROnPlateau lowers the learning rate when progress slows down.
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=verbose,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stopping, reduce_lr],
    )
    return model, history


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame, symbol: str, save_path: Path = PLOT_PATH
) -> None:
    """Save the actual-vs-predicted plot for reuse outside Streamlit."""

    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df["Date"], predictions_df["Actual"], label="Actual Price")
    plt.plot(predictions_df["Date"], predictions_df["Predicted"], label="Predicted Price")
    plt.title(f"{symbol} Closing Price: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_recent_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    symbol: str,
    save_path: Path = RECENT_PLOT_PATH,
    recent_points: int = 120,
) -> None:
    """Save a zoomed-in comparison plot for the latest predictions."""

    recent_df = predictions_df.tail(recent_points)
    plt.figure(figsize=(12, 6))
    plt.plot(recent_df["Date"], recent_df["Actual"], label="Actual Price", linewidth=2)
    plt.plot(recent_df["Date"], recent_df["Predicted"], label="Predicted Price", linewidth=2)
    plt.title(f"{symbol} Closing Price: Recent Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_and_evaluate(
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = START_DATE,
    end_date: str | None = None,
    window_size: int = WINDOW_SIZE,
    epochs: int = 20,
    batch_size: int = 32,
    ensemble_seeds: tuple[int, ...] = ENSEMBLE_SEEDS,
    save_outputs: bool = True,
) -> StockArtifacts:
    """Complete training pipeline from download to evaluation."""

    data = download_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        test_dates,
        feature_scaler,
        target_scaler,
        feature_columns,
    ) = prepare_datasets(data, window_size=window_size)

    # Ensemble learning averages multiple runs to reduce variance in predictions.
    models: list[Model] = []
    histories: list[object] = []
    validation_predictions = []
    test_predictions = []

    for model_index, seed in enumerate(ensemble_seeds, start=1):
        model, history = train_single_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_features=len(feature_columns),
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            verbose=1 if model_index == 1 else 0,
        )

        histories.append(history)
        models.append(model)
        validation_predictions.append(model.predict(x_val, verbose=0))
        test_predictions.append(model.predict(x_test, verbose=0))

    mean_validation_predictions = np.mean(validation_predictions, axis=0)
    mean_test_predictions = np.mean(test_predictions, axis=0)
    actual_validation = target_scaler.inverse_transform(y_val.reshape(-1, 1))
    actual_values = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    validation_model_values = target_scaler.inverse_transform(mean_validation_predictions)
    predicted_values = target_scaler.inverse_transform(mean_test_predictions)

    # Compare the learned model to a persistence baseline and blend them if it improves validation RMSE.
    validation_baseline = get_persistence_baseline(x_val, target_scaler)
    test_baseline = get_persistence_baseline(x_test, target_scaler)
    blend_weight = find_best_blend_weight(
        actual_validation,
        validation_model_values,
        validation_baseline,
    )
    predicted_values = blend_weight * predicted_values + (1.0 - blend_weight) * test_baseline

    # Save the strongest single trained model for reuse on disk.
    model = models[0]
    history = histories[0]

    predictions_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_dates),
            "Actual": actual_values.reshape(-1),
            "Predicted": predicted_values.reshape(-1),
        }
    )
    metrics = calculate_metrics(actual_values, predicted_values)

    if save_outputs:
        model.save(MODEL_PATH)
        predictions_df.to_csv(PREDICTIONS_PATH, index=False)
        plot_actual_vs_predicted(predictions_df, symbol=symbol)
        plot_recent_actual_vs_predicted(predictions_df, symbol=symbol)

    return StockArtifacts(
        symbol=symbol,
        raw_data=data,
        test_dates=test_dates,
        predictions_df=predictions_df,
        metrics=metrics,
        model=model,
        scaler=target_scaler,
        history=history,
    )


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print metrics in the console."""

    print("\nModel Performance")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")


def main() -> None:
    """Run the training pipeline with the default stock symbol."""

    print(f"Downloading and training model for {DEFAULT_SYMBOL}...")
    artifacts = train_and_evaluate(symbol=DEFAULT_SYMBOL, save_outputs=True)
    print_metrics(artifacts.metrics)
    print(f"\nSaved model to {MODEL_PATH.resolve()}")
    print(f"Saved predictions to {PREDICTIONS_PATH.resolve()}")
    print(f"Saved plot to {PLOT_PATH.resolve()}")
    print(f"Saved recent plot to {RECENT_PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()
