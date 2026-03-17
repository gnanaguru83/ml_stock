"""Streamlit dashboard for stock analysis and LSTM + Attention prediction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from stock_model import START_DATE, StockArtifacts, train_and_evaluate


st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

AVAILABLE_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "NIFTY 50 (^NSEI)": "^NSEI",
}


def calculate_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index."""

    delta = close_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages and RSI to the stock data."""

    enriched = data.copy()
    # Moving averages help visualize overall price trend more smoothly.
    enriched["MA20"] = enriched["Close"].rolling(window=20).mean()
    enriched["MA50"] = enriched["Close"].rolling(window=50).mean()
    # RSI is a momentum indicator commonly used for overbought/oversold analysis.
    enriched["RSI"] = calculate_rsi(enriched["Close"])
    return enriched


@st.cache_resource(show_spinner=False)
def load_analysis(symbol: str) -> StockArtifacts:
    """Cache training results so the dashboard feels responsive."""

    # Streamlit caching avoids retraining the same model repeatedly during one session.
    return train_and_evaluate(symbol=symbol, start_date=START_DATE, save_outputs=False)


def plot_price_chart(data: pd.DataFrame, symbol: str) -> None:
    """Render close price and moving averages."""

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data["Close"], label="Close Price", linewidth=2)
    ax.plot(data.index, data["MA20"], label="20-Day MA", linestyle="--")
    ax.plot(data.index, data["MA50"], label="50-Day MA", linestyle=":")
    ax.set_title(f"{symbol} Price Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)


def plot_rsi(data: pd.DataFrame, symbol: str) -> None:
    """Render the RSI chart."""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data["RSI"], label="RSI", color="darkorange")
    ax.axhline(70, color="red", linestyle="--", linewidth=1, label="Overbought")
    ax.axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold")
    ax.set_title(f"{symbol} RSI")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, symbol: str) -> None:
    """Render model predictions against actual test values."""

    # This plot is the core regression visualization: real values versus model estimates.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(predictions_df["Date"], predictions_df["Actual"], label="Actual", linewidth=2)
    ax.plot(predictions_df["Date"], predictions_df["Predicted"], label="Predicted", linewidth=2)
    ax.set_title(f"{symbol} Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)


def plot_recent_actual_vs_predicted(
    predictions_df: pd.DataFrame, symbol: str, recent_points: int = 120
) -> None:
    """Render a zoomed-in view so prediction closeness is easier to inspect."""

    # A zoomed recent window makes short-term prediction quality easier to inspect visually.
    recent_df = predictions_df.tail(recent_points)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_df["Date"], recent_df["Actual"], label="Actual", linewidth=2)
    ax.plot(recent_df["Date"], recent_df["Predicted"], label="Predicted", linewidth=2)
    ax.set_title(f"{symbol} Recent Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)


def get_trade_signal(predictions_df: pd.DataFrame) -> tuple[str, float, float]:
    """Generate a simple buy/sell signal from the latest prediction."""

    # This is a simple rule-based signal, not a full trading strategy.
    last_row = predictions_df.iloc[-1]
    actual_price = float(last_row["Actual"])
    predicted_price = float(last_row["Predicted"])
    signal = "BUY" if predicted_price >= actual_price else "SELL"
    return signal, actual_price, predicted_price


def save_dashboard_outputs(artifacts: StockArtifacts) -> None:
    """Allow the dashboard to export the latest model artifacts on demand."""

    artifacts.model.save(Path("lstm_stock_model.h5"))
    artifacts.predictions_df.to_csv(Path("predictions.csv"), index=False)


def render_metrics(metrics: dict[str, float]) -> None:
    """Display model metrics in a compact layout."""

    # These metrics summarize regression quality and direction prediction quality.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics['mae']:.2f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    col3.metric("R2 Score", f"{metrics['r2']:.4f}")
    col4.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.2f}%")


def main() -> None:
    """Render the Streamlit application."""

    st.title("Stock Price Prediction Dashboard")
    st.write("Analyze historical prices, technical indicators, and LSTM + Attention forecasts.")

    selection = st.selectbox("Select a stock", list(AVAILABLE_STOCKS.keys()))
    symbol = AVAILABLE_STOCKS[selection]

    with st.spinner("Downloading data and training the LSTM + Attention model..."):
        artifacts = load_analysis(symbol)

    enriched_data = add_technical_indicators(artifacts.raw_data)

    st.subheader("Recent Closing Prices")
    st.dataframe(enriched_data.tail(10), use_container_width=True)

    st.subheader("Price Chart")
    plot_price_chart(enriched_data, symbol)

    st.subheader("Relative Strength Index")
    plot_rsi(enriched_data, symbol)

    st.subheader("Actual vs Predicted")
    plot_actual_vs_predicted(artifacts.predictions_df, symbol)
    st.subheader("Recent Prediction Zoom")
    plot_recent_actual_vs_predicted(artifacts.predictions_df, symbol)

    st.subheader("Model Performance")
    st.caption(
        "Architecture: Bidirectional LSTM -> Dropout -> Attention -> LSTM -> Dropout -> Dense"
    )
    render_metrics(artifacts.metrics)

    signal, last_actual, last_predicted = get_trade_signal(artifacts.predictions_df)
    st.subheader("Trading Signal")
    st.write(f"Latest actual close: {last_actual:.2f}")
    st.write(f"Latest predicted close: {last_predicted:.2f}")

    if signal == "BUY":
        st.success("BUY signal: predicted price is above the latest actual close.")
    else:
        st.error("SELL signal: predicted price is below the latest actual close.")

    if st.button("Save model and predictions to disk"):
        save_dashboard_outputs(artifacts)
        st.success("Saved `lstm_stock_model.h5` and `predictions.csv` in the current directory.")


if __name__ == "__main__":
    main()
