# StockSense — LSTM + Attention Stock Price Prediction

A full-stack ML system for stock price forecasting using a Bidirectional LSTM with Self-Attention, served through an interactive Streamlit dashboard.

---

## Features

- **Deep Learning Model** — Bidirectional LSTM + Self-Attention architecture trained on multivariate time-series data
- **Technical Indicators** — MA20, MA50, MACD, RSI, and Daily Return engineered as input features
- **Ensemble Training** — Averages predictions across 3 independently seeded models to reduce variance
- **Baseline Blending** — Blends model predictions with a persistence baseline, selecting the optimal blend weight by minimizing validation RMSE
- **Adaptive Training** — Early Stopping and ReduceLROnPlateau callbacks prevent overfitting and wasted epochs
- **Interactive Dashboard** — Streamlit app with live stock selector, price charts, RSI, actual vs predicted plots, and BUY/SELL signal
- **Supported Stocks** — Apple (AAPL) and NIFTY 50 (^NSEI) out of the box; easily extensible

---

## Project Structure

```
ml_project/
├── app.py              # Streamlit dashboard
├── stock_model.py      # Core ML pipeline (data download, feature engineering, training, evaluation)
├── stock.py            # Baseline single LSTM script (NIFTY 50)
├── requirements.txt    # Python dependencies
├── predictions.csv     # Saved test set predictions (generated on run)
└── lstm_stock_model.h5 # Saved Keras model weights (generated on run)
```

---

## Model Architecture

```
Input (60 days × 7 features)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
Self-Attention
    ↓
LSTM (64 units)
    ↓
Dropout (0.2)
    ↓
Dense (1) → Predicted Close Price
```

**Loss function:** Huber (robust to outliers)  
**Optimizer:** Adam (lr=0.001)  
**Train / Val / Test split:** 70% / 15% / 15% (chronological, no leakage)

---

## Input Features

| Feature  | Description                          |
|----------|--------------------------------------|
| Close    | Closing price                        |
| Volume   | Trading volume                       |
| MA20     | 20-day moving average                |
| MA50     | 50-day moving average                |
| MACD     | Moving Average Convergence Divergence|
| RSI      | Relative Strength Index (14-day)     |
| Return   | Daily percentage change              |

---

## Evaluation Metrics

| Metric               | Description                                      |
|----------------------|--------------------------------------------------|
| MAE                  | Mean Absolute Error                              |
| RMSE                 | Root Mean Squared Error                          |
| R² Score             | Coefficient of determination                     |
| Directional Accuracy | % of days where price direction was predicted correctly |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/gnanaguru83/ml_stock.git
cd ml_stock
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard
```bash
streamlit run app.py
```

### 5. Or run the training script directly
```bash
python stock_model.py
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- Streamlit
- yFinance
- scikit-learn
- pandas, numpy, matplotlib

See `requirements.txt` for pinned versions.

---

## Trading Signal

The dashboard generates a simple rule-based signal from the latest prediction:

- **BUY** — predicted next close is above the latest actual close
- **SELL** — predicted next close is below the latest actual close

> This is not financial advice. The signal is for educational and demonstration purposes only.

---

## License

MIT
