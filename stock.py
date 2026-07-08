# =====================================================

# STOCK MARKET PREDICTION USING LSTM (NIFTY 50)
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# =====================================================
# 1️⃣ DATA COLLECTION (NIFTY 50)
# =====================================================

print("Downloading NIFTY 50 data...")

data = yf.download("^NSEI", start="2015-01-01", end="2024-12-31")

print(data.head())
print("\nTotal Data Points:", len(data))

# =====================================================
# 2️⃣ DATA PREPROCESSING
# =====================================================

# Select only Close price
close_data = data[['Close']]
close_data.dropna(inplace=True)

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

# =====================================================
# 3️⃣ SEQUENCE CREATION (60-day window)
# =====================================================

def create_sequences(dataset, window_size=60):
    X = []
    y = []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# =====================================================
# 4️⃣ TRAIN-VALIDATION-TEST SPLIT (70-15-15)
# =====================================================

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# =====================================================
# 5️⃣ LSTM MODEL BUILDING
# =====================================================

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(window_size,1)))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mean_squared_error')

model.summary()

# =====================================================
# 6️⃣ MODEL TRAINING
# =====================================================

print("\nTraining model...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# =====================================================
# 7️⃣ PREDICTION
# =====================================================

predictions = model.predict(X_test)

# Inverse scaling
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# =====================================================
# 8️⃣ EVALUATION METRICS
# =====================================================

mae = mean_absolute_error(actual, predictions)
mse = mean_squared_error(actual, predictions)
rmse = math.sqrt(mse)
r2 = r2_score(actual, predictions)

# Directional Accuracy
direction_actual = np.sign(actual[1:] - actual[:-1])
direction_pred = np.sign(predictions[1:] - predictions[:-1])
directional_accuracy = np.mean(direction_actual == direction_pred) * 100

print("\n===== MODEL PERFORMANCE =====")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Directional Accuracy:", directional_accuracy, "%")

# =====================================================
# 9️⃣ VISUALIZATION
# =====================================================

plt.figure(figsize=(12,6))
plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("NIFTY 50 Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Index Value")
plt.legend()
plt.show()