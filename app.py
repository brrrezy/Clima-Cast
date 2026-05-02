#!/usr/bin/env python3
"""
Temperature and Humidity Prediction using MLP Regressor

This script loads IoT sensor data from CSV files, trains a neural network
(MLPRegressor) to predict temperature and humidity values, and visualizes
the results.

Note: Originally written for TensorFlow/Keras. Modified to use scikit-learn's
MLPRegressor for compatibility with Python 3.14+.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(folder_path: str) -> pd.DataFrame:
    """Load all CSV files in *folder_path* and return a combined DataFrame.

    The CSV files are expected to contain a ``ts`` column with timestamps and
    sensor columns named ``humidity1``, ``temperature1`` … ``humidity8``,
    ``temperature8``.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(folder_path, f))
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["ts"] = pd.to_datetime(combined["ts"])
    combined.set_index("ts", inplace=True)
    return combined


def train_model(data: pd.DataFrame, seq_length: int = 30, epochs: int = 500):
    """Train an MLP regressor on *data*.

    Returns the trained model, the scaler used for normalization and the split
    datasets (X_test, y_test).
    """
    # Keep only the sensor columns – adjust if the CSV contains extra columns
    sensor_cols = [c for c in data.columns if "humidity" in c or "temperature" in c]
    data = data[sensor_cols]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i : i + seq_length].flatten())
        y.append(scaled[i + seq_length])
    X, y = np.array(X), np.array(y)

    # Train-test split (80 % train, 20 % test)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = MLPRegressor(
        hidden_layer_sizes=(50, 50),
        activation="relu",
        solver="adam",
        max_iter=epochs,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test


def plot_results(y_test, predictions, output_path="prediction.png"):
    """Create a plot comparing actual vs. predicted values.

    The plot is saved to *output_path*.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.xlabel("Time step")
    plt.ylabel("Scaled sensor values")
    plt.title("Actual vs Predicted Sensor Readings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    # If a folder path is provided use it, otherwise default to the bundled data directory
    if len(sys.argv) == 2:
        folder = sys.argv[1]
    else:
        folder = os.path.join(os.path.dirname(__file__), "data")
    data = load_and_preprocess_data(folder)
    model, scaler, X_test, y_test = train_model(data)
    preds = model.predict(X_test)
    # Inverse transform to original scale for readability
    preds_original = scaler.inverse_transform(preds)
    y_test_original = scaler.inverse_transform(y_test)
    plot_results(y_test_original, preds_original)


if __name__ == "__main__":
    main()
