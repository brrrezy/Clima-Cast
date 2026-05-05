# Clima-Case | Temperature & Humidity Prediction

A Python script that loads IoT sensor CSV files, trains a neural‑network model
(MLPRegressor) to predict temperature and humidity, and visualises actual vs.
predicted values.

## Requirements

- Python ≥ 3.14
- `pip install -r requirements.txt`

## Usage

```bash
# Run with the default data folder (./data)
python app.py

# Or specify a custom folder
python app.py /path/to/your/csvs
