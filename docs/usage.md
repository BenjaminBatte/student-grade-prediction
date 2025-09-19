---
title: Usage
---

# 🚀 Usage Guide

This section explains how to run the **Student Grade Prediction** pipeline, generate sample input data, and make predictions.

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/student-grade-prediction.git
   cd student-grade-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Training Models

Train and evaluate a model using the CLI:

```bash
# Train on math dataset
python src/main.py --dataset math --model random_forest

# Train on portuguese dataset
python src/main.py --dataset portuguese --model random_forest
```

- Metrics are saved to: `results/metrics/*.json`
- Trained models are saved to: `results/models/*.pkl`
- Logs are written to: `results/logs/project.log`

---

## 🛠 Generating Sample Prediction Data

Use the helper script to create valid input files for prediction (they match the training schema exactly):

```bash
python src/generate_sample_data.py
```

This creates:

- `data/new_data_math.csv` → 5 rows from the math dataset (without `G3`)
- `data/new_data.csv` → 5 rows from the Portuguese dataset (without `G3`)

---

## 🔍 Running Predictions

Run predictions on new data with a trained model:

```bash
# Predict with the math model
python src/predict.py --model results/models/random_forest_math.pkl --data data/new_data_math.csv --out results/predictions

# Predict with the Portuguese model
python src/predict.py --model results/models/random_forest_portuguese.pkl --data data/new_data.csv --out results/predictions
```

- The first 10 predictions are printed to the console.
- All predictions are saved to: `results/predictions/predictions.csv`

Example output:

```
First 10 predictions: [10.82 11.06 13.07 14.54 13.02]
```

---

[⬅️ Back: Architecture](architecture.md) | [➡️ Next: Results](results.md)

---
