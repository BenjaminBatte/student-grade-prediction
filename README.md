---
🎓 Student Grade Prediction
---

A machine learning pipeline for predicting students’ final grades (G3) using the UCI Student Performance dataset (Cortez & Silva, 2008).
The project includes **data preprocessing, exploratory data analysis, regression models, evaluation metrics, and reproducible results**.

## 📑 Project Report & Presentation

All course project deliverables are stored in [`docs/project_report/`](./docs/project_report/):

- [Student_Grade_Prediction_Report.doc](./docs/project_report/Student_Grade_Prediction_Report.doc)
- [Student_Grade_Prediction_Report.pdf](./docs/project_report/Student_Grade_Prediction_Report.pdf)
- [Student_Grade_Prediction_Presentation.pptx](./docs/project_report/Student_Grade_Prediction_Presentation.pptx)

🔗 **Live Documentation:** [https://benjaminbatte.github.io/student-grade-prediction/](https://benjaminbatte.github.io/student-grade-prediction/)

---

## 📂 Project Structure

```
student-grade-prediction/
│── data/                     # Raw datasets (student-mat.csv, student-por.csv)
│── results/                  # Outputs: figures, metrics, models, logs
│── src/                      # Source code
│   ├── __init__.py           # Package initializer
│   ├── data_loader.py        # Load datasets
│   ├── eda.py                # Exploratory data analysis (plots, summaries)
│   ├── preprocessing.py      # ColumnTransformer (scaling + encoding)
│   ├── model.py              # Train, evaluate, and save models
│   ├── main.py               # CLI entry point: full pipeline (EDA + train + eval)
│   ├── predict.py            # Run predictions on new data
│   ├── generate_sample_data.py # Create small test datasets
│   └── utils.py              # Logger and utilities
│── notebooks/                # Jupyter notebooks (EDA, model experiments)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│── docs/                     # Documentation site (GitHub Pages)
│   ├── index.md              # Home page
│   ├── about.md              # About the project
│   ├── architecture.md       # Project architecture
│   ├── usage.md              # Usage instructions
│   ├── results.md            # Results summary
│   ├── figures.md            # Figures and EDA plots
│   ├── discussion.md         # Discussion and implications
│   └── conclusion.md         # Conclusion and future directions
│── README.md                 # Project overview
│── requirements.txt          # Python dependencies
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/BenjaminBatte/student-grade-prediction.git
cd student-grade-prediction

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Full Pipeline (EDA + Train + Evaluate + Save Models)

```bash
python -m src.main
```

- Generates EDA plots → `results/figures/`
- Trains models (Linear Regression, Random Forest) on both datasets
- Saves metrics → `results/metrics/`
- Saves trained models → `results/models/`
- Logs all runs → `results/logs/project.log`

### Run Predictions on New Data

```bash
python -m src.predict --model results/models/random_forest_math.pkl --data data/new_data_math.csv
```

- Predictions saved to → `results/predictions/predictions.csv`

### Generate Sample Data for Testing

```bash
python -m src.generate_sample_data
```

---

## 📊 Models & Metrics

- **Linear Regression** → Baseline model (weaker on categorical-heavy datasets).
- **Random Forest Regression** → Stronger performance, captures non-linearities.

Metrics saved include:

- **R²** (Coefficient of Determination)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

---

## 📒 Jupyter Notebooks

- **01_data_exploration.ipynb** → Initial EDA, distributions, correlations
- **02_feature_engineering.ipynb** → Feature preprocessing, encoding, scaling
- **03_modeling.ipynb** → Model training, evaluation, hyperparameter tuning

These notebooks complement the `src/` pipeline and allow for interactive experimentation.

---

## 📖 Documentation

- **Home:** [docs/index.md](./docs/index.md)
- **About:** [docs/about.md](./docs/about.md)
- **Architecture:** [docs/architecture.md](./docs/architecture.md)
- **Usage:** [docs/usage.md](./docs/usage.md)
- **Results:** [docs/results.md](./docs/results.md)
- **Figures:** [docs/figures.md](./docs/figures.md)
- **Discussion:** [docs/discussion.md](./docs/discussion.md)
- **Conclusion:** [docs/conclusion.md](./docs/conclusion.md)

---

## 🧾 References

- Cortez, P., & Silva, A. M. G. (2008). _Student performance dataset_. UCI Machine Learning Repository. [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- McKinney, W. (2022). _Python for data analysis_ (3rd ed.). O’Reilly Media.
- Shorfuzzaman, M., Hossain, M. S., & Nazir, A. (2021). Performance prediction of students using machine learning algorithms: A case study. _Computers, 10_(8), 93. [https://doi.org/10.3390/computers10080093](https://doi.org/10.3390/computers10080093)
- Kumari, A., & Singh, V. (2023). Comparative analysis of machine learning techniques for student performance prediction. _International Journal of Emerging Technologies in Learning, 18_(4), 22–34. [https://doi.org/10.3991/ijet.v18i04.35575](https://doi.org/10.3991/ijet.v18i04.35575)
- Manhães, L. M. B., Souza, L. A., Moreira, E. D., & Rocha, Á. R. (2022). Using ensemble learning to predict academic performance. _Education and Information Technologies, 27_(3), 3259–3278. [https://doi.org/10.1007/s10639-021-10733-4](https://doi.org/10.1007/s10639-021-10733-4)

---
