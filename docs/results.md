---
##title: Results
---

# 📊 Results

## This section summarizes the outcomes of the **Student Grade Prediction** models, including evaluation metrics and interpretation of findings.

## 📈 Evaluation Metrics

Two models were applied to both datasets: **Linear Regression (LR)** and **Random Forest Regression (RF)**. The performance was evaluated using R², RMSE, and MAE.

### Mathematics Dataset

| Model             | R²    | RMSE | MAE  |
| ----------------- | ----- | ---- | ---- |
| Linear Regression | 0.724 | 2.38 | 1.65 |
| Random Forest     | 0.813 | 1.96 | 1.18 |

**Interpretation:** Random Forest outperformed Linear Regression, reducing prediction error by ~0.4–0.5 grade points and explaining ~8% more variance. The RF model typically predicts within **2 grade points** on a 0–20 scale.

---

### Portuguese Dataset

| Model             | R²    | RMSE | MAE  |
| ----------------- | ----- | ---- | ---- |
| Linear Regression | 0.849 | 1.22 | 0.77 |
| Random Forest     | 0.842 | 1.24 | 0.75 |

**Interpretation:** Both models performed strongly, with predictions accurate within **~1 grade point**. Interestingly, Linear Regression slightly outperformed Random Forest in this dataset, likely due to the stronger linear relationships among variables.

---

## 📊 Exploratory Data Analysis

- **Absences (Math):** Most students had few absences, but a small group showed extreme absenteeism, disproportionately affecting their performance.
- **Final Grades (Portuguese):** Distribution was approximately normal (10–12 out of 20), with a subset of students failing entirely (G3 = 0).
- **Correlation Analysis:** Strongest predictors of final grade (G3) were **first (G1) and second period grades (G2)**, while demographic and lifestyle factors showed weak correlations.

---

[⬅️ Back: Usage](usage.md) | [➡️ Next: Figures](figures.md)
