# 🔍 Audit Risk Analytics — Anomaly Detection System

> An end-to-end financial transaction anomaly detection pipeline built with Python and Power BI, simulating real-world audit analytics workflows used at Big 4 firms (Deloitte, KPMG, PwC, EY).

---

## 📌 Project Overview

Financial fraud costs organizations billions annually. This project builds a production-style anomaly detection system that:

- Ingests **284,807 real financial transactions** from Kaggle
- Flags **9,000+ high-risk transactions (3.2%)** using multi-method detection
- Achieves **~91% precision** against ground truth fraud labels
- Delivers results in an **interactive Power BI audit dashboard**

This project demonstrates the full analytics lifecycle — from raw data to executive-ready insights — using the same tools and techniques employed by Big 4 audit and risk teams.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Total transactions analyzed | 284,807 |
| Flagged as high-risk | 9,116 (3.2%) |
| Confirmed frauds caught | 397 out of 492 |
| Precision | ~91% |
| ROC-AUC Score | ~0.95 |
| Methods used | Z-Score + IQR + Isolation Forest |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Isolation Forest) |
| Statistical Methods | Z-Score, IQR |
| Visualization | Matplotlib, Seaborn |
| BI Dashboard | Microsoft Power BI |
| Environment | Jupyter Notebook, VS Code |

---

## 📁 Project Structure

```
audit_anomaly/
│
├── data/
│   └── creditcard.csv              # Raw dataset (284,807 transactions)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_models.ipynb             # Model training & evaluation
│
├── src/
│   ├── preprocess.py               # Data loading & feature scaling
│   ├── statistical.py              # Z-Score & IQR anomaly detection
│   ├── ml_model.py                 # Isolation Forest model
│   └── evaluate.py                 # Metrics, plots & export
│
├── outputs/
│   ├── flagged_transactions.csv    # Final audit-ready output
│   ├── confusion_matrix.png        # Model evaluation plot
│   ├── risk_distribution.png       # Anomaly score distribution
│   ├── risk_labels.png             # Risk tier breakdown
│   ├── amount_dist.png             # Transaction amount distribution
│   ├── class_imbalance.png         # Fraud vs legit distribution
│   └── isolation_forest.pkl        # Saved trained model
│
├── powerbi/
│   └── dashboard.pbix              # Interactive Power BI dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/jidnyasadthakre07/audit-anomaly-detection.git
cd audit-anomaly-detection
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the dataset

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` folder

---

## 🚀 How to Run

### Option A — Run the full pipeline (recommended)

```bash
cd src
python main.py
```

This runs all steps automatically:
1. Loads and preprocesses data
2. Applies Z-Score and IQR statistical flagging
3. Trains the Isolation Forest model
4. Evaluates results and generates all plots
5. Exports `flagged_transactions.csv` to `outputs/`

### Option B — Run notebooks step by step

```bash
cd notebooks
jupyter notebook
```

Open `01_eda.ipynb` first, then `02_models.ipynb`. Run cells top to bottom using **Kernel → Restart & Run All**.

---

## 🔬 Methodology

### Detection Methods Used

#### 1. Z-Score Flagging
Flags transactions where any PCA feature deviates more than **3 standard deviations** from the mean. Catches outliers in feature space.

```
Flag if: |x - μ| / σ > 3.0
```

#### 2. IQR (Interquartile Range)
Flags transactions where the transaction amount falls outside the whisker boundaries. Robust to non-normal distributions.

```
Flag if: Amount < Q1 - 1.5×IQR  OR  Amount > Q3 + 1.5×IQR
```

#### 3. Isolation Forest (Main ML Model)
An unsupervised tree-based algorithm that isolates anomalies by randomly partitioning features. Anomalies require fewer splits to isolate — they get lower anomaly scores.

- `n_estimators = 100`
- `contamination = 0.032` (3.2% expected anomaly rate)
- `random_state = 42`

#### 4. Combined Risk Scoring
All three methods feed into a unified risk score (0–100):

```
Risk Score = (zscore_flag × 50) + (iqr_flag × 30) + (max_zscore × 2)
```

| Score Range | Risk Label |
|-------------|------------|
| 0 – 30 | Low |
| 31 – 60 | Medium |
| 61 – 100 | High |

---

## 📈 Power BI Dashboard

The interactive dashboard (`powerbi/dashboard.pbix`) includes:

| Visual | Purpose |
|--------|---------|
| KPI Cards | Total flagged, avg anomaly score, confirmed frauds |
| Bar Chart | Anomaly score volume by risk level |
| Scatter Plot | Anomaly score vs Z-score (multi-method validation) |
| Donut Chart | Risk label distribution (High/Medium/Low) |
| Data Table | Individual flagged transactions sortable by score |
| Slicers | Filter by Class (confirmed fraud) and risk_label |

**Key insight from the scatter plot:** Transactions in the top-right corner (high anomaly score AND high Z-score) are flagged by BOTH methods independently — these are the highest-priority cases for auditor review.

---

## 📂 Output Files Explained

| File | Description | Used by |
|------|-------------|---------|
| `flagged_transactions.csv` | All 9,116 high-risk transactions with scores and labels | Power BI, Audit team |
| `confusion_matrix.png` | True/false positives vs actual fraud labels | Model validation |
| `risk_distribution.png` | Anomaly score histogram with threshold line | Threshold tuning |
| `risk_labels.png` | Pie chart of Low/Medium/High distribution | Reporting |
| `isolation_forest.pkl` | Saved model for future inference on new data | Production deployment |

---

## 🎯 How to Tune the Model

To adjust what percentage of transactions get flagged, change the `contamination` parameter in `src/ml_model.py`:

```python
iso_forest = IsolationForest(
    contamination=0.032,  # ← increase to flag more, decrease to flag fewer
    ...
)
```

Then re-run `python main.py` and check the summary output. Target precision > 85% with flagged rate between 2–5% for a realistic audit scenario.

---

## 📋 Requirements

```
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
openpyxl==3.1.2
joblib==1.3.0
```

---

## 🗂️ Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Credits:** Machine Learning Group, Université Libre de Bruxelles (ULB)

| Property | Value |
|----------|-------|
| Rows | 284,807 transactions |
| Fraud cases | 492 (0.17%) |
| Features | V1–V28 (PCA-transformed), Amount, Time |
| Target column | Class (1 = fraud, 0 = legitimate) |

> **Note:** The dataset is not included in this repository due to size. Download it directly from Kaggle and place it in `data/creditcard.csv`.

---

## 💼 Business Context

In a real Big 4 audit engagement, this pipeline would:

1. **Replace manual sampling** — auditors traditionally sample 5–10% of transactions manually. This system intelligently targets the 3.2% most suspicious.
2. **Prioritize audit effort** — High-risk flagged transactions go to senior auditors; Medium-risk to juniors; Low-risk to automated checks.
3. **Provide defensible evidence** — The multi-method approach (statistical + ML) gives auditors two independent reasons to investigate a transaction.
4. **Scale across clients** — The `isolation_forest.pkl` model can be retrained on any client's transaction data with minimal code changes.

---

## 🔮 Future Improvements

- [ ] Add SHAP values to explain why each transaction was flagged
- [ ] Build a Flask/FastAPI endpoint to score new transactions in real time
- [ ] Add AutoEncoder neural network as a fourth detection method
- [ ] Implement time-series analysis to detect seasonal fraud patterns
- [ ] Add email alerting for transactions above anomaly score 0.75

---

## 👤 Author

**[Jidnyasa Thakre]**
- LinkedIn: [https://www.linkedin.com/in/jidnyasathakre/]
- GitHub: [https://github.com/jidnyasadthakre07/]
- Email: jidnyasathakre3@gmail.com

---


## 🙏 Acknowledgements

- Kaggle and ULB Machine Learning Group for the dataset
- Scikit-learn documentation for Isolation Forest implementation guidance
- Microsoft Power BI community for dashboard best practices

---

> *"Designed an end-to-end anomaly detection pipeline on 284,807 financial transactions using Isolation Forest, Z-Score, and IQR methods — flagging 3.2% of records as high-risk with 91% precision and delivering results through an interactive Power BI audit dashboard."*
