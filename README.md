# 💳 Credit Card Fraud Detection — Streamlit App

An interactive Streamlit application for exploring and modeling the Kaggle
**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
dataset — 284,807 European card transactions from September 2013, with
492 (~0.172%) labeled as fraud.

The app covers the full workflow: load data → explore it → train a
classifier with proper handling of extreme class imbalance → evaluate it
with the right metrics → run live predictions on individual transactions.

---

## ✨ Features

- **Flexible data loading** — use the bundled synthetic sample out of the
  box, or upload the real `creditcard.csv` from Kaggle.
- **Data Explorer** — shape, dtypes, class balance, descriptive stats,
  missing-value check.
- **EDA dashboard** — class distribution, amount/time distributions split
  by class, per-feature box plots, correlation heatmap.
- **Model training** — Logistic Regression, Random Forest, or Gradient
  Boosting, with a choice of imbalance-handling strategy:
  - None
  - Class weighting (`class_weight="balanced"`)
  - SMOTE oversampling
  - Random undersampling
- **Imbalance-aware evaluation** — accuracy, precision, recall, F1,
  ROC-AUC, PR-AUC, confusion matrix, ROC curve, precision-recall curve,
  and feature importance/coefficients.
- **Live fraud prediction** — pick (or randomize) a transaction and see
  the trained model's prediction and fraud probability.
- **Deployment-ready** — includes config for Render.com and any
  Procfile-based host, plus Streamlit theming.

---

## 📁 Project Structure

```
creditcard-fraud-detection/
├── app.py                      # Streamlit app entry point
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # load / validate / clean / scale / split
│   ├── model.py                # training, resampling, metrics
│   └── visualization.py        # Plotly chart builders
├── data/
│   ├── generate_sample_data.py # regenerate the synthetic sample
│   └── sample_creditcard.csv   # bundled synthetic demo dataset
├── models/                     # trained models get saved here (.pkl)
├── notebooks/                  # optional space for exploratory notebooks
├── .streamlit/
│   └── config.toml             # theme + server settings
├── requirements.txt
├── Procfile                    # for Render / Heroku-style deploys
├── render.yaml                 # Render.com blueprint
├── runtime.txt                 # Python version pin
├── .gitignore
└── README.md
```

---

## 🗂️ About the Dataset

The real dataset is **not bundled in this repo** — it's ~150 MB and
distributed by Kaggle under its own terms, so you'll need a free Kaggle
account to download it yourself:

1. Go to <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.
2. Click **Download** (or use the Kaggle API: `kaggle datasets download -d mlg-ulb/creditcardfraud`).
3. Unzip and place `creditcard.csv` in the `data/` folder, **or** just
   upload it directly through the app's sidebar — no need to restart or
   redeploy.

**Schema** (both the real file and the bundled sample share this):

| Column | Description |
|---|---|
| `Time` | Seconds elapsed between this transaction and the first transaction in the dataset |
| `V1`–`V28` | Principal components from a PCA transformation (original features withheld for confidentiality) |
| `Amount` | Transaction amount |
| `Class` | Target: `0` = legitimate, `1` = fraud |

### About the bundled sample data

`data/sample_creditcard.csv` is a **synthetic** 6,000-row dataset generated
by `data/generate_sample_data.py`. It mimics the real dataset's schema,
scale, and ~0.17% fraud rate so the app is fully functional immediately
after cloning — but it is **not real transaction data** and any model
trained on it is a demo, not a production-grade fraud detector. Swap in
the real Kaggle file for meaningful results. To regenerate or resize the
sample:

```bash
cd data
python generate_sample_data.py
```

---

## 🚀 Getting Started (Local)

### 1. Clone / unzip and enter the project

```bash
cd creditcard-fraud-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to switch
between the bundled sample and an uploaded copy of the real dataset.

---

## ☁️ Deployment

### Option A — Streamlit Community Cloud (simplest, free)

1. Push this project to a GitHub repository.
2. Go to <https://share.streamlit.io>, sign in, and click **New app**.
3. Point it at your repo, branch, and `app.py`.
4. Deploy. Streamlit Cloud reads `requirements.txt` automatically.

> Note: the real `creditcard.csv` isn't in the repo (see `.gitignore`).
> Once deployed, use the in-app **file uploader** to supply it — uploaded
> files aren't persisted between sessions on Streamlit Cloud, so this
> keeps the deployment lightweight and Kaggle-compliant.

### Option B — Render.com

This repo includes a ready-to-use `render.yaml` blueprint.

1. Push the project to GitHub.
2. In the Render dashboard, choose **New → Blueprint** and select your repo.
   Render will detect `render.yaml` and configure the service automatically.
3. Alternatively, create a **New → Web Service** manually with:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Deploy. Render assigns a public URL automatically.

### Option C — Any Procfile-based host (Heroku, Railway, etc.)

The included `Procfile` and `runtime.txt` work as-is:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Option D — Docker (optional, bring your own Dockerfile)

If your platform expects a container, a minimal Dockerfile looks like:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🧠 Modeling Notes

Fraud is ~0.17% of transactions in the real dataset, so a model that
always predicts "legitimate" would be >99.8% accurate while catching zero
fraud. This app is built around that reality:

- **Stratified train/test split** keeps fraud proportion consistent
  across splits.
- **Resampling is applied to the training split only** — the test split
  always reflects real-world class proportions so metrics stay honest.
- **Metrics beyond accuracy**: precision, recall, F1, ROC-AUC, and
  PR-AUC (average precision) are all surfaced, since PR-AUC in particular
  is more informative than ROC-AUC under heavy imbalance.
- **Three algorithms** are available out of the box (Logistic Regression,
  Random Forest, Gradient Boosting); the `src/model.py:MODEL_REGISTRY`
  dict is written so adding another `sklearn`-compatible estimator is a
  one-line change.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [scikit-learn](https://scikit-learn.org/) — models & metrics
- [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE / undersampling
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

---

## 📄 License & Data Attribution

This project's code is provided as-is for educational/portfolio use. The
dataset itself is © the original authors and distributed via Kaggle;
please review Kaggle's terms and the dataset's own license before
redistributing it. Citation for the dataset (as requested by its
maintainers):

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
> Bontempi. *Calibrating Probability with Undersampling for Unbalanced
> Classification.* In Symposium on Computational Intelligence and Data
> Mining (CIDM), IEEE, 2015.

---

## 🙋 Troubleshooting

| Issue | Fix |
|---|---|
| `ImportError: imbalanced-learn` | `pip install -r requirements.txt` again — it's pinned there. |
| Upload fails with "Missing expected columns" | Confirm your CSV is the unmodified Kaggle `creditcard.csv` (needs `Time`, `V1`-`V28`, `Amount`, `Class`). |
| App is slow to train on the real dataset | Random Forest / Gradient Boosting on 280k+ rows takes longer than Logistic Regression — this is expected; consider a smaller test size split or fewer estimators for quick iteration. |
| Deployed app has no data | Real `creditcard.csv` is intentionally excluded from git — upload it via the sidebar after deploying. |
