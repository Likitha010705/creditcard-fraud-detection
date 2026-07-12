"""
Credit Card Fraud Detection -- Streamlit App
==============================================
Interactive exploration, model training, and prediction interface for the
Kaggle "Credit Card Fraud Detection" dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Run locally:
    streamlit run app.py

See README.md for full setup and deployment instructions.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import model as ml
from src import visualization as viz

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
defaults = {
    "df": None,
    "data_source": None,
    "train_result": None,
    "feature_cols": None,
    "trained_algo": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------------------------------------------
# Sidebar: data loading + navigation
# --------------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
st.sidebar.caption("Kaggle: mlg-ulb/creditcardfraud")

st.sidebar.subheader("1. Load Data")
data_choice = st.sidebar.radio(
    "Data source",
    ["Use bundled sample data", "Upload creditcard.csv"],
    help="The bundled sample is synthetic (same schema, small size) so the "
         "app works out of the box. For real results, upload the actual "
         "Kaggle creditcard.csv.",
)

if data_choice == "Use bundled sample data":
    if st.session_state["data_source"] != "sample" or st.session_state["df"] is None:
        st.session_state["df"] = dl.basic_clean(dl.load_sample_data())
        st.session_state["data_source"] = "sample"
        st.session_state["train_result"] = None
else:
    uploaded = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])
    if uploaded is not None:
        raw = dl.load_uploaded_data(uploaded)
        ok, msg = dl.validate_schema(raw)
        if ok:
            st.session_state["df"] = dl.basic_clean(raw)
            st.session_state["data_source"] = "upload"
            st.session_state["train_result"] = None
        else:
            st.sidebar.error(msg)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Navigate")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Data Explorer", "📊 EDA", "🤖 Model Training", "🔮 Fraud Prediction", "📖 About"],
    label_visibility="collapsed",
)

df = st.session_state["df"]

# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown(
        """
        This app explores and models the **Kaggle Credit Card Fraud Detection**
        dataset -- 284,807 real European card transactions from September 2013,
        where features `V1`-`V28` are PCA-transformed for confidentiality and
        only `Time`, `Amount`, and `Class` (0 = legitimate, 1 = fraud) keep
        their original meaning.

        **[Dataset on Kaggle →](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
        """
    )

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        n_total = len(df)
        n_fraud = int(df["Class"].sum())
        fraud_pct = 100 * n_fraud / n_total if n_total else 0
        c1.metric("Total Transactions", f"{n_total:,}")
        c2.metric("Fraudulent", f"{n_fraud:,}")
        c3.metric("Fraud Rate", f"{fraud_pct:.3f}%")
        c4.metric(
            "Data Source",
            "Bundled sample" if st.session_state["data_source"] == "sample" else "Uploaded file",
        )

        if st.session_state["data_source"] == "sample":
            st.info(
                "You're viewing the **bundled synthetic sample** (statistically similar "
                "shape, not real transactions). Upload the real `creditcard.csv` from "
                "Kaggle in the sidebar for genuine results.",
                icon="ℹ️",
            )

        st.subheader("Quick Look")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("No data loaded yet -- choose a data source in the sidebar.")

    st.markdown("### How to use this app")
    st.markdown(
        """
        1. **Load Data** -- use the bundled sample or upload the real Kaggle CSV.
        2. **Data Explorer** -- inspect shape, types, and class balance.
        3. **EDA** -- visualize class imbalance, amount/time patterns, correlations.
        4. **Model Training** -- pick an algorithm and imbalance-handling
           strategy, train, and review metrics built for imbalanced data
           (precision, recall, ROC-AUC, PR-AUC).
        5. **Fraud Prediction** -- test the trained model on a transaction.
        """
    )

# --------------------------------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------------------------------
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")
    if df is None:
        st.warning("Load data from the sidebar first.")
    else:
        st.subheader("Shape & Types")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])

        st.subheader("Class Balance")
        st.dataframe(dl.class_balance_summary(df), use_container_width=True)
        st.plotly_chart(viz.class_distribution_chart(df), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

        st.subheader("Missing Values")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing.rename("Missing Count"), use_container_width=True)

        st.subheader("Raw Data Preview")
        n_rows = st.slider("Rows to preview", 5, 100, 20)
        st.dataframe(df.head(n_rows), use_container_width=True)

# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    if df is None:
        st.warning("Load data from the sidebar first.")
    else:
        st.plotly_chart(viz.class_distribution_chart(df), use_container_width=True)

        st.plotly_chart(viz.amount_distribution_chart(df), use_container_width=True)

        st.plotly_chart(viz.time_distribution_chart(df), use_container_width=True)

        st.subheader("Feature Distribution by Class")
        v_cols = [c for c in df.columns if c.startswith("V")]
        feature = st.selectbox("Choose a PCA feature", v_cols, index=v_cols.index("V17") if "V17" in v_cols else 0)
        st.plotly_chart(viz.feature_boxplots(df, feature), use_container_width=True)
        st.caption(
            "Features like V17, V14, V12, and V10 tend to separate fraud from "
            "legitimate transactions most strongly in the original dataset."
        )

        st.subheader("Correlation Heatmap")
        default_cols = v_cols[:10] + ["Amount", "Class"]
        chosen_cols = st.multiselect(
            "Columns to include", options=list(df.columns), default=default_cols
        )
        if len(chosen_cols) >= 2:
            st.plotly_chart(viz.correlation_heatmap(df, chosen_cols), use_container_width=True)
        else:
            st.info("Select at least 2 columns to render the heatmap.")

# --------------------------------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------------------------------
elif page == "🤖 Model Training":
    st.title("🤖 Model Training & Evaluation")
    if df is None:
        st.warning("Load data from the sidebar first.")
    else:
        st.markdown(
            "Fraud detection is a **highly imbalanced classification** problem. "
            "Accuracy alone is misleading here -- watch **Recall** (catching "
            "fraud), **Precision** (avoiding false alarms), and **PR-AUC**."
        )

        c1, c2, c3 = st.columns(3)
        algo_name = c1.selectbox("Algorithm", list(ml.MODEL_REGISTRY.keys()))
        resample_strategy = c2.selectbox(
            "Imbalance handling",
            ["None", "Class weights", "SMOTE (oversample minority)", "Random undersampling"],
        )
        test_size = c3.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

        use_class_weight = resample_strategy == "Class weights"
        actual_resample = "None" if use_class_weight else resample_strategy

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                X, y, feature_cols = dl.get_feature_target_split(df)
                X_train, X_test, y_train, y_test = dl.train_test_split_data(X, y, test_size=test_size)
                try:
                    result = ml.train_and_evaluate(
                        X_train, X_test, y_train, y_test,
                        algo_name=algo_name,
                        resample_strategy=actual_resample,
                        use_class_weight=use_class_weight,
                        feature_cols=feature_cols,
                    )
                    st.session_state["train_result"] = result
                    st.session_state["feature_cols"] = feature_cols
                    st.session_state["trained_algo"] = algo_name
                    st.success(f"{algo_name} trained successfully.")
                except ImportError:
                    st.error(
                        "SMOTE/undersampling require the `imbalanced-learn` package. "
                        "It's included in requirements.txt -- reinstall with "
                        "`pip install -r requirements.txt`."
                    )

        result = st.session_state["train_result"]
        if result is not None:
            st.subheader(f"Results -- {st.session_state['trained_algo']}")
            m = result.metrics
            cols = st.columns(len(m))
            for col, (k, v) in zip(cols, m.items()):
                col.metric(k, f"{v:.4f}" if not np.isnan(v) else "N/A")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.confusion_matrix_chart(result.confusion), use_container_width=True)
            with c2:
                st.plotly_chart(
                    viz.roc_curve_chart(result.fpr, result.tpr, m["ROC-AUC"]),
                    use_container_width=True,
                )

            st.plotly_chart(
                viz.precision_recall_chart(result.precision_curve, result.recall_curve, m["PR-AUC (Avg Precision)"]),
                use_container_width=True,
            )

            imp_df = ml.get_feature_importance(result.model, result.feature_cols)
            if imp_df is not None:
                st.plotly_chart(viz.feature_importance_chart(imp_df), use_container_width=True)

            os.makedirs("models", exist_ok=True)
            if st.button("💾 Save trained model to models/fraud_model.pkl"):
                ml.save_model(result.model, "models/fraud_model.pkl")
                st.success("Model saved to models/fraud_model.pkl")

# --------------------------------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------------------------------
elif page == "🔮 Fraud Prediction":
    st.title("🔮 Fraud Prediction")
    result = st.session_state["train_result"]
    if df is None:
        st.warning("Load data from the sidebar first.")
    elif result is None:
        st.warning("Train a model on the **Model Training** page first.")
    else:
        st.markdown(
            "Pick a transaction from the loaded dataset (or randomize one) and "
            "see what the trained model predicts."
        )

        c1, c2 = st.columns([3, 1])
        max_idx = len(df) - 1
        idx = c1.number_input("Row index", min_value=0, max_value=max_idx, value=0, step=1)
        if c2.button("🎲 Random transaction"):
            idx = int(np.random.randint(0, max_idx + 1))
            st.session_state["_rand_idx"] = idx
        idx = st.session_state.get("_rand_idx", idx)

        row = df.iloc[[idx]]
        st.dataframe(row, use_container_width=True)

        # Scale the whole dataset once so Time/Amount scaling matches what
        # the model was trained on, then pull out just this row's features.
        full_X, _, feature_cols = dl.get_feature_target_split(df)
        row_features = full_X.iloc[[idx]][result.feature_cols]

        model = result.model
        pred = model.predict(row_features)[0]
        proba = (
            model.predict_proba(row_features)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        actual = int(row["Class"].values[0])
        st.subheader("Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", "🚨 Fraud" if pred == 1 else "✅ Legitimate")
        c2.metric("Fraud Probability", f"{proba:.2%}" if proba is not None else "N/A")
        c3.metric("Actual Label", "🚨 Fraud" if actual == 1 else "✅ Legitimate")

        if pred == actual:
            st.success("Model prediction matches the actual label.")
        else:
            st.error("Model prediction does NOT match the actual label.")

# --------------------------------------------------------------------------
# ABOUT
# --------------------------------------------------------------------------
elif page == "📖 About":
    st.title("📖 About This Project")
    st.markdown(
        """
        ### Dataset
        [Credit Card Fraud Detection (Kaggle, mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) --
        284,807 transactions made by European cardholders in September 2013,
        with 492 labeled as fraud (~0.172%). Features `V1`-`V28` are the
        result of a PCA transformation for confidentiality; only `Time`
        (seconds since the first transaction) and `Amount` are untransformed.

        ### Why not accuracy alone?
        With fraud at ~0.17% of transactions, a model that predicts
        "legitimate" for everything scores over 99.8% accuracy while
        catching zero fraud. This app surfaces **Precision**, **Recall**,
        **F1**, **ROC-AUC**, and **PR-AUC** instead, and offers imbalance
        handling (class weighting, SMOTE, undersampling) on the training
        split.

        ### Project structure
        ```
        creditcard-fraud-detection/
        ├── app.py                    # Streamlit app (this UI)
        ├── src/
        │   ├── data_loader.py        # loading, cleaning, scaling, splitting
        │   ├── model.py              # training, resampling, metrics
        │   └── visualization.py      # Plotly chart builders
        ├── data/
        │   ├── generate_sample_data.py
        │   └── sample_creditcard.csv # bundled synthetic demo data
        ├── models/                   # saved .pkl models land here
        ├── requirements.txt
        ├── Procfile / render.yaml    # deployment configs
        └── README.md
        ```

        Built with Streamlit, scikit-learn, imbalanced-learn, and Plotly.
        """
    )
