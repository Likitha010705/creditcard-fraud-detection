"""
visualization.py
-----------------
Plotly chart builders used across the Streamlit app. Centralizing these
keeps app.py focused on layout/flow rather than chart construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
FRAUD_COLOR = "#E4572E"
LEGIT_COLOR = "#2E86AB"


def class_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Class"].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=[counts.get(0, 0), counts.get(1, 0)],
                marker_color=[LEGIT_COLOR, FRAUD_COLOR],
                text=[counts.get(0, 0), counts.get(1, 0)],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Transaction Class Distribution",
        yaxis_title="Number of Transactions",
        template=TEMPLATE,
        height=380,
    )
    return fig


def amount_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Amount",
        color=df["Class"].map({0: "Legitimate", 1: "Fraud"}),
        color_discrete_map={"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR},
        nbins=60,
        marginal="box",
        opacity=0.75,
        log_y=True,
        barmode="overlay",
    )
    fig.update_layout(
        title="Transaction Amount Distribution by Class (log scale y-axis)",
        template=TEMPLATE,
        legend_title="Class",
        height=420,
    )
    return fig


def time_distribution_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["Hour"] = (plot_df["Time"] % 86400) // 3600
    fig = px.histogram(
        plot_df,
        x="Hour",
        color=plot_df["Class"].map({0: "Legitimate", 1: "Fraud"}),
        color_discrete_map={"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR},
        nbins=24,
        barmode="overlay",
        opacity=0.75,
        histnorm="percent",
    )
    fig.update_layout(
        title="Transactions by Hour of Day (% within each class)",
        template=TEMPLATE,
        xaxis_title="Hour of Day (0-23)",
        legend_title="Class",
        height=380,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, cols) -> go.Figure:
    corr = df[cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(title="Feature Correlation Heatmap", template=TEMPLATE, height=600)
    return fig


def feature_boxplots(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.box(
        df,
        x=df["Class"].map({0: "Legitimate", 1: "Fraud"}),
        y=feature,
        color=df["Class"].map({0: "Legitimate", 1: "Fraud"}),
        color_discrete_map={"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR},
    )
    fig.update_layout(
        title=f"{feature} Distribution by Class",
        template=TEMPLATE,
        showlegend=False,
        height=380,
        xaxis_title="",
    )
    return fig


def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    labels = ["Legitimate", "Fraud"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18},
        )
    )
    fig.update_layout(title="Confusion Matrix", template=TEMPLATE, height=420)
    return fig


def roc_curve_chart(fpr, tpr, auc_score) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc_score:.3f})",
                              line=dict(color=FRAUD_COLOR, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=TEMPLATE,
        height=420,
    )
    return fig


def precision_recall_chart(precision, recall, ap_score) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                              name=f"PR (AP = {ap_score:.3f})",
                              line=dict(color=LEGIT_COLOR, width=3)))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template=TEMPLATE,
        height=420,
    )
    return fig


def feature_importance_chart(imp_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    plot_df = imp_df.head(top_n).sort_values(imp_df.columns[1])
    fig = go.Figure(
        go.Bar(
            x=plot_df[imp_df.columns[1]],
            y=plot_df["Feature"],
            orientation="h",
            marker_color=LEGIT_COLOR,
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        template=TEMPLATE,
        height=450,
    )
    return fig
