"""
Professional Bank Customer Churn Prediction Dashboard
----------------------------------------------------
Expected files in the same directory:
    app.py (this file)
    model.keras
    preprocessor.pkl
    Churn_Modelling.csv

Run:
    streamlit run app.py

The app expects the common Bank Customer Churn dataset with target `Exited`.
The saved preprocessor should accept these raw model features:
CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
HasCrCard, IsActiveMember, EstimatedSalary.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
MODEL_PATH = BASE_DIR / "model.keras"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"

TARGET = "Exited"
RAW_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary"
]
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


# ----------------------------- Styling -----------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8edf5;
        padding: 16px 18px;
        border-radius: 12px;
    }

    .hero {
        padding: 24px 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #14213d, #244b78);
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        color: #dbeafe;
        margin: 8px 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------- Artifact loading -----------------------------
@st.cache_resource
def load_artifacts():
    missing = [p.name for p in [MODEL_PATH, PREPROCESSOR_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_default_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model input columns in the expected order."""
    data = frame.copy()
    data = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns], errors="ignore")
    missing = [c for c in RAW_FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")
    return data[RAW_FEATURES]


def get_probabilities(model, transformed):
    predictions = np.asarray(model.predict(transformed, verbose=0))
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        return predictions[:, -1].astype(float)
    return predictions.reshape(-1).astype(float)


def make_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1-score": f1_score(y_true, predicted, zero_division=0),
    }
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_true, probabilities)
    except ValueError:
        metrics["ROC-AUC"] = np.nan
    return metrics, predicted


def plot_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ----------------------------- Load data -----------------------------
st.sidebar.title("🏦 Churn Intelligence")
st.sidebar.caption("ANN-powered bank customer analytics")
uploaded_data = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])

try:
    model, preprocessor = load_artifacts()
    artifacts_ready = True
except Exception as exc:
    model, preprocessor = None, None
    artifacts_ready = False
    st.sidebar.error(f"Model loading issue: {exc}")

try:
    if uploaded_data is not None:
        df = pd.read_csv(uploaded_data)
    else:
        df = load_default_data()
except Exception as exc:
    df = None
    st.error(f"Could not read dataset: {exc}")

if df is None:
    st.markdown("""
    <div class="hero">
      <h1>Bank Churn Intelligence</h1>
      <p>Explore customer behaviour, review ANN performance, and estimate churn risk.</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("Upload `Churn_Modelling.csv` using the sidebar, or place it beside this app.")
    st.stop()

if TARGET not in df.columns:
    st.error(f"The dataset must contain the target column `{TARGET}`.")
    st.stop()

# Clean target and preserve only valid binary target rows for evaluation.
df = df.copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)
df = df[df[TARGET].isin([0, 1])]

try:
    X = prepare_features(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

y = df[TARGET].to_numpy()
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Data Explorer", "Feature Analysis", "Training Results", "Churn Prediction"]
)

st.markdown("""
<div class="hero">
  <h1>Bank Churn Intelligence</h1>
  <p>Customer analytics • Feature exploration • ANN evaluation • Individual risk prediction</p>
</div>
""", unsafe_allow_html=True)

# Shared dataset facts
total_customers = len(df)
churn_count = int((y == 1).sum())
retained_count = int((y == 0).sum())
churn_rate = churn_count / total_customers if total_customers else 0


# ============================= OVERVIEW =============================
if page == "Overview":
    st.subheader("Business Overview")
    st.caption("A high-level snapshot of the customer base and churn distribution.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{total_customers:,}")
    c2.metric("Churned customers", f"{churn_count:,}")
    c3.metric("Retained customers", f"{retained_count:,}")
    c4.metric("Churn rate", f"{churn_rate:.1%}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### Customer outcome distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
        ax.bar(["Retained", "Churned"], counts.values)
        ax.set_ylabel("Number of customers")
        ax.set_title("Retained vs. churned customers")
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(counts.values):
            ax.text(i, value, f"{value:,}", ha="center", va="bottom")
        plot_fig(fig)

    with right:
        st.markdown("#### Dataset profile")
        profile = pd.DataFrame({
            "Measure": ["Rows", "Columns", "Duplicate rows", "Missing cells"],
            "Value": [
                f"{len(df):,}", f"{df.shape[1]:,}",
                f"{df.duplicated().sum():,}", f"{int(df.isna().sum().sum()):,}"
            ]
        })
        st.dataframe(profile, hide_index=True, use_container_width=True)

    st.markdown("#### Churn rate by selected customer groups")
    group_col = st.selectbox(
        "Choose a categorical feature",
        [c for c in ["Geography", "Gender", "IsActiveMember", "HasCrCard", "NumOfProducts"] if c in df.columns]
    )
    grouped = df.groupby(group_col, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
    grouped["mean"] *= 100
    grouped = grouped.rename(columns={"mean": "Churn rate (%)", "count": "Customers"})
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=grouped, x=group_col, y="Churn rate (%)", ax=ax)
    ax.set_title(f"Churn rate by {group_col}")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(group_col)
    ax.spines[["top", "right"]].set_visible(False)
    plot_fig(fig)
    st.dataframe(grouped, hide_index=True, use_container_width=True)


# ============================= DATA EXPLORER =============================
elif page == "Data Explorer":
    st.subheader("Dataset Explorer")
    st.caption("Inspect records, data types, completeness, and descriptive statistics.")

    t1, t2, t3 = st.tabs(["Data preview", "Data quality", "Statistics"])
    with t1:
        st.write(f"Showing {min(100, len(df))} of {len(df):,} records.")
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button(
            "Download dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bank_churn_dataset.csv",
            mime="text/csv"
        )
    with t2:
        quality = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(x) for x in df.dtypes],
            "Missing values": df.isna().sum().values,
            "Missing (%)": (df.isna().mean().values * 100).round(2),
            "Unique values": df.nunique(dropna=False).values
        })
        st.dataframe(quality, hide_index=True, use_container_width=True)
        st.metric("Duplicate rows", f"{df.duplicated().sum():,}")
        fig, ax = plt.subplots(figsize=(10, 4))
        missing = df.isna().sum().sort_values(ascending=False)
        sns.barplot(x=missing.values, y=missing.index, ax=ax)
        ax.set_title("Missing values by column")
        ax.set_xlabel("Missing cells")
        ax.set_ylabel("")
        plot_fig(fig)
    with t3:
        st.markdown("**Numerical feature summary**")
        st.dataframe(df.describe(include=[np.number]).T, use_container_width=True)
        st.markdown("**Categorical feature summary**")
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical:
            st.dataframe(df[categorical].describe().T, use_container_width=True)
        else:
            st.info("No categorical columns were detected.")


# ============================= FEATURE ANALYSIS =============================
elif page == "Feature Analysis":
    st.subheader("Feature Analysis")
    st.caption("Explore distributions, relationships, and how customer attributes vary with churn.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [c for c in numeric_cols if c != TARGET]
    categorical_features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    categorical_features += [c for c in ["HasCrCard", "IsActiveMember", "NumOfProducts", "Tenure"] if c in df.columns and c not in categorical_features]

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Churn comparison", "Correlation", "Categorical breakdown"])

    with tab1:
        if numeric_features:
            selected = st.selectbox("Select a numerical feature", numeric_features)
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.histplot(data=df, x=selected, kde=True, bins=30, ax=ax)
            ax.set_title(f"Distribution of {selected}")
            plot_fig(fig)
            st.write(df[selected].describe().to_frame("Summary"))
        else:
            st.info("No numerical features available.")

    with tab2:
        if numeric_features:
            selected = st.selectbox("Numerical feature to compare", numeric_features, key="compare_num")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x=TARGET, y=selected, ax=ax)
            ax.set_xticklabels(["Retained (0)", "Churned (1)"])
            ax.set_title(f"{selected} by churn outcome")
            plot_fig(fig)
            comparison = df.groupby(TARGET)[selected].agg(["mean", "median", "std", "count"])
            comparison.index = ["Retained", "Churned"]
            st.dataframe(comparison, use_container_width=True)
        if categorical_features:
            selected_cat = st.selectbox("Categorical feature to compare", categorical_features, key="compare_cat")
            cat_rate = df.groupby(selected_cat, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
            cat_rate["mean"] *= 100
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(data=cat_rate, x=selected_cat, y="mean", ax=ax)
            ax.set_title(f"Churn rate by {selected_cat}")
            ax.set_ylabel("Churn rate (%)")
            ax.tick_params(axis="x", rotation=25)
            plot_fig(fig)
            st.dataframe(cat_rate.rename(columns={"mean": "Churn rate (%)", "count": "Customers"}), hide_index=True)

    with tab3:
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Numerical feature correlation matrix")
        plot_fig(fig)
        st.caption("Correlation indicates association, not causation.")

    with tab4:
        if categorical_features:
            selected_cat = st.selectbox("Select category", categorical_features, key="cat_breakdown")
            counts = df[selected_cat].fillna("Missing").astype(str).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(x=counts.values, y=counts.index, ax=ax)
            ax.set_title(f"Customer count by {selected_cat}")
            ax.set_xlabel("Customers")
            plot_fig(fig)
        else:
            st.info("No categorical features available.")


# ============================= TRAINING RESULTS =============================
elif page == "Training Results":
    st.subheader("ANN Training & Evaluation Results")
    st.caption(
        "This page evaluates the saved ANN on a reproducible holdout split from the loaded dataset. "
        "It does not reconstruct the original training history unless a history file is supplied."
    )

    if not artifacts_ready:
        st.error("The trained model and preprocessor must both load successfully to evaluate the ANN.")
        st.stop()

    test_size = st.slider("Holdout test size (%)", 10, 40, 20, 5) / 100
    threshold = st.slider("Classification threshold", 0.10, 0.90, 0.50, 0.05)
    stratify = y if len(np.unique(y)) == 2 else None

    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        # The preprocessor is already fitted and is used only to transform the holdout set.
        X_test_transformed = preprocessor.transform(X_test_raw)
        probabilities = get_probabilities(model, X_test_transformed)
        metrics, predictions = make_metrics(y_test, probabilities, threshold)
    except Exception as exc:
        st.error(
            "Evaluation could not run. Check that `preprocessor.pkl` was fitted with the same "
            "feature columns and compatible scikit-learn version used by this app."
        )
        st.exception(exc)
        st.stop()

    mcols = st.columns(5)
    for col, (name, value) in zip(mcols, metrics.items()):
        col.metric(name, "N/A" if pd.isna(value) else f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"], ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plot_fig(fig)
    with right:
        st.markdown("#### ROC curve")
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, label=f"AUC = {metrics['ROC-AUC']:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Receiver operating characteristic")
            ax.legend()
            plot_fig(fig)
        else:
            st.info("ROC curve requires both target classes in the holdout data.")

    st.markdown("#### Classification report")
    report = classification_report(
        y_test, predictions, target_names=["Retained", "Churned"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("#### Prediction probability distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(probabilities, bins=25, kde=True, ax=ax)
    ax.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Predicted churn probability")
    ax.set_title("ANN output probabilities on holdout data")
    ax.legend()
    plot_fig(fig)

    result_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions,
        "Churn probability": probabilities.round(4)
    })
    st.download_button(
        "Download holdout predictions",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="ann_holdout_results.csv",
        mime="text/csv"
    )


# ============================= PREDICTION =============================
elif page == "Churn Prediction":
    st.subheader("Individual Customer Churn Prediction")
    st.caption("Enter customer information to estimate churn probability using the saved ANN.")

    if not artifacts_ready:
        st.error("The model and preprocessor artifacts are not available. Check the sidebar error.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("#### Customer profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650, step=1)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=100, value=38, step=1)
        with c2:
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5, step=1)
            balance = st.number_input("Account balance", min_value=0.0, value=75000.0, step=1000.0)
            products = st.selectbox("Number of products", [1, 2, 3, 4], index=1)
            salary = st.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0)
        with c3:
            has_card = st.selectbox("Has credit card?", ["Yes", "No"])
            active = st.selectbox("Is active member?", ["Yes", "No"])
            st.info("The model uses the same preprocessing artifact used during training.")

        submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if active == "Yes" else 0,
            "EstimatedSalary": salary
        }])

        try:
            transformed = preprocessor.transform(customer[RAW_FEATURES])
            probability = float(get_probabilities(model, transformed)[0])
            predicted_class = int(probability >= 0.5)

            st.divider()
            st.markdown("### Prediction result")
            r1, r2 = st.columns([1, 1])
            with r1:
                st.metric("Estimated churn probability", f"{probability:.1%}")
                st.progress(float(np.clip(probability, 0, 1)))
            with r2:
                if predicted_class == 1:
                    st.error("Higher predicted churn risk")
                    st.write("The ANN classifies this customer as likely to churn at the default 0.50 threshold.")
                else:
                    st.success("Lower predicted churn risk")
                    st.write("The ANN classifies this customer as likely to remain at the default 0.50 threshold.")

            st.caption(
                "This is a model estimate, not a guarantee of future customer behaviour. "
                "The result depends on the quality and representativeness of the training data."
            )
            with st.expander("View submitted feature values"):
                st.dataframe(customer, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error("Prediction failed. Confirm that the preprocessor was trained with the expected feature schema.")
            st.exception(exc)


# ----------------------------- Footer -----------------------------
st.sidebar.divider()
st.sidebar.caption("ANN Bank Churn Prediction • Streamlit")
st.sidebar.caption(f"Dataset: {len(df):,} customer records")
