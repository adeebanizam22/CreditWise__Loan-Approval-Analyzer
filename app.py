import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="LoanIQ", page_icon="💳", layout="wide")

st.title("💳 LoanIQ — Smart Approval Predictor")
st.markdown("An end-to-end machine learning project that predicts loan approval using multiple classification algorithms.")
st.markdown("---")

@st.cache_data
def train_all_models():
    df = pd.read_csv("loan_approval_data.csv")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    df = df.drop("Applicant_ID", axis=1)

    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

    cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
    df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train all 3 models
    log_model = LogisticRegression()
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)

    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    nb_pred = nb_model.predict(X_test_scaled)

    results = {
        "Logistic Regression": {
            "model": log_model, "pred": log_pred,
            "accuracy": accuracy_score(y_test, log_pred),
            "precision": precision_score(y_test, log_pred),
            "recall": recall_score(y_test, log_pred),
            "f1": f1_score(y_test, log_pred),
            "cm": confusion_matrix(y_test, log_pred)
        },
        "KNN": {
            "model": knn_model, "pred": knn_pred,
            "accuracy": accuracy_score(y_test, knn_pred),
            "precision": precision_score(y_test, knn_pred),
            "recall": recall_score(y_test, knn_pred),
            "f1": f1_score(y_test, knn_pred),
            "cm": confusion_matrix(y_test, knn_pred)
        },
        "Naive Bayes": {
            "model": nb_model, "pred": nb_pred,
            "accuracy": accuracy_score(y_test, nb_pred),
            "precision": precision_score(y_test, nb_pred),
            "recall": recall_score(y_test, nb_pred),
            "f1": f1_score(y_test, nb_pred),
            "cm": confusion_matrix(y_test, nb_pred)
        }
    }

    return results, nb_model, scaler, ohe, X.columns.tolist(), y_test

results, nb_model, scaler, ohe, feature_cols, y_test = train_all_models()

# ===================== TABS =====================
tab1, tab2 = st.tabs(["📊 Model Comparison", "🔍 Make a Prediction"])

# ===================== TAB 1: MODEL COMPARISON =====================
with tab1:
    st.subheader("📊 How All 3 Models Performed")
    st.markdown("I trained and evaluated three classification algorithms on the same dataset. Here's how they compare:")

    # Summary table
    summary_data = {
        "Model": ["Logistic Regression", "KNN (k=5)", "Naive Bayes ⭐ Best"],
        "Accuracy": [f"{results['Logistic Regression']['accuracy']*100:.1f}%",
                     f"{results['KNN']['accuracy']*100:.1f}%",
                     f"{results['Naive Bayes']['accuracy']*100:.1f}%"],
        "Precision": [f"{results['Logistic Regression']['precision']*100:.1f}%",
                      f"{results['KNN']['precision']*100:.1f}%",
                      f"{results['Naive Bayes']['precision']*100:.1f}%"],
        "Recall":    [f"{results['Logistic Regression']['recall']*100:.1f}%",
                      f"{results['KNN']['recall']*100:.1f}%",
                      f"{results['Naive Bayes']['recall']*100:.1f}%"],
        "F1 Score":  [f"{results['Logistic Regression']['f1']*100:.1f}%",
                      f"{results['KNN']['f1']*100:.1f}%",
                      f"{results['Naive Bayes']['f1']*100:.1f}%"],
    }
    st.table(pd.DataFrame(summary_data))

    st.markdown("---")

    # Bar chart comparison
    st.subheader("📈 Metric Comparison Chart")
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    model_names = ["Logistic Regression", "KNN", "Naive Bayes"]
    values = {
        "Logistic Regression": [results["Logistic Regression"]["accuracy"],
                                  results["Logistic Regression"]["precision"],
                                  results["Logistic Regression"]["recall"],
                                  results["Logistic Regression"]["f1"]],
        "KNN":                  [results["KNN"]["accuracy"],
                                  results["KNN"]["precision"],
                                  results["KNN"]["recall"],
                                  results["KNN"]["f1"]],
        "Naive Bayes":          [results["Naive Bayes"]["accuracy"],
                                  results["Naive Bayes"]["precision"],
                                  results["Naive Bayes"]["recall"],
                                  results["Naive Bayes"]["f1"]],
    }

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (name, vals) in enumerate(values.items()):
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Confusion matrices
    st.subheader("🔲 Confusion Matrices")
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Rejected", "Approved"],
                    yticklabels=["Rejected", "Approved"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    st.info("✅ **Why Naive Bayes?** It achieved the highest precision (81.1%), meaning it has the lowest chance of wrongly approving a bad loan — which is the most important metric in loan approval scenarios.")

# ===================== TAB 2: PREDICTION =====================
with tab2:
    st.subheader("🔍 Predict Loan Approval")
    st.markdown("Fill in the applicant details in the sidebar and click **Predict** to get an instant result.")

    # ---------- SIDEBAR INPUTS ----------
    st.sidebar.header("📋 Applicant Details")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single"])
    education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    employer_cat = st.sidebar.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])
    loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
    property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    st.sidebar.markdown("---")
    age = st.sidebar.slider("Age", 21, 59, 35)
    dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3])
    existing_loans = st.sidebar.selectbox("Existing Loans", [0, 1, 2, 3, 4])

    st.sidebar.markdown("---")
    applicant_income = st.sidebar.number_input("Applicant Income (₹)", 2000, 20000, 10000, step=500)
    coapplicant_income = st.sidebar.number_input("Co-applicant Income (₹)", 0, 10000, 3000, step=500)
    savings = st.sidebar.number_input("Savings (₹)", 0, 20000, 5000, step=500)
    collateral_value = st.sidebar.number_input("Collateral Value (₹)", 0, 50000, 20000, step=1000)
    loan_amount = st.sidebar.number_input("Loan Amount (₹)", 1000, 40000, 15000, step=1000)
    loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    credit_score = st.sidebar.slider("Credit Score", 550, 800, 680)
    dti_ratio = st.sidebar.slider("DTI Ratio", 0.10, 0.60, 0.35, step=0.01)

    # ---------- PREDICT BUTTON ----------
    if st.button("🔍 Predict Loan Approval"):

        education_encoded = 0 if education == "Graduate" else 1

        cat_input = pd.DataFrame([[employment, marital_status, loan_purpose, property_area, gender, employer_cat]],
                                  columns=["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"])
        cat_encoded = ohe.transform(cat_input)
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(
            ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]))

        dti_sq = dti_ratio ** 2
        credit_sq = credit_score ** 2

        num_input = pd.DataFrame([[applicant_income, coapplicant_income, age, dependents,
                                    existing_loans, savings, collateral_value,
                                    loan_amount, loan_term, education_encoded]],
                                  columns=["Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
                                           "Existing_Loans", "Savings", "Collateral_Value",
                                           "Loan_Amount", "Loan_Term", "Education_Level"])

        final_input = pd.concat([num_input.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)
        final_input["DTI_Ratio_sq"] = dti_sq
        final_input["Credit_Score_sq"] = credit_sq
        final_input = final_input[feature_cols]
        final_scaled = scaler.transform(final_input)

        prediction = nb_model.predict(final_scaled)[0]
        probability = nb_model.predict_proba(final_scaled)[0]

        st.markdown("### 📊 Prediction Result")

        if prediction == 1:
            st.success("✅ Loan is likely to be APPROVED!")
            st.metric("Approval Probability", f"{probability[1]*100:.1f}%")
        else:
            st.error("❌ Loan is likely to be REJECTED.")
            st.metric("Rejection Probability", f"{probability[0]*100:.1f}%")

        st.markdown("---")
        st.markdown("### 📝 Applicant Summary")
        summary = {
            "Gender": gender, "Age": age, "Marital Status": marital_status,
            "Education": education, "Employment": employment,
            "Applicant Income": f"₹{applicant_income:,}", "Loan Amount": f"₹{loan_amount:,}",
            "Credit Score": credit_score, "DTI Ratio": dti_ratio
        }
        st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))

st.markdown("---")
st.caption("LoanIQ — Built with Naive Bayes · Scikit-learn · Streamlit | By Nadeeba")