import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Student Grade Predictor", layout="wide")
st.title("📊 Student Grade Predictor App")

# Upload CSV
df = None
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data Loaded Successfully!")

# Data Cleaning Function
def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df['Age'] = df['Age'].apply(lambda x: int(x.split('-')[0]) + 1 if '-' in x else int(x))
    df['Sex'] = df['Sex'].str.lower()
    df['Scholarship'] = df['Scholarship'].str.rstrip('%').astype(int)
    grade_map = {'FF': 0, 'DD': 1, 'DC': 2, 'CC': 3, 'CB': 3.5, 'BB': 4, 'BA': 4.5, 'AA': 5}
    df['Grade'] = df['Grade'].map(grade_map)
    return df

# EDA Section
def show_eda(df):
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Distribution of Grades")
        fig, ax = plt.subplots()
        sns.histplot(df['Grade'], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.write("### Feature vs Grade")
    cat_features = df.select_dtypes(include='object').columns
    selected = st.selectbox("Select Categorical Feature", cat_features)
    fig, ax = plt.subplots()
    sns.barplot(x=selected, y='Grade', data=df, ax=ax)
    st.pyplot(fig)

# Modeling Section
def train_model(df):
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('Grade', axis=1)
    y = df['Grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("🧠 Model Performance")
    st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.2f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")

    st.write("### Prediction vs Actual")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([0, 5], [0, 5], color='red')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    st.write("### Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=importances[:10], y=importances.index[:10], ax=ax)
    st.pyplot(fig)

# Main Logic
if df is not None:
    df_clean = clean_data(df)
    st.write("### Cleaned Data Preview")
    st.dataframe(df_clean.head())

    show_eda(df_clean)
    train_model(df_clean)
else:
    st.info("Please upload a CSV file to get started.")
