import streamlit as st

# Set this FIRST
st.set_page_config(page_title="Student Performance Predictor", page_icon="📘", layout="wide")

# Then imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from streamlit_lottie import st_lottie

# ---------------- DARK THEME ---------------- #
def set_dark_theme():
    st.markdown("""
        <style>
            body {
                background-color: #121212;
                color: #f1f1f1;
            }
            .stApp {
                background-color: #121212;
                color: #ffffff;
            }
            .css-1v0mbdj, .css-1y4p8pa {
                color: #ffffff !important;
            }
            .stSlider > div > div {
                background-color: #333 !important;
            }
            .stSelectbox > div > div {
                background-color: #333 !important;
                color: #fff !important;
            }
            @media (max-width: 768px) {
                .st-emotion-cache-1lcbmhc {  /* main container */
                    padding: 1rem !important;
                }
                .st-emotion-cache-10trblm {  /* title class */
                    font-size: 1.5rem !important;
                }
            }
        </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# ---------------- APP TITLE ---------------- #
st.title("📘 Student Performance Predictor")
st.write("Predict whether a student will pass or fail based on hours studied using logistic regression.")

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv('dataset.csv')  # Must have 'Hours' and 'Passed' columns

X = data[['Hours']]
y = data['Passed']

# ---------------- TRAIN MODEL ---------------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------- EVALUATE ---------------- #
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.markdown(f"✅ **Model Accuracy:** `{round(acc * 100, 2)}%`")
with st.expander("🔢 View Confusion Matrix"):
    st.text(cm)

# -------------------- INPUT UI -------------------- #
st.subheader("🎯 Predict From Your Input")

# Use responsive columns
col1, col2 = st.columns([1, 2])

with col1:
    hours = st.slider("How many hours did the student study?", 0.0, 12.0, 5.0, 0.1)

with col2:
    pred = model.predict([[hours]])[0]
    prob = model.predict_proba([[hours]])[0][1]

    st.success(f"Prediction: You will **{'Pass ✅' if pred == 1 else 'Fail ❌'}**")
    st.info(f"Probability of Passing: {round(prob * 100, 2)}%")

# ---------------- VISUALIZATION ---------------- #
st.subheader("📈 Logistic Regression Curve")
fig, ax = plt.subplots()
ax.scatter(data['Hours'], data['Passed'], color='skyblue', label='Actual Data')
x_values = np.linspace(0, 12, 100).reshape(-1, 1)
y_values = model.predict_proba(x_values)[:, 1]
ax.plot(x_values, y_values, color='orange', label='Prediction Curve')
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Probability of Passing')
ax.set_title('Prediction Trend')
ax.legend()
ax.grid(True)
st.pyplot(fig, use_container_width=True)

# ---------------- DOWNLOAD ---------------- #
result_text = f"Study Hours: {hours} hrs\nPrediction: {'Pass' if pred == 1 else 'Fail'}\nConfidence: {round(prob * 100, 2)}%"
st.download_button("📥 Download Your Result", result_text, file_name="student_prediction.txt")
