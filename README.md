# 📘 Student Performance Predictor

A simple, interactive **machine learning web app** built using **Streamlit** to predict whether a student will pass or fail based on the number of hours studied. Powered by **Logistic Regression**, this app demonstrates how even a single input feature can be used to create an effective classifier.

---

## 🚀 Demo

👉 **Live App:** [Click to Open](https://student-performance-predictor-svgv8tttp3tmj4n8qk6hqz.streamlit.app/)  

---

## 📊 Features

- 🔍 Predicts pass/fail outcome based on study hours  
- 📈 Visualizes logistic regression prediction curve  
- 📥 Download the prediction result  
- 🎯 Shows accuracy and confusion matrix  
- 🌙 Responsive **dark mode UI**  
- 📱 Mobile-ready layout

---

## 🧠 Built With

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Lottie animations (optional)

---

## 📁 Dataset

The dataset (`dataset.csv`) includes two columns:

| Hours | Passed |
|-------|--------|
| 1.5   | 0      |
| 5.0   | 1      |
| 9.2   | 1      |
| ...   | ...    |

`Passed`: `1` = Passed, `0` = Failed

---

## ⚙️ How to Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/SatyamGupta001/student-performance-predictor.git
cd student-performance-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run student_predictor.py
