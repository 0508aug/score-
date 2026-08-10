import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load the dataset
# -----------------------------
data = pd.read_csv("data.csv")

# Input features
X = data[["study_hours", "attendance", "previous_score"]]

# Target/output
y = data["exam_score"]

# -----------------------------
# Train the model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Streamlit page
# -----------------------------
st.title("📚 Exam Score Predictor")

st.write("Enter your details below to predict your exam score.")

# Student inputs
study_hours = st.number_input(
    "📖 Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "🏫 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

previous_score = st.number_input(
    "📝 Previous Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Score"):

    prediction = model.predict([
        [study_hours, attendance, previous_score]
    ])

    score = prediction[0]

    # Keep score between 0 and 100
    score = max(0, min(100, score))

    st.success("Prediction completed!")

    st.metric(
        "Predicted Exam Score",
        f"{score:.2f} / 100"
    )

    # Performance message
    if score >= 90:
        st.balloons()
        st.write("🏆 Excellent performance!")

    elif score >= 75:
        st.write("👏 Very good! Keep working hard.")

    elif score >= 50:
        st.write("👍 You passed, but there is room for improvement.")

    else:
        st.write("📖 You should spend more time preparing.")

