# app.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
ln_model = joblib.load('student_score_model.pkl')

st.title("📚 Student Score Predictor")
st.write("Predict a student's score based on hours studied.")

# User input
hours = st.number_input("Enter hours studied:", min_value=0.0, max_value=24.0, value=1.0, step=0.5)

# Prediction
if st.button("Predict Score"):
    predicted_score = ln_model.predict([[hours]])[0]
    st.success(f"Predicted Score: {predicted_score:.2f}")

# Optional: show dataset and regression line
if st.checkbox("Show Dataset and Regression Line"):
    df = pd.read_csv('rounded_hours_student_scores.csv', encoding='latin1')
    st.subheader("Dataset")
    st.dataframe(df)

    plt.scatter(df['Hours'], df['Scores'], color='red')
    plt.plot(df['Hours'], ln_model.predict(df[['Hours']]), color='blue')
    plt.xlabel("Hours Studied")
    plt.ylabel("Scores")
    plt.title("Hours vs Scores Regression")
    st.pyplot(plt)
