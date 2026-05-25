import streamlit as st
import joblib

# ======================================
# Load Trained Model
# ======================================

model = joblib.load("spam_detector_model.pkl")


# ======================================
# Website Title
# ======================================
st.sidebar.title("About Project")

st.sidebar.write("""
Dataset:
SMS Spam Collection Dataset

Algorithm:
Linear SVM

Vectorization:
TF-IDF

Purpose:
Detect spam SMS messages
""")

st.title("📩 SMS Spam Detection")

# ======================================
# User Input
# ======================================

message = st.text_area(
    "Enter your SMS Message",
    height=150
)


# ======================================
# Prediction Button
# ======================================

if st.button("Predict"):

    if message.strip() == "":

        st.warning("Please enter a message")

    else:

        prediction = model.predict([message])

        if prediction[0] == 1:

            st.error("🚨 SPAM MESSAGE")

        else:

            st.success("✅ NOT SPAM")


# ======================================
# Footer
# ======================================

st.markdown("---")
st.write("Machine Learning Mini Project")
st.write("Model: TF-IDF + Linear SVM")