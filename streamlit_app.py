# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/"  # Update if hosted elsewhere

st.title("Vietnamese Text Classifier")

text_input = st.text_area("Enter your Vietnamese text:")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        response = requests.post(API_URL, json={"input": text_input})
        if response.status_code == 200:
            label = response.json().get("label", "Unknown")
            st.success(f"Predicted Label: **{label}**")
        else:
            st.error("Failed to get prediction from API.")
