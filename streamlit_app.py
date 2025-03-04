import streamlit as st
import requests
import pandas as pd
import io

st.title("📊 Profit Prediction App (CSV)")
st.write("Upload a CSV file to predict profit using the trained ElasticNet model.")

# File uploader (CSV only)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Show uploaded file preview
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # Send file to Flask API
    with st.spinner("Predicting..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict_csv", files=files)

    # Display results
    if response.status_code == 200:
        result = response.json()
        if "output_file" in result:
            st.success("Prediction completed! Download results below.")
            with open("predicted_results.csv", "rb") as file:
                st.download_button("Download Predicted CSV", file, file_name="predicted_results.csv")
        else:
            st.error(result["error"])
    else:
        st.error("Error: Unable to get a response from the API.")
