import streamlit as st
import requests
import pandas as pd
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Profit Prediction App", layout="wide")

# Sidebar options
st.sidebar.title("📌 Navigation")
option = st.sidebar.radio("Go to", ["📂 Upload Excel File", "⌨️ Manual Prediction", "📊 Model Info"])

# Function to send request to Flask API
def predict_from_excel(uploaded_file):
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:5000/predict_excel", files=files)
    return response

# Upload Excel File for Prediction
if option == "📂 Upload Excel File":
    st.title("📈 Profit Prediction App")
    st.write("Upload an Excel file to predict profit using the trained ElasticNet model.")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Please upload a valid Excel file.") 
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

        with st.spinner("Predicting..."):
            response = predict_from_excel(uploaded_file)

        if response.status_code == 200:
            result = response.json()
            if "output_file" in result:
                df_pred = pd.read_excel("predicted_results.xlsx")
                st.write("### Predictions")
                st.dataframe(df_pred)

                # Visualization: Actual vs. Predicted
                if "Profit" in df.columns:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.bar(df.index, df["Profit"], label="Actual Profit", color="blue", alpha=0.7)
                    ax.bar(df.index, df_pred["Predicted Profit"], label="Predicted Profit", color="red", alpha=0.7)
                    plt.xlabel("Index")
                    plt.ylabel("Profit")
                    plt.title("Actual vs. Predicted Profit")
                    plt.legend()
                    st.pyplot(fig)
                
                with open("predicted_results.xlsx", "rb") as file:
                    st.download_button("Download Predicted Excel", file, file_name="predicted_results.xlsx")
            else:
                st.error(result["error"])
        else:
            st.error("Error: Unable to get a response from the API.")

# Manual Prediction
elif option == "⌨️ Manual Prediction":
    st.title("⌨️ Manual Profit Prediction")
    st.write("Enter feature values to get an instant profit prediction.")

    feature1 = st.number_input("Feature 1", value=100000)
    feature2 = st.number_input("Feature 2", value=120000)
    feature3 = st.number_input("Feature 3", value=300000)

    if st.button("Predict Profit"):
        with st.spinner("Predicting..."):
            response = requests.post("http://127.0.0.1:5000/predict_single", json={"features": [feature1, feature2, feature3]})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Profit: ${result['predicted_profit']:.2f}")
        else:
            st.error("Error: Unable to get a response from the API.")

# Model Information & Accuracy
elif option == "📊 Model Info":
    st.title("📊 Model Information")
    st.write("### ElasticNet Regression Model")
    st.write("This model was trained to predict profit based on key business features.")

    st.metric(label="✅ Model Accuracy", value="94%")
    
    st.write("### About ElasticNet")
    st.write("ElasticNet is a hybrid of Lasso and Ridge regression, balancing L1 and L2 penalties for better feature selection and regularization.")

    st.image("https://miro.medium.com/max/875/1*iKnNXAPMT0OrnnEXChaluw.png", caption="ElasticNet Regularization")

