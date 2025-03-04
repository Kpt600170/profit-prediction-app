import streamlit as st
import requests
import pandas as pd
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Profit Prediction App", layout="wide")

# Sidebar options
st.sidebar.title("📌 Navigation")
option = st.sidebar.radio("Go to", ["📂 Upload Excel File", "⌨️ Manual Prediction", "📊 Model Info"])

# Function to send request to Flask API for Excel prediction
def predict_from_excel(uploaded_file):
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif file_extension == "xls":
            df = pd.read_excel(uploaded_file, engine="xlrd")
        elif file_extension == "csv":  # ✅ Handling CSV files
            df = pd.read_csv(uploaded_file)
        else:
            return {"error": "Unsupported file format. Please upload an Excel (.xls, .xlsx) or CSV (.csv) file."}

        # Convert DataFrame to JSON for API request
        json_data = {"features": df.values.tolist()}  
        
        # Send request to Flask API
        response = requests.post("http://127.0.0.1:5000/predict_excel", json=json_data)
        
        return response

    except Exception as e:
        return {"error": str(e)}

# 📂 Upload Excel File for Prediction
if option == "📂 Upload Excel File":
    st.title("📈 Profit Prediction App")
    st.write("Upload an Excel or CSV file to predict profit using the trained ElasticNet model.")

    # Initialize file uploader
    uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()

            if file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            elif file_extension == "xls":
                df = pd.read_excel(uploaded_file, engine="xlrd")
            elif file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload an Excel (.xls, .xlsx) or CSV (.csv) file.")
                st.stop()  # Stop execution if the file format is wrong

            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())  

            with st.spinner("Predicting..."):
                response = predict_from_excel(uploaded_file)

            if isinstance(response, dict) and "error" in response:
                st.error(response["error"])
            elif response.status_code == 200:
                result = response.json()

                if "predicted_profit" in result:
                    df["Predicted Profit"] = result["predicted_profit"]
                    st.write("### Predictions")
                    st.dataframe(df)

                    # Visualization: Actual vs. Predicted
                    if "Profit" in df.columns:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.bar(df.index, df["Profit"], label="Actual Profit", color="blue", alpha=0.7)
                        ax.bar(df.index, df["Predicted Profit"], label="Predicted Profit", color="red", alpha=0.7)
                        plt.xlabel("Index")
                        plt.ylabel("Profit")
                        plt.title("Actual vs. Predicted Profit")
                        plt.legend()
                        st.pyplot(fig)

                    # Downloadable file
                    output = io.BytesIO()
                    df.to_csv(output, index=False)
                    output.seek(0)
                    st.download_button("Download Predicted CSV", output, file_name="predicted_results.csv")

                else:
                    st.error(result.get("error", "Unexpected response format."))

            else:
                st.error("Error: Unable to get a response from the API.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ⌨️ Manual Prediction
# Manual Prediction
elif option == "⌨️ Manual Prediction":
    st.title("⌨️ Manual Profit Prediction")
    st.write("Enter feature values to get an instant profit prediction.")

    # Define input fields
    rd_spend = st.number_input("R&D Spend", value=100000)
    administration = st.number_input("Administration", value=120000)
    marketing_spend = st.number_input("Marketing Spend", value=300000)

    if st.button("Predict Profit"):
        with st.spinner("Predicting..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict_single", 
                json={"features": [rd_spend, administration, marketing_spend]}
            )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Profit: ${result['predicted_profit']:.2f}")
        else:
            st.error("Error: Unable to get a response from the API.")

# 📊 Model Information
elif option == "📊 Model Info":
    st.title("📊 Model Information")
    st.write("### ElasticNet Regression Model")
    st.write("This model was trained to predict profit based on key business features.")

    st.metric(label="✅ Model Accuracy", value="94%")
    
    st.write("### About ElasticNet")
    st.write("ElasticNet is a hybrid of Lasso and Ridge regression, balancing L1 and L2 penalties for better feature selection and regularization.")

    st.image("https://miro.medium.com/max/875/1*iKnNXAPMT0OrnnEXChaluw.png", caption="ElasticNet Regularization")
