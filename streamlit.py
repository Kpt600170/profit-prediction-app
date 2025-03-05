import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# Streamlit app title
st.title("Startup Profit Prediction 📈")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    # Display basic info
    st.write("### Data Summary")
    st.write(df.describe())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Select features and target
    features = ['R&D Spend', 'Administration', 'Marketing Spend']
    target = 'Profit'

    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ElasticNet model with best parameters
    best_model = ElasticNet(alpha=1, l1_ratio=1.0)
    best_model.fit(X_train_scaled, y_train)

    # Predict on uploaded dataset
    df['Predicted Profit'] = best_model.predict(scaler.transform(X))
    st.write("### Predicted Values for Uploaded Dataset")
    st.write(df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit', 'Predicted Profit']])

    # Line graph to show difference between actual and predicted profit
    st.write("### Actual vs Predicted Profit")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Profit'], label='Actual Profit', marker='o')
    ax.plot(df.index, df['Predicted Profit'], label='Predicted Profit', marker='x')
    ax.set_xlabel("Index")
    ax.set_ylabel("Profit")
    ax.legend()
    st.pyplot(fig)

    # Predict on new input
    st.write("### Make a Prediction")
    input_data = []
    for feature in features:
        value = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
        input_data.append(value)

    if st.button("Predict Profit"):
        input_data = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_scaled)[0]
        st.write(f"### Predicted Profit: ${prediction:,.2f}")



