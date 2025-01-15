import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to train the model
def train_model(train_data):
    y = train_data['cnt']
    X = train_data.drop(['cnt', 'atemp', 'registered'], axis=1)

    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r_squared = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return model, r_squared, rmse

# Function to test the model
def test_model(model, test_data):
    y_test = test_data['cnt']
    X_test = test_data.drop(['cnt', 'atemp', 'registered'], axis=1)

    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return rmse_test

# Streamlit app
st.title("Bike Sharing Prediction Model")

# File upload for training data
st.header("Upload Training Data")
train_file = st.file_uploader("Choose a CSV file for training", type="csv")

if train_file is not None:
    train_data = pd.read_csv(train_file)
    st.write("Training Data Preview:")
    st.write(train_data.head())

    # Train the model
    model, train_r2, train_rmse = train_model(train_data)

    st.success(f"Model trained successfully!")
    st.write(f"Training R-squared: {train_r2:.4f}")
    st.write(f"Training RMSE: {train_rmse:.4f}")

# File upload for test data
st.header("Upload Test Data")
test_file = st.file_uploader("Choose a CSV file for testing", type="csv")

if test_file is not None and train_file is not None:
    test_data = pd.read_csv(test_file)
    st.write("Test Data Preview:")
    st.write(test_data.head())

    # Test the model
    test_rmse = test_model(model, test_data)

    st.write(f"Test RMSE: {test_rmse:.4f}")

elif test_file is not None and train_file is None:
    st.error("Please upload training data before uploading test data.")