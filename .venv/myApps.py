import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat


data, meta = pyreadstat.read_sav('Dataset Final.sav')

# EDA
st.title("Obesity Prediction App")
st.write("## Exploratory Data Analysis")

st.write("### Dataset Overview")
st.write(data.head())

st.write("### Dataset Description")
st.write(data.describe())



# Check for null values in the dataset
if data.isnull().sum().sum() > 0:
    st.warning("The dataset contains null values. These will be dropped.")
    data = data.dropna()

# Define features and target
X = data.drop(columns=['Y'])
y = data['Y']

# Drop columns that are not needed
columns_to_drop = ['Pulau', 'B1R1', 'weight_final', 'filter_$']
data = data.drop(columns=columns_to_drop, errors='ignore')
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {accuracy:.2f}")

# User input
st.write("## Input Features")
input_data = []
for i in range(17):
    input_data.append(st.number_input(f"Feature {chr(65+i)}", min_value=0.0, max_value=100.0, step=0.1))

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=[f"Feature {chr(65+i)}" for i in range(17)])
    prediction = model.predict(input_df)
    st.write(f"### Prediction: {'Obese' if prediction[0] == 1 else 'Not Obese'}")