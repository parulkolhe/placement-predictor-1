import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv('modified_placement_data.csv')

# Replace comma with dot and convert to float
df['cgpa'] = df['cgpa'].str.replace(',', '.', regex=False).astype(float)

# Convert placement column to numeric
df['placement_numeric'] = df['placement'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop any rows with missing values to avoid training errors
df.dropna(inplace=True)

# Features and labels
X = df[['cgpa', 'iq']]
y = df['placement_numeric']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# Model accuracy
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Placement Predictor")
st.markdown("Enter your **CGPA** and **IQ** to predict your placement chances.")

# Display model accuracy
st.markdown(f"**Model Accuracy:** `{accuracy * 100:.2f}%`")

# Input fields
cgpa = st.number_input("Enter CGPA ", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ ", min_value=50, max_value=200, step=1)

if st.button("Predict Placement"):
    input_data = np.array([[cgpa, iq]])
    input_scaled = scaler.transform(input_data)
    prediction = clf.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Likely to be Placed!")
    else:
        st.error("Not Likely to be Placed.")

# Save model (optional)
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
