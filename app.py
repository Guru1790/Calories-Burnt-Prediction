import sklearn
print(sklearn.__version__)

import pickle
from sklearn.ensemble import RandomForestRegressor

# Assuming X_train and Y_train are already defined
model = RandomForestRegressor()
model.fit(X_train, Y_train)

# Save the model
with open('calorie_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model
with open('calorie_model.pkl', 'rb') as file:
    model = pickle.load(file)


import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open('calorie_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit UI
st.title('Calories Burned Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=25)
height = st.number_input('Height (in cm)', min_value=0.0, value=170.0)
weight = st.number_input('Weight (in kg)', min_value=0.0, value=70.0)
duration = st.number_input('Duration of Activity (in minutes)', min_value=0.0, value=30.0)
gender = st.selectbox('Gender', ['Male', 'Female'])

# Preprocess inputs
gender_encoded = 0 if gender == 'Male' else 1
inputs = np.array([[age, height, weight, duration, gender_encoded]])
scaled_inputs = scaler.transform(inputs)

# Prediction
if st.button('Predict Calories Burned'):
    prediction = model.predict(scaled_inputs)
    st.write(f'Estimated Calories Burned: {prediction[0]:.2f}')
