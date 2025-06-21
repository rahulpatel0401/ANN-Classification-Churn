import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st

#load the trained model
model = load_model('model.keras')

##Load the encoder scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('sscaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

##Streamlit
st.title('Customer Chrun Prediction')

##User Input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cc_card = st.selectbox('Has Credit Card', [0, 4])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#Prepare the input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cc_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}
input_df = pd.DataFrame(input_data)

#Encode geography column
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

#Merge encoded Geography column
input_data = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

#Print the prediction
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')


