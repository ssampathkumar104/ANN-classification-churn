import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load model and encoders
model = tf.keras.models.load_model('model.h5', compile=False)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 30)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=600)
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

if st.button('Predict Churn'):
    try:
        # Preprocess input
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        geo_df = pd.DataFrame(
            onehot_encoder_geography.transform([[geography]]).toarray(),
            columns=onehot_encoder_geography.get_feature_names_out(['Geography'])
        )

        input_data = pd.concat([input_data, geo_df], axis=1)
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        prediction_prob = prediction[0][0]

        st.subheader(f'Churn Probability: {prediction_prob:.2f}')
        if prediction_prob > 0.5:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')
    except Exception as e:
        st.error(f"An error occurred: {e}")
