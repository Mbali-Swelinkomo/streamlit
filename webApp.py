import streamlit as st
import joblib
import pandas as pd

# Load the saved machine learning model
model = joblib.load('heart_disease_model.pkl')


# Function to make predictions
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    prediction = model.predict(input_data)
    return prediction[0]


# Streamlit UI
st.title('Heart Disease Prediction')

age = st.number_input('Age', min_value=0, max_value=150)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['0', '1', '2', '3'])
trestbps = st.number_input('Resting Blood Pressure', min_value=0)
chol = st.number_input('Cholesterol Level', min_value=0)
fbs = st.selectbox('Fasting Blood Sugar', ['0', '1'])
restecg = st.selectbox('Resting Electrocardiographic Results', ['0', '1', '2'])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)
exang = st.selectbox('Exercise Induced Angina', ['0', '1'])
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['0', '1', '2'])
ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0)
thal = st.selectbox('Thalassemia', ['0', '1', '2', '3'])

if st.button('Predict'):
    # Convert sex and other categorical variables to numerical values
    sex = 1 if sex == 'Male' else 0
    cp = int(cp)
    fbs = int(fbs)
    restecg = int(restecg)
    exang = int(exang)
    slope = int(slope)
    thal = int(thal)

    # Make prediction
    result = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    # Display result
    if result == 1:
        st.write('The patient is likely to have heart disease.')
    else:
        st.write('The patient is unlikely to have heart disease.')
