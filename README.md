# Heart Disease Prediction Web Application

This project aims to develop a web application for predicting whether a patient is likely to suffer from heart disease based on their medical data. 
The application utilises machine learning models trained on a heart disease dataset to provide predictions.

### Features

    Prediction Form: Allows doctors to enter the details of patients, including age, sex, chest pain type, resting blood pressure, serum cholesterol level, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina, ST depression induced by exercise relative to rest, slope of the peak exercise ST segment, number of major vessels (0-3) colored by fluoroscopy, and thallium heart scan results.
    Prediction Results: Displays the prediction result indicating whether the patient is likely to suffer from heart disease or not.
    Model Selection: Supports multiple machine learning models for prediction.
    Error Handling: Provides informative messages for invalid inputs and errors.

### Usage

To run the application locally, follow these steps:

    Install the required dependencies by running pip3 install -r requirements.txt.
    Navigate to the project directory.
    Run the Streamlit app by navigating to the question4 directory and execute streamlit run webApp.py in your terminal.
    Access the application in your web browser at http://localhost:8501.

## Project Structure

The project directory structure is as follows:

    webApp.py: The main Streamlit application script.
    heart_disease_model.pkl: The trained machine learning model for heart disease prediction.
    requirements.txt: The list of Python dependencies required to run the application.
    heart.csv: Directory containing the heart disease dataset.
    mydatabase.db: database where the patient  records are stored.
    README.md: This README file.

## Technologies Used

    Python
    Streamlit
    Pandas
    Scikit-learn

## Contributors

Mbali Swelinkomo

