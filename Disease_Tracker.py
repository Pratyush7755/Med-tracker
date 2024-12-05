
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu


# loading the saved models

diabetes_model = pickle.load(open('trained_model_diabetes.sav', 'rb'))

heart_disease_model = pickle.load(open('trained_model_heart-disease.sav', 'rb'))

parkinsons_model = pickle.load(open('trained_model_parkinson.sav', 'rb'))

hepatitis_model = pickle.load(open('trained_model_hepatitis.sav', 'rb'))

# sidebar for navigation

with st.sidebar:
    
    st.header('Disease Prediction using Machine learning')
    
    selected = option_menu('Personal MedTracker',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                          'Hepatitis Prediction'],
                          icons=['activity','heart','person','align-middle'],
                          default_index=0)
 
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction ')

    # Add input fields for user to input data
    Pregnancies = st.slider('Pregnancies', 0, 25, 3)
    Glucose = st.slider('Glucose', 0, 500, 117)
    BloodPressure = st.slider('BloodPressure', 0, 300, 72)
    SkinThickness = st.slider('SkinThickness', 0, 200, 23)
    Insulin = st.slider('Insulin', 0, 2000, 30)
    BMI = st.slider('BMI', 0.0, 100.0, 32.0)
    DiabetesPedigreeFunction = st.slider('DiabetesPedigreeFunction', 0.078, 10.0, 0.3725)
    Age = st.slider('Age', 0, 200, 21)
    
    
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    # Make prediction
    prediction = diabetes_model.predict(input_data)
    # Predict button
    if st.button('Predict'):
        # Call the predict_diabetes function to make prediction
        if prediction == 1:
            st.success('Prediction: Diabetes')
        else:
            st.success('Prediction: No Diabetes')


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction ')
    
    co1,co2 = st.columns(2)
     
    with co1:
        age = st.text_input('Age')
        
    with co2:
        sex = st.radio('Sex', ['Male', 'Female','Other'])
        if sex == 'Male':
            sex=1
        elif sex == 'Female':
            sex=0
        else :
            sex=2
            
    c1,c2=st.columns(2)
    
    with c1:
        cp = st.slider('Chest Pain types', 0, 3, 0)
     
    with c2:
        fbs = st.slider('Fasting Blood Sugar > 120 mg/dl', 0, 1, 0)
    
    with c1:
        restecg = st.slider('Resting Electrocardiographic results', 0, 2, 0)
        
    with c2:
        exang = st.slider('Exercise Induced Angina', 0, 1, 0)
        
    with c1:
        oldpeak = st.slider('ST depression induced by exercise', 0.00, 10.00, 1.0)
        
    with c2:
        slope = st.slider('Slope of the peak exercise ST segment', 0, 2, 0)
    with c1:
        ca = st.slider('Major vessels colored by flourosopy', 0, 5, 0)
    
              
        
    col1, col2 = st.columns(2)   
    
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')     
  
    with col1:
        thalach = st.text_input('Maximum Heart Rate achieved')
             
    with col2:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
    
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction")
    
    st.write('Please Enter numerical values to the following field :')
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP_Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP_Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP_Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP_Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP_Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP_RAP')
        
    with col2:
        PPQ = st.text_input('MDVP_PPQ')
        
    with col3:
        DDP = st.text_input('Jitter_DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP_Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP_Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer_APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer_APQ5')
        
    with col3:
        APQ = st.text_input('MDVP_APQ')
        
    with col4:
        DDA = st.text_input('Shimmer_DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)


# Hepatitis Prediction Page
if (selected == "Hepatitis Prediction"):

    # Define the input fields
    st.title('Hepatitis Prediction')
    
    st.write('Please input the following features to predict hepatitis:')
    
    age = st.slider('Age', min_value=0, max_value=100, step=1)
    
    col1, col2 = st.columns(2)
        
    with col1:    
        sex = st.radio('Sex', ['Male', 'Female'])
    
    with col2:
        steroid = st.radio('Steroid', ['Yes', 'No'])
    
    with col1:
        antivirals = st.radio('Antivirals', ['Yes', 'No'])
    
    with col2:
        fatigue = st.radio('Fatigue', ['Yes', 'No'])
    
    with col1:
        spiders = st.radio('Spiders', ['Yes', 'No'])
    
    with col2:
        ascites = st.radio('Ascites', ['Yes', 'No'])
    
    with col1:
        varices = st.radio('Varices', ['Yes', 'No'])
    
    bilirubin = st.slider('Bilirubin', min_value=0.0, max_value=10.0, step=0.1)
    alk_phosphate = st.slider('Alkaline Phosphate', min_value=0, max_value=500, step=1)
    sgot = st.slider('SGOT', min_value=0, max_value=1000, step=1)
    albumin = st.slider('Albumin', min_value=0.0, max_value=10.0, step=0.1)
    protime = st.slider('Protime', min_value=0, max_value=200, step=1)
    histology = st.radio('Histology', ['Yes', 'No'])

    # Convert categorical inputs to numerical
    sex = 1 if sex == 'Male' else 0
    steroid = 1 if steroid == 'Yes' else 0
    antivirals = 1 if antivirals == 'Yes' else 0
    fatigue = 1 if fatigue == 'Yes' else 0
    spiders = 1 if spiders == 'Yes' else 0
    ascites = 1 if ascites == 'Yes' else 0
    varices = 1 if varices == 'Yes' else 0
    histology = 1 if histology == 'Yes' else 0

    # Predict hepatitis
    input_data = np.array([[age, sex, steroid, antivirals, fatigue,spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin, protime, histology]])
    prediction = hepatitis_model.predict(input_data)

    # Display prediction
    if st.button('Predict'):
        # Call the predict_diabetes function to make prediction
        if prediction == 1:
            st.success('Prediction: The patient is predicted to have hepatitis')
        else:
            st.success('Prediction: The patient is predicted to not have hepatitis')
    
    
    
