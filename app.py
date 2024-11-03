import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# loading the saved models

diabetes_model = pickle.load(open('diabetes_trained_model.pkl', 'rb'))

heart_disease_model = pickle.load(open('heart_trained_model.pkl', 'rb'))

parkinsons_model = pickle.load(open('parkinson_trained_model.pkl', 'rb'))


st.title("Multiple Disease Prediction using Machine Learning")
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)

def Disease(input_data):    
    input_data_as_array = np.asarray(input_data)
    reshape_data = input_data_as_array.reshape(1, -1)
    std_data = scaler.transform(reshape_data)
    pred = diabetes_model.predict(std_data)
    print(pred)
    if pred[0] == 0:
        return st.success( "The user no need to worry")
        
    else:
        return st.error("The user need to consult doctor and take precautions")

    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    # page title
    st.title('Diabetes Prediction using ML')
    dataset_new = pd.read_csv("dataset_new.csv")
    x = dataset_new.iloc[:,:-1]
    scaler = StandardScaler()
    scaler.fit(x.values)
    Glucose = st.number_input("Enter Glucose ", step = 0.1,min_value=44.0,max_value=199.0,format='%0.1f')
    Bp = st.number_input("Diastolic blood pressure (mm Hg)",step = 0.1,max_value=200.0,min_value=60.0,format='%0.1f')
    Insulin = st.number_input("2-Hour serum insulin (mu U/ml)", step = 1,min_value=14,max_value=1000)
    BMI = st.number_input("Enter Body Mass Index",min_value=18.0,max_value=52.0,format='%0.1f')
    Age = st.number_input("Enter you Age", step = 1,min_value=10,max_value=90)
    diagnosis = ''
    if st.button('PREDICT'):
        diagnosis = Disease([Glucose,Bp,Insulin,BMI,Age])



def heart_disease(input_data):
    input_data_as_array = np.asarray(input_data)
    reshape_data = input_data_as_array.reshape(1, -1)
    pred = heart_disease_model.predict(reshape_data)
    if pred[0] == 0:
        return st.success( "The user no need to worry")
        
    else:
        return st.error("The user need to consult doctor and take precautions")

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    # page title
    st.title('Heart Disease Prediction using ML')
    age =  st.number_input("Patient’s age ", step = 2,min_value=29,max_value=77)
    genders = ['1.male','0.female']
    x = st.radio("Gender",genders)
    if(x=='1.male'):
        sex=1
    else:
        sex=0
    option = st.selectbox(
    'Chest pain type ',
    ('0.typical angina', '1.atypical angina', '2.non-anginal pain','3.asymptomatic'))
    if(option=='1.typical angina'):
        cp = 0
    elif option == '2.atypical angina':
        cp = 1
    elif option == '3.non-anginal pain':
        cp = 2
    else:
        cp = 3
    trestbps =  st.number_input("Enter resting blood pressure ", min_value=94, max_value=200, step = 2)
    chol =  st.number_input("Serum cholesterol in mg/dl", step = 1,min_value=126,max_value=564)
    # Fasting blood sugar>120 mg/dl, true- 1 false-0
    fbs_values = ['1.true','0.false']
    y = st.radio("Fasting blood sugar>120 mg/dl(true- 1 false-0) ",fbs_values)
    if(x=='1.true'):
        fbs=1
    else:
        fbs=0
    #restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
    option2 = st.selectbox(
    'Resting electorcardiographic measurement ',
    ('0.normal', '1.ST_T wave abnormality', '2.left entricular hypertrophy'))
    if(option2=='0.normal'):
        restecg = 0
    elif option2 == '1.ST_T wave abnormality':
        restecg = 1
    else:
        restecg = 2
    thalach =  st.number_input("Maximum heart rate achieved", step = 2,min_value=71,max_value=202)
    z = st.radio("Exercise induced agina",['1.Yes','0.No'])
    if(z=='1.Yes'):
        exang=1
    else:
        exang=0
    oldpeak =  st.number_input("ST depression introduced by exercise relative to rest", step = 0.01,min_value=0.0,max_value=6.2,format='%0.1f')
    # slope of st_segment
    # 0- down sloping
    # 1 - flat
    # 2 - upsloping
    option3 = st.selectbox(
    'Slope of peak exercise St_Segment ',
    ('0.Down sloping', '1.flat', '2.upsloping'))
    if option3 == '0.Down sloping':
        slope = 0
    elif option3 == '1.flat':
        slope = 1
    else:
        slope = 2
    # slope =  st.number_input("Enter slope ", step = 2)
    ca =  st.number_input("Number of major vessels ", step = 1,min_value=0,max_value=4)
    diagnosis = ''
    if st.button('PREDICT'):
        diagnosis = heart_disease([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca])
    
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    fo = st.number_input('MDVP Fo(Hz)',format='%0.1f',min_value=88.0,max_value=260.0)
    
    fhi = st.number_input('MDVP Fhi(Hz)',format='%0.1f',min_value=102.0,max_value=592.0)

    flo = st.number_input('MDVP Flo(Hz)',format='%0.1f',min_value=65.0,max_value=239.0)
        
    NHR = st.number_input('NHR',format='%0.6f',min_value=0.000650,max_value=0.314820)        
    
    HNR = st.number_input('HNR',format='%0.6f',min_value=8.441000,max_value=33.047000)
        
    DFA = st.number_input('DFA',format='%0.6f',min_value=0.574282,max_value=0.825288)
        
    spread = st.number_input('spread2',format='%0.6f',min_value=0.006274,max_value=0.450493)
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo,NHR,HNR,DFA,spread]])                          
        
        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The User need to consult a doctor and take necessary precautions"
            st.error(parkinsons_diagnosis)
        else:
          parkinsons_diagnosis = "The User no need to worry"
          st.success(parkinsons_diagnosis)
        
    
