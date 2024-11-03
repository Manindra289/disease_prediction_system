import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
st.title("This is title")
loaded_model = pickle.load(open('heart_trained_model_2.pkl','rb'))

def heart_disease(input_data):
    input_data_as_array = np.asarray(input_data)
    reshape_data = input_data_as_array.reshape(1, -1)
    pred = loaded_model.predict(reshape_data)
    if pred[0] == 0:
        return st.success( "The person don't get heart disease")
        
    else:
        return st.error("The person will get heart disease")


def main():
    st.write("Heart disease prediction system")
    age =  st.number_input("Enter age ", step = 2,min_value=10,max_value=100)
    genders = ['1.male','0.female']
    x = st.radio("Gender",genders)
    if(x=='1.male'):
        sex=1
    else:
        sex=0
    option = st.selectbox(
    'Select chest pain type ',
    ('0.typical angina', '1.atypical angina', '2.non-anginal pain','3.asymptomatic'))
    if(option=='1.typical angina'):
        cp = 0
    elif option == '2.atypical angina':
        cp = 1
    elif option == '3.non-anginal pain':
        cp = 2
    else:
        cp = 3
    trestbps =  st.number_input("Enter trestbps ", min_value=94, max_value=200, step = 2)
    chol =  st.number_input("Enter chol ", step = 2)
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
    thalach =  st.number_input("Enter thalach ", step = 2,min_value=71,max_value=202)
    z = st.radio("Exercise induced angina",['1.Yes','0.No'])
    if(z=='1.Yes'):
        exang=1
    else:
        exang=0
    oldpeak =  st.number_input("Enter oldpeak ", step = 0.01,min_value=0.0,max_value=6.2,format='%0.1f')
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
    ca =  st.number_input("Enter ca ", step = 1,min_value=0,max_value=4)
    diagnosis = ''
    if st.button('PREDICT'):
        diagnosis = heart_disease([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca])


if __name__ == '__main__':
    main()