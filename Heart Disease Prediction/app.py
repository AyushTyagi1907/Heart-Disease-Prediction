import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st
from PIL import Image

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://victoria.mediaplanet.com/app/uploads/sites/45/2020/11/GettyImages-1205212219.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



pickle_in = open('m.pkl', 'rb')
classifier = pickle.load(pickle_in)
st.title("Welcome to Heart Disease Prediction")
def prediction(input_data_reshaped):
    res=classifier.predict(input_data_reshaped)
    return res
def main():
    v1 = st.number_input('Age', min_value=0, max_value=1000,step=1) 
    v2 = st.number_input('Sex', min_value=0, max_value=1000,step=1)
    v3 = st.number_input('Cp', min_value=0, max_value=1000,step=1)
    v4 = st.number_input('Trestbps', min_value=0, max_value=1000,step=1)
    v5 = st.number_input('Cholesterol', min_value=0, max_value=1000,step=1)
    v6 = st.number_input('Fbs', min_value=0, max_value=1000,step=1)
    v7= st.number_input('ECG', min_value=0, max_value=1000,step=1)
    v8= st.number_input('Thalach', min_value=0, max_value=1000,step=1)
    v9= st.number_input('Exang', min_value=0, max_value=1000,step=1)
    v10= st.number_input('Oldpeak', min_value=0.0, max_value=1000.0,step=0.1)
    v11= st.number_input('Slope', min_value=0, max_value=1000,step=0)
    v12= st.number_input('Coronary Calcium', min_value=0, max_value=1000,step=1)
    v13= st.number_input('Thal', min_value=0, max_value=1000,step=1)
    input_data = (v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    if st.button("Predict"):
        result = prediction(input_data_reshaped)
        if(result==0):
            st.success("No,chance of Heart Disease")
        else:
            st.success("Yes,there are chance of Heart Disease")    
if __name__=='__main__':
    main()        