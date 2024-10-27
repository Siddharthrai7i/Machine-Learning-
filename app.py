from PIL import Image
import streamlit as st
import pickle
import numpy as np
import pandas as pd 
st.header("Machine Learning  MODEL ")
st.image("diabtrean.jpg")
with open('classifier.pkl', 'rb') as file:
   modal=pickle.load(file)

def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = modal.predict(input_data)
    return prediction

st.title("You enter your you details")
st.info(" Medical Information Details of a Person")
st.warning("Please insure that your data is valid")
Pregnancies= st.number_input("Input for The number of Pregnancies" , placeholder="enter")
Glucose= st.number_input("Input for The Gloucse Level of your body ")
BloodPressure= st.number_input("Input for BLOOD PRESSURE level for your body")
skinThickness= st.number_input("Input for Skin Thickness ")
Insulene= st.number_input("Input for Insulene level of your body")
BMI= st.number_input("Input for BMI level of your body ")
DiabetesPedigreeFunction= st.number_input("Input for The DiabetesPedigreeFunction ")
Age= st.number_input("Input for The AGE of the person ")

if st.button("Predict"):
    # Make prediction based on user input
    prediction = predict([Pregnancies,Glucose , BloodPressure,skinThickness ,Insulene ,BMI ,DiabetesPedigreeFunction ,Age ])  # Add more inputs if needed
    st.write(f"Prediction: {prediction}")
    if (prediction==1):
        st.write("The person has 'DIABETEAS' ")
        st.warning("Please take Care of your health ")
    else :
        st.write("The person is safe ")


