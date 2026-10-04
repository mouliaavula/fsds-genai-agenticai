import streamlit as st
import numpy as np
import pickle
import os 

# Load the saved model

model = pickle.load(open(r"E:\vs-code-prakash\fsds-genai-agenticai-ws\9.ML\1.Regression\linear_regression_model.pkl",'rb'))

# Set the title of the streamlit app 
st.title("Salary Prediction Application")

st.write("This application predicts salary based on years experience using simple linear regression")

# Add input widget for user to enter years of experience
years_of_exp = st.number_input("Enter Years of Experience: ",min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# when button is click make the predictions 

if st.button("Predict Salary"):
    #Convert the input into 2D array for prediction
    exp_input = np.array([[years_of_exp]])
    # make the prediction using trained model
    prediction = model.predict(exp_input)
    # Display the result 
    st.success(f"The predicted salary for {years_of_exp}  years of experience is: ${prediction[0]:,.2f}")
    
st.write("The model was trained using a dataset of salaries and years of experience")    

