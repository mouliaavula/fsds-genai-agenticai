import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("My First StreamLit App")

# Description
st.write("This is a simple app to demonstrate the basic functionalities of streamlit")

#  Interactive widgets in sidebar.....
st.sidebar.header("User Input Features")

# text input
user_name = st.sidebar.text_input("What is your name","Mouli Aavula")

# slider
age = st.sidebar.slider("Select your Age",0,200,25)

# Select box
favorite_color = st.sidebar.selectbox("What is your favorite color",["Blue","Red","Green","Yellow"])

# ---Main page content---
st.header(f"Welcome {user_name}!")
st.write(f"Your age is {age} old and your favorite color is {favorite_color}.")

# --Display data
st.subheader("Here is some random data")
columns=('col %d' % i for i in range(5))
# Create a sample dataframe
data = pd.DataFrame(
        np.random.randn(10,5),
        columns=('col %d' % i for i in range(5)))
st.dataframe(data)

# Check box to show and hide content
if st.checkbox("Show ra data"):
    st.subheader("Raw Data")
    st.write(data)

# --- Button to trigger an action
if st.button("Say hello"):
    st.write("Hello there!")
else:
    st.write("Goodbye")
    
# use https://github.com/streamlit/streamlit for more details in streamlit
