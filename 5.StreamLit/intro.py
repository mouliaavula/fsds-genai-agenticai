# 1. import streamlit
import streamlit as st

# 2. Add some title to your app
st.title("My app created by Mouli Aavula")

# 3. Add some text
st.write("Welcome this app calculate square of a number")

# 4 .Create an interactive slider
st.header("Select a Number")
number = st.slider("Pick a number",0,100,25) # min max default

# 5. Calculate and display the result
st.subheader("Result")
squared_number = number * number
st.write(f"The square of the **{number}** is **{squared_number}**.")

# 6. Command to run this *.py is go to file folder and do streamlit run *.py
