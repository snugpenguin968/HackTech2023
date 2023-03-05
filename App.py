import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
st.title('Solar Energy Classifier ☀️')
st.subheader("Will someone *flip the switch* to solar energy?")
st.markdown(
         """   
         There are many factors that people consider when deciding whether or not to utilize a renewable solar energy source. \
         This model considers the following features at the STATE level:
         - Net solar energy generated (gigawatthours)
         - Average retail price of electricity in general ($)
         - Average retail price of solar-based electricity ($)
         - Average monthly temperature (°F)
         - Political affiliation - check the description of this [image](https://en.wikipedia.org/wiki/File:Red_state,_blue_state.svg)
         - Median household income ($)
         """)
st.write('Based on this data, it is possible to predict whether or not someone in a certain state in a certain month will start using solar-based energy or continue with their current non-renewable energy source.')
with st.form('input'):
    st.caption('This is data from January 2014 in Texas')
    solar_energy=st.number_input('Amount of solar energy generated/month (GWh)',min_value=0.0,value=4.93823)
    all_prices=st.number_input('Retail price of all sources of electricity ($)',min_value=0.0,value=11.22)
    solar_price=st.number_input('Retail price of solar-based electricity ($)',min_value=0.0,value=16.830)
    temp=st.number_input('Average monthly temperature (°F)',min_value=0.0,value=44.5)
    affiliation=st.selectbox('State political affiliation',('Red','Pink','Purple','Light Blue','Dark Blue'))
    income=st.number_input('Median household income ($)',min_value=0.0,value=58146.0)
    columns=st.columns((2,1,2))
    button=columns[1].form_submit_button('Submit')
st.subheader('Prediction...')
if button:
    affiliation_value=25
    if affiliation=='Pink':
        affiliation_value=50
    elif affiliation=='Purple':
        affiliation_value=75
    elif affiliation=='Light Blue':
        affiliation_value=100
    elif affiliation=='Dark Blue':
        affiliation_value=125
    xgb_model=xgb.XGBClassifier()
    xgb_model.load_model('xgb_model.txt')
    input_arr=np.array([solar_energy,all_prices,solar_price,temp,affiliation_value,income])
    y_pred=xgb_model.predict(input_arr.reshape(1,6))[0]
    
    if y_pred==1:
        st.write('Switch!')
    else:
        st.write('No Switch!')
