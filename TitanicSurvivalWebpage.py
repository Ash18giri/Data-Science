# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:58:50 2024

@author: admin
"""

import numpy as np
import pickle
import streamlit as st

#Loading the saved model
loaded_model=pickle.load(open('C:/Users/admin/Downloads/DA assignments zip file/Logistic Regression/Titanic_model.sav','rb'))


st.title('Titanic Survival Prediction')
Pclass=st.text_input('Enter the Passenger Class')
Age=st.text_input('Enter the Passenger Age')
SibSp=st.text_input('Enter the Passenger SibSp')
Parch=st.text_input('Enter the Passenger Parch')
Fare=st.text_input('Enter the Passenger Fare')
Sex_male=st.text_input('Enter the Passenger sex as 1 if male otherwise 0')
Embarked_C=st.text_input('Enter the Passenger Embarked_C if yes as 1')
Embarked_Q=st.text_input('Enter the Passenger Embarked_Q if yes as 1')
Embarked_S=st.text_input('Enter the Passenger Embarked_S if yes as 1')
    
diagnosis=''
    #creating a button
if st.button('Survival Test Result'):
    input_data=([[Pclass,Age,SibSp,Parch,Fare,Sex_male,Embarked_C,Embarked_Q,Embarked_S]])
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    if(prediction[0]==0):
          diagnosis='Did not survive'
    else:
          diagnosis='Survived'
      
st.success(diagnosis)
    
 
    