# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:58:50 2024

@author: admin
"""

import numpy as np
import pickle
import streamlit as st

#Loading the saved model
load=open('classifier.pkl','rb')
model=pickle.load(load)

def predict(Pclass,Age,SibSp,Parch,Fare,Sex_male,Embarked_C,Embarked_Q,Embarked_S):
    prediction=model.predict([[Pclass,Age,SibSp,Parch,Fare,Sex_male,Embarked_C,Embarked_Q,Embarked_S]])
    return prediction

def main():
    st.title('Titanic Survival Prediction')
    Pclass=st.number_input('Enter the Passenger Class')
    Age=st.number_input('Enter the Passenger Age')
    SibSp=st.number_input('Enter the Passenger SibSp')
    Parch=st.number_input('Enter the Passenger Parch')
    Fare=st.number_input('Enter the Passenger Fare')
    Sex_male=st.number_input('Enter the Passenger sex as 1 if male otherwise 0')
    Embarked_C=st.number_input('Enter the Passenger Embarked_C if yes as 1')
    Embarked_Q=st.number_input('Enter the Passenger Embarked_Q if yes as 1')
    Embarked_S=st.number_input('Enter the Passenger Embarked_S if yes as 1')
    
    
    if st.button('Predict'):
       result=predict(Pclass,Age,SibSp,Parch,Fare,Sex_male,Embarked_C,Embarked_Q,Embarked_S)
       if result==0:
          st.success('Did not survive')
       else:
          st.success('Survived')
          
          
if __name__=='__main__':
    main()

    
 
    