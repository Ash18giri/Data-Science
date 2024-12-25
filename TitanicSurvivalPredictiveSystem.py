# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model=pickle.load(open('C:/Users/admin/Downloads/DA assignments zip file/Logistic Regression/Titanic_model.sav','rb'))
input_data=(3,22.0,1,0,7.2500,1,0,0,1)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
  print('Did not survive')
else:
  print('Survived')