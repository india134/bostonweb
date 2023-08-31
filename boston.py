import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import shap
df= pd.read_csv("C:/Users/tutel/Downloads/boston-housing-dataset.csv")
st.write("Boston House Prediction")
Y= df['MEDV']
X= df.drop('MEDV',axis=1)
st.write(X)
st.sidebar.subheader('scatterplot settings')
xvalues= st.sidebar.selectbox('X axis',options= df.columns)
yvalues=st.sidebar.selectbox('Y axis',options= df.columns)
plot=plt.scatter(data=df,x=xvalues,y=yvalues)
st.write(plot)
st.sidebar.header('input your paramters')
def user_input():
    st.sidebar.slider('CRIM',float(X.CRIM.min()),float(X.CRIM.max()),float(X.CRIM.mean()))
    st.sidebar.slider('ZN',float(X.ZN.min()),float(X.ZN.max()),float(X.ZN.mean()))
    st.sidebar.slider('CHAS',float(X.CHAS.min()),float(X.CHAS.max()),float(X.CHAS.mean()))
    st.sidebar.slider('INDUS',float(X.INDUS.min()),float(X.INDUS.max()),float(X.INDUS.mean()))
    st.sidebar.slider('NOX',float(X.NOX.min()),float(X.NOX.max()),float(X.NOX.mean()))
    st.sidebar.slider('RM',float(X.RM.min()),float(X.RM.max()),float(X.RM.mean()))
    st.sidebar.slider('AGE',float(X.AGE.min()),float(X.AGE.max()),float(X.AGE.mean()))
    st.sidebar.slider('DIS',float(X.DIS.min()),float(X.DIS.max()),float(X.DIS.mean()))
    st.sidebar.slider('RAD',float(X.RAD.min()),float(X.RAD.max()),float(X.RAD.mean()))
    st.sidebar.slider('TAX',float(X.TAX.min()),float(X.TAX.max()),float(X.TAX.mean()))
    st.sidebar.slider('PTRATIO',float(X.PTRATIO.min()),float(X.PTRATIO.max()),float(X.PTRATIO.mean()))
    st.sidebar.slider('B',float(X.B.min()),float(X.B.max()),float(X.B.mean()))
    st.sidebar.slider('LSTAT',float(X.LSTAT.min()),float(X.LSTAT.max()),float(X.LSTAT.mean()))
    

df1=user_input()
model= RandomForestRegressor()
model.fit(xtrain,ytrain)

prediction= model.predict(df1)
st.header('Prediction of house price')
st.write(prediction)
