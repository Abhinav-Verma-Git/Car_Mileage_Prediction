import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

data=pd.read_csv("mpg.csv")
model=joblib.load("model.pkl")

menu=st.sidebar.radio("Menu",["Home","Analysis","MPG Prediction"])
if menu=="Home":
    st.title(" MPG Prediction") 
    st.image("mpg.png",width=500)
    st.markdown("This project builds and evaluates a machine learning model to predict a car's Miles Per Gallon (MPG) using key features like Horsepower and Weight. The model is trained using XGBoost Regressor with hyperparameter tuning via GridSearchCV.")
    st.divider()
if menu=="Analysis":
    st.title("Analytical  Data")
    st.image("analysis.png",width=500)
    if st.checkbox("Tabular Data"):
        st.table(data.head(50))
        st.text("Note: Model is trained on:")
        st.text(data.shape)
        st.divider()
    if st.checkbox("Statistics"):
        st.table(data.describe())
        st.divider()
    if st.checkbox("Correlation Heatmap"):
        fig=plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
        st.divider()
    if st.checkbox("Scatterplot"):
       value=st.slider("Filter the data using Horsepower",46,230)
       data=data.loc[data["Horsepower"]>=value]
       fig,ax=plt.subplots(figsize=(10,6))
       sns.scatterplot(data=data,x="Horsepower",y="MPG",hue="Weight")
       st.pyplot(fig)
       st.divider()
    if st.checkbox("Lineplot"):
        value=st.slider("Filter using the data using Horsepower",46,230)
        data=data.loc[data["Horsepower"]>=value]
        fig,ax=plt.subplots(figsize=(10,6))   
        sns.lineplot(x="Horsepower",y="MPG",data=data,estimator="mean",errorbar=None)
        st.pyplot(fig)
        st.divider()

if menu=="MPG Prediction":
    st.title("MPG Prediction of the Car") 
    st.image("analysis.png",width=500)
    features=[["Horsepower","Weight"]]
    value1=st.number_input("Horsepower")
    value2=st.number_input("Weight")
    if st.button("Predict the MPG"):
        with st.spinner("Predicting"):
            time.sleep(5)
            st.balloons()
        prediction=model.predict([[value1,value2]])[0]
        st.write(f"Price Prediction is {prediction:,.2f}")
