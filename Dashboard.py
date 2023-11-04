import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os

st.set_page_config(layout='wide', page_title='Dashboard', page_icon=':sparkles:')
st.title ("Analytics Dashboard")



os.chdir(r'C:\Users\purva\Downloads\Dashboard\Data')

options = os.listdir()

selData = st.selectbox(label='Select dataset',options=options)

df = pd.read_csv(selData)

def categorical_variable(df):
    return list(df.select_dtypes(include = ['category', 'object']))

def numerical_variable(df):
    return list(df.select_dtypes(exclude = ['category', 'object']))

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Records','Description','Shape','Information','Profile'])

with tab1:
    
    col1, col2 = st.columns([3,1])
    
    with col1:
        records = st.slider('Number of records',1,20)

    with col2:    
        ht = st.radio("First or Last records",("First", "Last"))
    
    st.header(ht + ' ' + str(records) + ' records')
    if ht == "First":
        st.write(df.head(records))
    else:
        st.write(df.tail(records))
        
with tab2:
    st.header('Description of the dataset')
    st.write(df.describe())
    
with tab3:
    st.header('Number of rows and columns in the dataset')
    st.write(df.shape)
    a = categorical_variable(df)
    st.subheader("Categorical Variables")
    st.write(a)
    a = numerical_variable(df)
    st.subheader("Numerical Variables")
    st.write(a)
    
    
with tab4:
    st.header('Dataset Information')
    st.write(df.dtypes)

with tab5:
    st.header("Dataset Profile Report")
    if st.button("Show Profile Report"):
        report = ProfileReport(df)
        st_profile_report(report)
    
    
    