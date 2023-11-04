
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import streamlit as st
import plotly.express as px

st.title("Multiple Linear Regression")
#st.set_page_config(layout='wide',page_title='MLR')

try:
    os.chdir(r'C:\Users\purva\Downloads\Dashboard\MLR')
except FileNotFoundError:
    print("")


    

#st.write(os.getcwd())

#os.chdir('MLR')
        
options = os.listdir()

flSel = st.selectbox(label='Data Files',options=options)

df = pd.read_csv(flSel)

if flSel == 'insurance.csv':
    target = df.loc[(df.index), 'charges']
    features = df.loc[(df.index), ['age','gender','bmi','children','smoker','region']] 
elif flSel == 'LifeExpectancy.csv':
    target = df.loc[(df.index), 'Life expectancy ']
    features = df.loc[(df.index), ['Adult Mortality','infant deaths','Alcohol','percentage expenditure','Hepatitis B','Measles ',' BMI ','under-five deaths ','Polio','Total expenditure','Diphtheria ',' HIV/AIDS','GDP','Population',' thinness  1-19 years',' thinness 5-9 years','Income composition of resources','Schooling']]

else: 
    target = df.loc[(df.index), 'Y house price of unit area']
    features = df.loc[(df.index), ['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']]

df2 = df.copy()
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Features','EDA', 'Label Encoding','Correlation and ANOVA','Model'])

with tab1:
    st.header("Features and Target variables")
    col1, col2 = st.columns([3,1])
    
    with col1:
        st.write("Features")
        st.write(features)
    with col2:
        st.write("Target")
        st.write(target)
    
with tab2:
    st.header('Exploratory Description')
    g = sns.catplot(x="smoker", y="charges",col_wrap=3, col="gender",data= df, kind="box",height=5, aspect=0.8);
    st.pyplot(g)
    g = sns.catplot(x="smoker", y="age",col_wrap=3, col="gender",data= df, kind="box",height=5, aspect=0.8);
    st.pyplot(g)

with tab3:
    st.header('Label Encoding')
    if st.button('Encode Labels'):
#        df2 = df.copy()
        le = LabelEncoder()
        le.fit(df2.gender.drop_duplicates())
        df2.gender = le.transform(df2.gender)
        le.fit(df2.smoker.drop_duplicates()) 
        df2.smoker = le.transform(df2.smoker)
        df2 = pd.get_dummies(df2, columns = ['region'], drop_first=True)
#        le.fit(df2.region.drop_duplicates()) 
#        df2.region = le.transform(df2.region)
        st.write(df2.head(10))

with tab4:
    st.header('Correlation and ANOVA')
    if st.button('Correlation'):
        le = LabelEncoder()
        le.fit(df2.gender.drop_duplicates())
        df2.gender = le.transform(df2.gender)
        le.fit(df2.smoker.drop_duplicates()) 
        df2.smoker = le.transform(df2.smoker)
        df2 = pd.get_dummies(df2, columns = ['region'], drop_first=True)
        st.write(df2.corr()['charges'])
        
with tab5:   
    st.header('Linear Regression Model')
    if st.button('Fit Model'):
        le = LabelEncoder()
        le.fit(df2.gender.drop_duplicates())
        df2.gender = le.transform(df2.gender)
        le.fit(df2.smoker.drop_duplicates()) 
        df2.smoker = le.transform(df2.smoker)
        df2 = pd.get_dummies(df2, columns = ['region'], drop_first=True)
        y = df2['charges']
        X = df2.drop(['charges'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

        lr = LinearRegression().fit(X_train,y_train)
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)
        fig = px.scatter(
            
            x=y_test,
            y=y_test_pred
#            size="pop",
#            color="continent",
#            hover_name="country",
#            log_x=True,
#            size_max=60,
        )

        st.code(f'coefficients {lr.coef_}')
        st.write("Plotting Actual Charges and Predicted Charges")
        st.plotly_chart(fig, theme=None, use_container_width=True)
    

    