import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


import os

st.set_page_config(layout='wide',page_icon=':shark:')
st.title('Classification and Clustering')

try:
    os.chdir(r'C:\Users\purva\Downloads\Dashboard\Classification')
except FileNotFoundError:
    print("")

#st.write(os.getcwd())

options = os.listdir()

#flSel = st.selectbox(label='Data Files',options=options)

#if st.button('Select a dataset'):
#    df = pd.read_csv(flSel)
#    st.write(df.head(5))
    
#if st.button('Logistic Regression'):
df = pd.read_csv('titanic.csv')
features = df.drop(['Survived'],axis=1)
target = df[['Survived']]
with st.expander('This model is for titanic.csv'):
#    df = pd.read_csv('titanic.csv')
    st.write('Total records : ', features.shape)
    st.write(df.head(10))
with st.expander('Let us separate features from the target variable'):
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("#### Features")
        
        st.write(features.head())
    with col2:
        st.markdown("#### Target")
        
        st.write(target.head())
with st.expander('Removing unwanted fields'):
#    ['Pclass','Gender','Age','SibSp','Parch','Fare','Cabin']
# Visual Python: Data Analysis > File
#	PassengerId 	Survived 	Pclass 	Name 	Gender 	Age 	SibSp 	Parch 	Ticket 	Fare 	Cabin 	Embarked
#    features = features.drop(['Name'],axis=1)
    st.write("Removing Cabin, Name, PassengerId, Ticket, and Embarked fields")
    features.drop(['Cabin','Name', 'PassengerId','Ticket','Embarked'],axis=1,inplace=True)
    
    st.write(features.head())
with st.expander('Encoding Gender field'):
    features['Gender'].replace(['female','male'],[0,1],inplace=True)
    st.write(features.head())
with st.expander('Replacing null values'):
    col1, col2 = st.columns([1,1])
    with col1:
        st.write('Age column has null values')
        st.write(features['Age'].isna())
    with col2:
        st.write('We will replace that with the median age')
        med = features['Age'].median()
        features['Age'].fillna(med,inplace=True)
        st.write(features['Age'].isna())
with st.expander('Splitting the data in training and testing dataframes (80,20)'):
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=123)
    with col1:
        st.write('Training Features Dataset - ', X_train.shape)
        st.write(X_train.head())
    with col2:
        st.write('Training Target Dataset')
        st.write(y_train.head())
    with col3:
        st.write('Testing Features Dataset - ', X_test.shape)
        st.write(X_test.head())
    with col4:
        st.write('Testing Target Dataset')
        st.write(y_test.head())
with st.expander('Fitting the model'):
    st.write('Fitting Logistic Regression Model')
    model = LogisticRegression(C=1.0, random_state=123)
    model.fit(X_train,y_train)
    score = model.score(X_train, y_train)
    st.write("Model Score : " , score)
    
        
with st.expander('Predicting Values with the Model'):
    y_pred = model.predict(X_test)
    rmse = metrics.mean_squared_error(y_test, y_pred)**0.5
    st.write("Model Root Mean Squared Error : ", rmse)    
with st.expander('ROC (Receiver Operating Characteristic) Curve'):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.decision_function(X_test))                                    
    plt.plot(fpr, tpr, label='ROC Curve')                                    
    plt.xlabel('Sensitivity')                                    
    plt.ylabel('Specificity')                                    
    st.pyplot(plt)

with st.expander('Evaluation of the model'):
    conf_mat_fig = plt.figure(figsize=(6,6))
    ax1 = conf_mat_fig.add_subplot(111)
    labels = ['True','False']
    # print(classification_report(y_test, y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred),display_labels=labels)
    st.write(disp.plot().figure_)
    


with st.expander('Let us understand the evaluation'):
    f1Score = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    prec = precision_score(y_test,y_pred)
    predicted_probabilities = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predicted_probabilities)
    
    st.markdown('#### We have three values to assess model with')
    st.markdown('#### Precision = True Positive / (True Positive + False Positive)')
    st.markdown('#### Recall = True Positive / True Positive + False Negative')
    st.markdown('#### F1 Score = Harmonic Mean of Precision and Recall')
    
    data = {
        "Parameter":['Precision','Recall','F1 Score'],
        "Value":[prec,recall,f1Score]
    }
    df1 = pd.DataFrame(data)
    st.table(df1)
    st.write('Area Under Curve (ROC) :', auc_score)
    st.write('AUC score of ', auc_score, ' suggests that the model is dependable and the predictions are more than just a random guess')
    
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.write('Probabilities ', predicted_probabilities)
    with col2:
        st.write('X Test values', X_test)
    with col3:
        st.write('Y Test values', y_test)
        
        