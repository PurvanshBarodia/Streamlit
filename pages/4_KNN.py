import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# from sklearn.datasets import load_iris

# df = load_iris()
df = pd.read_csv(r'C:\Users\purva\Downloads\Dashboard\Classification\iris.csv')
st.set_page_config(layout ='wide', page_title='K Nearest Neighbor Classification', page_icon = ':fallen_leaf:')


features = df.drop(['Species','Id'], axis=1)
target = df[['Species']]
a = """cm = confusion_matrix(y_true=y_test, y_pred=y_pred),
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Setosa','Versicolor','Verginica']),
st.write(disp.plot().figure_)"""
b = """features = df.drop(['Species','Id'], axis=1),
target = df[['Species']],
target['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[1,2,3],inplace=True)"""

tab1, tab2, tab3 = st.tabs(['Data','Features and Target', 'KNN Model'])

with tab1:
    st.code(line_numbers = True, body = """# Reading csv file in a dataframe. The datafile (csv) must be on the same path as the python file,
df = pd.read_csv('iris.csv'),
st.write(df.head(10))""")
    st.write(df.head(10))
with tab2:
    st.write("Separating Features and Targets")
    st.code(line_numbers = True, body = b)
    
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.write('Features')
        st.write(features.head(10))
    with col2:
        st.write('Target')
        st.write(target.head(10))
    with col3:
        st.write('Encoded Target')
        
        target['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[1,2,3],inplace=True)
        st.write(target.head(10))
with tab3:
    st.subheader('KNN Model')
    tab1, tab2, tab3, tab4 = st.tabs(['Train-Test','Fit Model','Predict','Evaluate'])
    with tab1:
        st.write('Splitting Features and Targets into Train and Test Datasets')
        st.code('X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)')
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)
        col1, col2, col3, col4 = st.columns([2,1,2,1])
        with col1:
            st.write('X Train')
            st.write(X_train)
        with col2:
            st.write('Y Train')
            st.write(y_train)
        with col3:
            st.write('X Test')
            st.write(X_test)
        with col4:
            st.write('Y Test')
            st.write(y_test)
        
    with tab2:
        
        st.code('knn = KNeighborsClassifier(n_neighbors=3) # Creating Model with 3 clusters')
        knn = KNeighborsClassifier(n_neighbors=3)
        st.write(knn)
        st.code('knn.fit(X_train, y_train) # Fitting Model on the training data')
        knn.fit(X_train,y_train)
        
    with tab3:
        
        y_pred = knn.predict(X_test)
        st.code('y_pred = knn.predict(X_test) # Predicting values for the test dataset')
        col1, col2 = st.columns([1,1])
        with col1:
            st.write('y_pred')
            st.write(y_pred)
        with col2:
            st.write('y_test')
            st.write(y_test)
    with tab4:
        st.subheader("Confusion Matrix")
    
        st.code( line_numbers = True, body= a)
        
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Setosa','Versicolor','Verginica'])
        st.write(disp.plot().figure_)
        st.subheader("Accuracy Score")
        accuracyScore = accuracy_score(y_true=y_test, y_pred=y_pred)
        st.code("accuracyScore = accuracy_score(y_true=y_test, y_pred=y_pred)")
        st.write("Accuracy Score : ", accuracyScore)
