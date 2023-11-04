import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(layout='wide',page_title='KMeans Clustering')

df = pd.read_csv('../Classification/iris.csv')
features = df.drop(['Species'],axis=1)
target = df[['Species']]

#encode target values with LabelEncoder
le = LabelEncoder()
target['Species'] = le.fit_transform(target['Species'])



# target['Species'].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2],inplace=True)
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

with st.expander('Iris Data'):
    st.write(df.head(10))

with st.expander('Raw Plot'):
    fig, ax = plt.subplots()
    
    ax.scatter(df['SepalLengthCm'],df['SepalWidthCm'])
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
  
    
    st.pyplot(fig)
with st.expander('Let us check organic clusters possible'):
    inert = '''#Elbow model
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)'''

    st.code(line_numbers = True, body = inert)

    data = list(zip(x, y))

    inertias = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)


    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.figure(figsize=(5,3))
    st.pyplot(fig)

with st.expander('Splitting Training and Testing data'):
    st.write('Splitting Features and Targets into Train and Test Datasets')
    st.code('X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)')
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)
with st.expander('Model fit and prediction'):
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(features)
    cds = '''
kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
'''
    st.code(line_numbers = True, body = cds)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=kmeans.labels_)
    st.pyplot(fig)

with st.expander('Prediction and Evaluation'):
    st.write('Prediction')
    st.code('y_pred = kmeans.predict(X_test)')
    y_pred = kmeans.predict(X_test)
   
    
    accs = accuracy_score(y_test['Species'], y_pred)
    st.write(f'Accuracy score: {accs}')
    cm = confusion_matrix(y_test['Species'], y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Setosa','Versicolor','Virginica'])
    st.write(disp.plot().figure_)