import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\purva\Downloads\Dashboard\PCA\PCA_Data.csv')
st.set_page_config(layout ='wide', page_title='Principal Componant Analysis', page_icon = ':fallen_leaf:')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Features','Normalized','Feature_Transformation','Information','Profile'])

with tab1:
    df.head()

    features = df.copy()

    st.write(features)

with tab2:

    from sklearn.decomposition import PCA

    model = PCA(n_components=4, random_state=4)

    from sklearn.preprocessing import StandardScaler

    features = StandardScaler().fit_transform(df)

    st.write(features)

# Visual Python: Machine Learning > Fit/Predict
with tab3:

    model.fit(features)

    # Visual Python: Machine Learning > Dimension
    from sklearn.decomposition import PCA

    model = PCA(n_components=4, random_state=4)
    trans = model.fit_transform(features)

    st.write(trans)

    pca_df = pd.DataFrame(data=trans, columns=['PC1','PC2', 'PC3','PC4'])

    st.write(pca_df.head(10))

with tab4:

    eigenvalues = model.explained_variance_
    prop_var = eigenvalues / np.sum(eigenvalues)
    plt.figure(figsize=(14,10))
    plt.plot(np.arange(1, len(eigenvalues)+1),
            eigenvalues, marker='o')
    plt.xlabel('Principal Component',
            size = 20)
    plt.ylabel('Eigenvalue',
            size = 20)
    plt.title('Figure 1: Scree Plot for Eigenvalues',
            size = 25)
    plt.axhline(y=1, color='r',
                linestyle='--')
    plt.grid(True)
    plt.savefig('scree_plot.png')
    st.image('scree_plot.png', use_column_width=True)

with tab5:

    from sklearn.decomposition import PCA

    model = PCA(n_components=4, random_state=4)
    trans = model.fit_transform(features)
    pca_df = pd.DataFrame(data=trans, columns=['PC1','PC2', 'PC3','PC4'])
    df = pd.DataFrame(model.components_, columns=list(df.columns))

    st.write(df.head())

    import seaborn as sns
    sns.heatmap(df, cmap ='RdYlGn', linewidths = 0.50, annot = True)
    plt.savefig('scree_plot.png')
    st.image('scree_plot.png', use_column_width=True)

