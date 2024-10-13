import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
def main():
    st.title("Feature Engineering")
    st.write("*Feature enginering is the process of domain knowledge to extract features from raw data. These features can be used to improve the performance of machine learning algorithms*")
    st.image('images/featureeng.webp')

    st.title("Feature Scaling")
    st.write("Feature scaling is a technique to standardize the independent feautes present in data in a fixed range, We need feature scaling imagine in euclidean space age and salary axis, and in KNN we calculate distance between salary and age, where salary will dominate.")
    st.write("#### Type of feature scaling")
    st.markdown("1) Standardization")
    st.markdown("2) Normalization")

    st.write("#### Standardization")
    st.write("Also called Z-SCORE normalization")

    st.latex(r'z_i = \frac{x_i - \bar{x}}{\sigma}')
    st.write("The mean of standardise value will be 0 and standard deviation will be 1, its mean centric")
    st.title("We will consider an example of standardization")
    df = pd.read_csv('used_datasets/Social_Network_Ads.csv')    
    df_05 = df.head(5)
    st.dataframe(df_05)
    encoder = OneHotEncoder(sparse=False)
    category_encoded = encoder.fit_transform(df[['Gender']])
    df.drop(columns=['Gender', 'User ID'], inplace=True)
    st.write('After applying OHE on Gender colums')
    df = df.join(pd.DataFrame(category_encoded, columns=encoder.categories_[0]))
    st.dataframe(df.head(5))

    X_train, X_test, y_train, y_test = train_test_split(df.drop('Purchased', axis=1),
                                                        df['Purchased'], 
                                                        test_size=0.3, 
                                                        random_state=2)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler.mean_

    st.code(f'''
    scaler = StandardScaler()
    #To learn on xtrain data, like pattern and range
    scaler.fit(X_train)
    #Transforming X_train, X_test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(scaler.mean_)
 
''', language='python')
    st.code(f"Output: {scaler.mean_[:2]}")

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    st.write("This is how dataframe look like after passing through standard scalar")
    st.dataframe(np.round(X_train_scaled.head(5)))
    st.write("After using pd.describe(), check what happened to mean and std columns")

    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Before describe")
        st.dataframe(np.round(X_train.describe(), 1))
    with col2:
        st.write("##### After describe")
        st.dataframe(np.round(X_train_scaled.describe(), 1))


    st.code("""
    st.write("Let's visualise what's changes in data")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
    ax1.set_title('Before Scaling')
    
    ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'], color='red')
    ax2.set_title('After Scaling')
    st.pyplot(fig)
    st.write("Surprisingly no changes in data nature")
  
""", language='python')


    st.write("Let's visualise what's changes in data")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
    ax1.set_title('Before Scaling')
    
    ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'], color='red')
    ax2.set_title('After Scaling')
    st.pyplot(fig)
    st.write("Surprisingly no changes in data nature")

    st.write("Let's check how the data is converted to normal distribution collectively")

    st.code("""
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    #before scaling
    ax1.set_title("Before scaling")
    sns.kdeplot(X_train['Age'], ax=ax1)
    sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

    #after scaling 
    ax2.set_title("After scaling")
    sns.kdeplot(X_train_scaled['Age'], ax=ax2)
    sns.kdeplot(X_train_scaled['EstimatedSalary'], ax=ax2)
    plt.show()
    st.pyplot(fig)
""", language='python')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    #before scaling
    ax1.set_title("Before scaling")
    sns.kdeplot(X_train['Age'], ax=ax1)
    sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

    #after scaling 
    ax2.set_title("After scaling")
    sns.kdeplot(X_train_scaled['Age'], ax=ax2)
    sns.kdeplot(X_train_scaled['EstimatedSalary'], ax=ax2)
    plt.show()
    st.pyplot(fig)
    st.write("*Better use of standardisation is K-means, KNN, PRINCIPLE COMPONENT ANALYSIS, ARTIFICIAL NEURAL NETWORK GRADIENT DESCENT* in KNN AND K-MEANS we use euclidian distances, so to reduce scale we standardise, in PCA to maximize variance, in gradient descent to converge for multiple weights, not use in decision tree, random forest, gradboost, Xgboost")