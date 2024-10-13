import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.title("Univariate analysis")
    st.write("Independent analysis of every indivisual colums is univariate analysis, Data of generally two type **Numerical and catgorical**. Here we are going to consider titanic datasets, let's load our data")

    titanic = pd.read_csv('used_datasets/titanic.csv')
    st.dataframe(titanic)

    st.write("Let's gather information about dataset, univariately, first we'll start with categorical columns")

    st.write("### 1) Categorical data ")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(data=titanic, x='Survived', ax=ax[0], color='green')
    ax[0].set_title("Survived")
    sns.countplot(data=titanic, x='Pclass', ax=ax[1], color='green')
    ax[1].set_title("Pclass")
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### 2) PieChart ")
    column = st.selectbox("Enter the categorical column to the plot", ['Pclass', 'Sex'])
    st.write(f"Pie chart of {column}")
    fig, ax = plt.subplots(figsize=(10, 3))
    column_data = titanic[column].value_counts()
    ax.pie(column_data, labels=column_data.index, autopct="%1.1f", startangle=90)
    ax.axis('equal')
    st.pyplot(fig=fig)

    st.write("### 2) Numerical data")
    st.write("##### Histogram (Age)")
    bins = st.slider("Bins", 5, 20, 8, step=1)
    fig, ax = plt.subplots(figsize=(14, 5))
    plt.hist(titanic['Age'], bins=bins)
    st.pyplot(fig)

    st.write("##### Distplot")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.distplot(titanic['Age'], bins=bins, kde=True)
    st.pyplot(fig=fig)

    st.write("##### Boxplot")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(titanic['Age'], orient='h')
    st.pyplot(fig=fig)

