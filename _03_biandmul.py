import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    iris = sns.load_dataset('iris')
    tips = sns.load_dataset('tips')
    flights = sns.load_dataset('flights')
    titanic = pd.read_csv('used_datasets/titanic.csv')

    st.title("Bivariate and multivariate analysis")
    st.write("Here we are going to visualize multiple combinations of dataset columns to extract informations from it. Datasets are **isis, tips, flights and titanics** thats how dataset looks like with 10rows each")

    iris_05 = iris.head(5)
    tips_05 = tips.head(5)
    flights_05 = flights.head(5)
    titanic_05 = titanic.head(5)

    st.write("#### Iris:")
    st.dataframe(iris_05)
    st.write("#### Tips:")
    st.dataframe(tips_05)
    st.write("#### Flight:")
    st.dataframe(flights_05)
    st.write("#### Titanic:")
    st.dataframe(titanic_05)

    st.write("#### 1) Scatterplot (Numerical - Numerical) tips dataset")
    code_1 = """
        fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=tips, x='total_bill', y='tip', c='red')
"""
    st.code(code_1, language='python')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=tips, x='total_bill', y='tip', c='red')
    st.pyplot(fig=fig)

    column = st.selectbox("Gender wise", ['sex', 'smoker', 'size'])
    st.write("#### 1) Scatterplot (NUmerical - Numerical) tips dataset")
    code_2 = """
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=tips, x='total_bill', y='tip', c='red', hue= ['sex', 'smoker', 'size'])
    """
    st.code(code_2, language='python')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=tips, x='total_bill', y='tip', c='red', hue=column)
    st.pyplot(fig=fig)

    st.write("#### 2) Bar (Numerical - Categorical) titanic dataset")
    code_3 = """
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=titanic,  x='Pclass', y=['Fare', 'Age'], hue='Sex')
"""
    st.code(code_3, language='python')
    relation = st.selectbox("Columns wise relation", ['Fare', 'Age'])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=titanic,  x='Pclass', y=relation, hue='Sex')
    st.pyplot(fig=fig)

    st.write("#### 3) Boxplot (Numerical - Categorical) titanic dataset")

    code_4 = """
sns.boxplot(data=titanic, x='Sex', y='Age', hue='Survived')

"""
    st.code(code_4, language='python')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=titanic, x='Sex', y='Age', hue='Survived')
    st.pyplot(fig=fig)

    st.write("#### 4) Distplot (Numerical - Categorical) titanic dataset")
    fig, ax = plt.subplots(figsize=(10, 5))
    st.write("Age who didnot survived (blue) and who survived (orange)")
    code_5 = """
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.distplot(titanic, x=titanic[titanic['Survived']==0]['Age'], kde=True, hist=False)
    sns.distplot(titanic, x=titanic[titanic['Survived']==1]['Age'], kde=True, hist=False)
"""
    st.code(code_5, language='python')
    sns.distplot(
        titanic, x=titanic[titanic['Survived'] == 0]['Age'], kde=True, hist=False)
    sns.distplot(
        titanic, x=titanic[titanic['Survived'] == 1]['Age'], kde=True, hist=False)
    st.pyplot(fig)

    st.write("#### 5) Heatmap (Categorical - Categorical) titanic dataset")
    st.write("By percentage terms based on pclass")
    code_6 = """
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))
"""
    st.code(code_6, language='python')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))
    st.pyplot(fig=fig)
