from libs import *
def main():
    st.header('Encoding categorical variables')
    st.write("Data are of two type:")
    st.write("1) *Numerical* ")
    st.write("2) *Categorical* in two nominal and ordinal")
    st.write("In nominal data, categories have no relation or border (each of them are equal ex in engineering barches, you cant say CSE is better than mechanical), we use one hot encoding")
    st.write("In ordinal data, categories have relation or border (good, bad), it has ordering we use ordinal encoding in input data, there is one more encoding for labels is *label encoding* Y")

    st.write("#### Ordinal Encoding")
    df = pd.read_csv("used_datasets/customer.csv")
    st.dataframe(df.sample(5))
    st.write("Here gender, purhased is nominal data, review, education is ordinal, here we will focus in review education and purchased ")
    df = df.iloc[:, 2:]
    st.dataframe(df.head(5))

    X_train, X_test, y_train, y_test  = train_test_split(df.iloc[:, 0:2], df.iloc[:, -1], test_size=0.2)
    oe = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], ['School', 'UG', 'PG']])
    oe.fit(X_train)
    X_train_scaled = oe.transform(X_train)
    X_train_scaled = oe.transform(X_test)

    st.code("""

    X_train, X_test, y_train, y_test  = train_test_split(df.iloc[:, 0:2], df.iloc[:, -1], test_size=0.2)
    oe = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], ['School', 'UG', 'PG']])
    oe.fit(X_train)
    X_train_scaled = oe.transform(X_train)
    X_train_scaled = oe.transform(X_test)

""", language='python')
    st.write("*Output*")
    st.code(f"Xtrain: {X_train_scaled[:5]}")

    st.write("#### One Hot Encoding")
    st.write("One-hot encoding is a technique used to convert categorical variables into a binary matrix. Each category is represented by a vector where only one element is '1' (hot), and the rest are '0'.")

    df = pd.read_csv('used_datasets/cars.csv')
    st.dataframe(df.head(5))
    st.code(f"shape is {df.shape}")

    st.write("#### OHE using pandas")
    st.code("""dummies = pd.get_dummies(df, columns=['fuel', 'owner'], dtype=int)
st.dataframe(dummies.head(5))""", language='python')
    dummies = pd.get_dummies(df, columns=['fuel', 'owner'], dtype=int)
    st.dataframe(dummies.head(5))
    st.code(f"shape after OHE is {dummies.shape}")


    st.write("Multicolinearity will remain problem here to solve it we use K-1 encoding")

    st.write("#### K-1 OHE")
    dummies = pd.get_dummies(df, columns=['fuel', 'owner'], dtype=int, drop_first=True)
    st.code("""dummies = pd.get_dummies(df, columns=['fuel', 'owner'], dtype=int, inplace=True)""", language='python')
    st.dataframe(dummies.head(5))

    st.code(f"shape after OHE is {dummies.shape}")


    st.write("#### OHE using SKlearn")
    st.write("we use sklearn OHE coz it remembers all the values")

    X_train, X_test, y_train, y_test  = train_test_split(df.iloc[:, 0:4], df.iloc[:, -1], test_size=0.2)

    st.dataframe(X_train.head(5))

    ohe= OneHotEncoder(drop='first')
    X_train_new = ohe.fit_transform(X_train[['fuel', 'owner']]).toarray()
    X_test_new = ohe.fit_transform(X_test[['fuel', 'owner']]).toarray()
    np.hstack((X_train[['brand', 'km_driven']].values, X_train_new))
    st.code("""
    ohe= OneHotEncoder(drop='first')
    X_train_new = ohe.fit_transform(X_train[['fuel', 'owner']]).toarray()
    X_test_new = ohe.fit_transform(X_test[['fuel', 'owner']]).toarray()
    np.hstack((X_train[['brand', 'km_driven']].values, X_train_new))
""", language='python')
    st.code(f"{np.hstack((X_train[['brand', 'km_driven']].values, X_train_new))}")



    st.write("#### OHE using Top categories")

    st.code("""

    counts = df['brand'].value_counts()
    threshold = 100
    df['brand'].nunique()
    repl = counts[counts <=threshold].index
    dummies = pd.get_dummies(df['brand'].replace(repl, 'uncommon'), dtype=int)
""")

    counts = df['brand'].value_counts()
    threshold = 100
    df['brand'].nunique()
    repl = counts[counts <=threshold].index
    dummies = pd.get_dummies(df['brand'].replace(repl, 'uncommon'), dtype=int)
    st.dataframe(dummies.head(5))


    st.write("#### Using column transformer")
    st.write("ColumnTransformer in scikit-learn allows you to apply different preprocessing steps to different columns of your dataset.  The SimpleImputer in scikit-learn is a powerful tool for handling missing data by imputing (filling) missing values with strategies like the mean, median, most frequent value, or a constant value.")

    df = pd.read_csv('used_datasets/covid_toy.csv')
    st.dataframe(df.head(5))
    st.write("We will use gender OHE, fever SI, cough OE, city OHE")

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['has_covid']), df['has_covid'], test_size=0.2)

    transformer = ColumnTransformer(transformers=[
        ('tnf1', SimpleImputer(), ['fever']), 
        ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']), 
        ('tnf3', OneHotEncoder(sparse=False, drop='first'), ['gender', 'city']) 

    ], remainder='passthrough')

    st.code("""
        transformer = ColumnTransformer(transformers=[
        ('tnf1', SimpleImputer(), ['fever']), 
        ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']), 
        ('tnf3', OneHotEncoder(sparse=False, drop='first'), ['gender', 'city']) 

    ], remainder='passthrough')

""")

    st.code(f"{transformer.fit_transform(X_train)[:5]}", language='python')