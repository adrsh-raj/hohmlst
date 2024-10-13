from libs import *

def main():
    set_config(display='diagram')
    st.header("Pipelines")
    st.write("*Pipeline chains together multiple steps so that the output of each step is used as input to the next step*. Pipelines make it easy to apply the same preprocessing to train and test.")

    df = pd.read_csv("used_datasets/train.csv")
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    st.dataframe(df.head(5))
    st.write("We are going to use titanic data sets and drop some columns")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'], test_size=0.2)
    st.dataframe(X_train.head(5))

    st.latex(r'\text{Input Data} \rightarrow \text{Missing values/ Impute} \rightarrow \text{OHE} \rightarrow \text{scaling} \rightarrow \text{Model}')

    st.write("Our first step is to do impute by column transformer, we use index value coz values are converted to numpy array not in dataframe")

    trf1 = ColumnTransformer([
        ('impute_age', SimpleImputer(strategy='mean'), [2]),         
        ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6]) 
    ], remainder='passthrough')

    trf2 = ColumnTransformer([
        ('ohe_sx_embarked', OneHotEncoder(sparse=False, handle_unknown='ignore'), [1, 6])  
    ], remainder='passthrough')

    trf3 = ColumnTransformer([
        ('scale', MinMaxScaler(), slice(0, 8)) 
    ])


    trf4 = SelectKBest(score_func=chi2, k=5)

    trf5 = DecisionTreeClassifier()

    pipe = Pipeline([
        ('trf1', trf1),
        ('trf2', trf2),
        ('trf3', trf3),
        ('trf4', trf4),
        ('trf5', trf5),
    ])

    st.code("""
    trf1 = ColumnTransformer([
        ('impute_age', SimpleImputer(strategy='mean'), [2]),         
        ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6]) 
    ], remainder='passthrough')

    trf2 = ColumnTransformer([
        ('ohe_sx_embarked', OneHotEncoder(sparse=False, handle_unknown='ignore'), [1, 6])  
    ], remainder='passthrough')

    trf3 = ColumnTransformer([
        ('scale', MinMaxScaler(), slice(0, 8)) 
    ])
            
    trf4 = SelectKBest(score_func=chi2, k=5)
            
    trf5 = DecisionTreeClassifier()

    pipe = Pipeline([
        ('trf1', trf1),
        ('trf2', trf2),
        ('trf3', trf3),
        ('trf4', trf4),
        ('trf5', trf5),
    ])
    pipe.fit(X_train, y_train)
""")
    st.write("Output")

    st.code(f"{pipe.fit(X_train, y_train)}")
