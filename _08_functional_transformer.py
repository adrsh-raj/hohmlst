import scipy.stats
from libs import *
import scipy


@st.cache_data
def comp(X_train, X_train_transformed ):
    for col in X_train.columns:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        
        # Plot the distribution using seaborn on the first subplot
        sns.histplot(X_train[col], kde=True, ax=ax1)
        ax1.set_title(f"Distribution of {col}")
        
        # Plot the Q-Q plot on the second subplot
        sns.histplot(X_train_transformed[col], kde=True, ax=ax2)

        ax2.set_title(f"After transformation of {col}")
        
        # Display the figure in Streamlit
        st.pyplot(fig)


def main():
    
    st.header("Functional Transformer")
    st.write("*The operation over columns (mathematically) coz mostly ML algorithms works better on normal distribution*")

    st.write(" How to find data is normal? use seaborn.distplot or pd.skew() or qq plot")
    st.image("images/qqplot.jpg")
    st.write("O X axis theoratical quantiles and on y axis data sample quantiles, if data is 45degree with line then it's normally distributed here are different more examples")
    st.image("images/qqskewwed.webp")


    df = pd.read_csv('used_datasets/titanic.csv', usecols=['Age', 'Fare', 'Survived'])
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    st.write("We will use this datasets for examples taken")
    st.dataframe(df.head(5))
    X = df.iloc[:, 1:3]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    st.code("""
    #train test split
    X = df.iloc[:, 1:3]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

""")
    st.code("""
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Age PDF")
    sns.histplot(X_train['Age'], ax=ax1, kde=True)
    ax2.set_title("Age QQ plot")
    scipy.stats.probplot(X_train['Age'], dist='norm', plot=ax2)

""")
    st.write("**Output: Age distribution**")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Age PDF")
    sns.histplot(X_train['Age'], ax=ax1, kde=True)
    ax2.set_title("Age QQ plot")
    scipy.stats.probplot(X_train['Age'], dist='norm', plot=ax2)
    st.pyplot(fig)

    st.code("""
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Fare PDF")
    sns.histplot(X_train['Fare'], ax=ax1, kde=True)
    ax2.set_title("Fare QQ plot")
    scipy.stats.probplot(X_train['Fare'], dist='norm', plot=ax2)

""")
    st.write("**Output: Fare distribution**")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Fare PDF")
    sns.histplot(X_train['Fare'], ax=ax1, kde=True)
    ax2.set_title("Fare QQ plot")
    scipy.stats.probplot(X_train['Fare'], dist='norm', plot=ax2)
    st.pyplot(fig)




    st.write("#### Log Transform")
    st.write("Use it when data is right-skewed, not on negative data")
    st.code("""
    # Applying Functional transformer
    trf = FunctionTransformer(func=np.log1p)
    X_train_transformed = trf.fit_transform(X_train)
    X_test_transformed = trf.fit_transform(X_test)
            
    print("**Output: Fare before(left) | after(right) log transformer**")
    # Applying on Fare Log1P
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Before Transform")
    scipy.stats.probplot(X_train['Fare'], dist='norm', plot=ax1)
    ax2.set_title("After Transform")
    scipy.stats.probplot(X_train_transformed['Fare'], dist='norm', plot=ax2)
    st.pyplot(fig)

            
    # Applying on Age Log1P
    print("**Output: Age before(left) | after(right) log transformer** and due to forcebly transform it distribution get worse than last")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Before Transform")
    scipy.stats.probplot(X_train['Age'], dist='norm', plot=ax1)
    ax2.set_title("After Transform")
    scipy.stats.probplot(X_train_transformed['Age'], dist='norm', plot=ax2)
    st.pyplot(fig)


""", language='python')

    trf = FunctionTransformer(func=np.log1p)
    X_train_transformed = trf.fit_transform(X_train)
    X_test_transformed = trf.fit_transform(X_test)
    st.write("**Output: Fare before(left) | after(right) log transformer**")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Before Transform")
    scipy.stats.probplot(X_train['Fare'], dist='norm', plot=ax1)
    ax2.set_title("After Transform")
    scipy.stats.probplot(X_train_transformed['Fare'], dist='norm', plot=ax2)
    st.pyplot(fig)



    st.write("**Output: Age before(left) | after(right) log transformer** and due to forcebly transform it distribution get worse than last")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Before Transform")
    scipy.stats.probplot(X_train['Age'], dist='norm', plot=ax1)
    ax2.set_title("After Transform")
    scipy.stats.probplot(X_train_transformed['Age'], dist='norm', plot=ax2)
    st.pyplot(fig)

    st.write("You can do it with columntransformer too")
    trf2 = ColumnTransformer([
        ('log', FunctionTransformer(func=np.log1p), ['Fare'])
    ], remainder='passthrough')

    X_train_transformed = trf.fit_transform(X_train)
    X_test_transformed = trf.fit_transform(X_test)

    st.code("""
    trf2 = ColumnTransformer([
        ('log', FunctionTransformer(func=np.log1p), ['Fare'])
    ], remainder='passthrough')

    X_train_transformed = trf.fit_transform(X_train)
    X_test_transformed = trf.fit_transform(X_test)
""", language='python')


    st.write("Lets write a common function for other transformer so that we can avoid  DRY")


    st.code("""
    def apply_transformer(transformer):
        X = df.iloc[:, 1:3]
        y = df.iloc[:, 0]

        # Create a ColumnTransformer
        trf2 = ColumnTransformer([
            ('log', FunctionTransformer(func=transformer), ['Fare'])
        ], remainder='passthrough')
        
        # Fit and transform the data
        X_TRANS = trf2.fit_transform(X)

        # Extract the transformed 'Fare' values
        # The transformed data is now a NumPy array, and you need to access it appropriately
        fare_transformed = X_TRANS[:, 0]  # Since 'Fare' is the first column in the transformed output
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        ax1.set_title("Before Transform")
        scipy.stats.probplot(X['Fare'], dist='norm', plot=ax1)
        ax2.set_title("After Transform")
        scipy.stats.probplot(fare_transformed, dist='norm', plot=ax2)

        # Show the plot in Streamlit
        st.pyplot(fig)

    apply_transformer(lambda x: np.log1p(x))
    #You can see Fare same as above result
""")

    def apply_transformer(transformer):
        X = df.iloc[:, 1:3]
        y = df.iloc[:, 0]

        # Create a ColumnTransformer
        trf2 = ColumnTransformer([
            ('log', FunctionTransformer(func=transformer), ['Fare'])
        ], remainder='passthrough')
        
        # Fit and transform the data
        X_TRANS = trf2.fit_transform(X)

        # Extract the transformed 'Fare' values
        # The transformed data is now a NumPy array, and you need to access it appropriately
        fare_transformed = X_TRANS[:, 0]  # Since 'Fare' is the first column in the transformed output
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        ax1.set_title("Before Transform")
        scipy.stats.probplot(X['Fare'], dist='norm', plot=ax1)
        ax2.set_title("After Transform")
        scipy.stats.probplot(fare_transformed, dist='norm', plot=ax2)

        # Show the plot in Streamlit
        st.pyplot(fig)

    apply_transformer(lambda x: np.log1p(x))

    st.write("#### Reciprocal Transform")
    st.write("All small values become large and large values become small")

    st.code("apply_transformer(lambda x: 1/(x+0.1)) #Fare")
    apply_transformer(lambda x: 1/(x+0.1))

    st.write("##### Square Transform")
    st.write("Use it for left squared data")
    st.code("apply_transformer(lambda x: x**2) #Fare")
    apply_transformer(lambda x: x**2)

    st.write("##### Sqrt Transform")
    st.write("Without the square root scaling, the values in the numerator could become very large when the dimensionality of the vectors increases, leading to vanishing gradients in training. The square root helps normalize these values to a manageable range, ensuring stable gradients.")

    st.code("apply_transformer(lambda x: np.sqrt(x)) #Fare")
    apply_transformer(lambda x: np.sqrt(x))



    st.write("#### Box-Cox Transformation")
    st.write("The exponent here is variable called lambda that varies over -5 to 5 and in the process of seaching , we examine all the values of lambda, Finally we choose optimal values, its general formula of log , square transformation. Drawback is strctly applicable number greater than 0")
    st.latex(r"""
    y(\lambda) = 
    \begin{cases} 
    \frac{y^\lambda - 1}{\lambda}, & \text{if } \lambda \neq 0 \\
    \ln(y), & \text{if } \lambda = 0 
    \end{cases}
    """)

    df = pd.read_csv("used_datasets/concrete_data.csv")
    st.dataframe(df.sample(5))
    st.write("We will use given dataset and lets extract X and y")

    X = df.drop(columns=['Strength'])
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    st.code("""
    X = df.drop(columns=['Strength'])
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
""", language='python')

    st.write("Lets see the distribution of X train dataset")
    for col in X_train.columns:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        
        # Plot the distribution using seaborn on the first subplot
        sns.histplot(X_train[col], kde=True, ax=ax1)
        ax1.set_title(f"Distribution of {col}")
        
        # Plot the probability plot (Q-Q plot) on the second subplot
        scipy.stats.probplot(X_train[col], dist='norm', plot=ax2)
        ax2.set_title(f"Q-Q Plot of {col}")
        
        # Display the figure in Streamlit
        st.pyplot(fig)

    pt = PowerTransformer(method='box-cox')
    X_train_transformed = pt.fit_transform(X_train+0.0000001)
    X_test_transformed = pt.fit_transform(X_test + 0.0000001)

    st.code("""
    pt = PowerTransformer(method='box-cox')
    X_train_transformed = pt.fit_transform(X_train+0.0000001)
    X_test_transformed = pt.fit_transform(X_test + 0.0000001)

""")
    pd.DataFrame({'cols': X_train.columns, 'box_cox_lambdas': pt.lambdas_})
    st.dataframe(pd.DataFrame({'cols': X_train.columns, 'box_cox_lambdas': pt.lambdas_})
)
    st.write("Lets visualize the transformation")

    X_train_transformed = pd.DataFrame(X_train_transformed, columns=X_train.columns)
    comp(X_train, X_train_transformed)

    st.write("#### Yeo-Johnson transformation")
    st.write("Somewhat adjustment to the Box-Cox, by which we can apply it to negative number")
    st.latex(r"""
    y(\lambda) = 
    \begin{cases} 
    \frac{((y + 1)^\lambda - 1)}{\lambda}, & \text{if } \lambda \neq 0, \ y \geq 0 \\
    \ln(y + 1), & \text{if } \lambda = 0, \ y \geq 0 \\
    \frac{-(|y| + 1)^{2 - \lambda} - 1}{2 - \lambda}, & \text{if } \lambda \neq 2, \ y < 0 \\
    -\ln(|y| + 1), & \text{if } \lambda = 2, \ y < 0
    \end{cases}
    """)


    pt2 = PowerTransformer(method='yeo-johnson')
    X_train_transformed = pt2.fit_transform(X_train)
    X_test_transformed = pt2.fit_transform(X_test )

    st.code("""
    pt2 = PowerTransformer(method='yeo-johnson')
    X_train_transformed = pt2.fit_transform(X_train)
    X_test_transformed = pt2.fit_transform(X_test)

""")
    pd.DataFrame({'cols': X_train.columns, 'box_cox_lambdas': pt.lambdas_})
    st.dataframe(pd.DataFrame({'cols': X_train.columns, 'box_cox_lambdas': pt.lambdas_})
)
    st.write("Lets visualize the transformation")

    X_train_transformed = pd.DataFrame(X_train_transformed, columns=X_train.columns)

    comp(X_train, X_train_transformed=X_train_transformed)
    comparision = pd.DataFrame({
        'cols': X_train.columns, 'box_cox': pt.lambdas_, 'yeo-johnson': pt2.lambdas_
    })

    st.dataframe(comparision)



