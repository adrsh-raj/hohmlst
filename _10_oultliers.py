from libs import *

def main():

    st.header("Outliers")
    # Set title
    st.write("Outliers are data points that are significantly different from the rest of the data. These values are unusually high or low compared to other observations and can skew statistical analyses. Identifying outliers helps in understanding the distribution of data and improving model performance by either accounting for or removing them. Outliers are not good for Linear, logistic, adaboost and in deeplearning too.. (weights based)")
    # Generate sample data with outliers    

    st.write("How to treat outliers? Trimming, capping (limit on distribution), to act them as missing values, and discreatization based on starndard deviation")

    st.write("**Using Z-score**")

    st.latex(r'''
    Z = \frac{X - \mu}{\sigma}
    ''')
    data = pd.read_csv("used_datasets/placement.csv")
    st.dataframe(data.head(5))

    st.code("""
    def plot_(data, param_1, param_2, kde=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        plt.subplots(121)
        sns.histplot(data[param_1], ax=ax1, kde=kde)
        plt.subplots(122)
        sns.histplot(data[param_2], ax=ax2, kde=kde)
        st.pyplot(fig=fig)
    plot_(data=data, param_1='cgpa', param_2='placement_exam_marks', kde=True)
""")
    @st.cache_data
    def plot_(data, param_1, param_2, kde=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        plt.subplots(121)
        sns.histplot(data[param_1], ax=ax1, kde=kde)
        plt.subplots(122)
        sns.histplot(data[param_2], ax=ax2, kde=kde)
        st.pyplot(fig=fig)
    plot_(data=data, param_1='cgpa', param_2='placement_exam_marks', kde=True)

    st.write("You can see that placement section is right skewed")
    mean_cgpa = (f"Mean value of cgpa is {data['cgpa'].mean()}")
    std_cgpa = (f"Mean value of cgpa is {data['cgpa'].std()}")
    min_cgpa = (f"Mean value of cgpa is {data['cgpa'].min()}")
    max_cgpa = (f"Mean value of cgpa is {data['cgpa'].max()}")
    st.code(f"""
    mean_cgpa = (f"Mean value of cgpa is {data['cgpa'].mean()}")
    std_cgpa = (f"Mean value of cgpa is {data['cgpa'].std()}")
    min_cgpa = (f"Mean value of cgpa is {data['cgpa'].min()}")
    max_cgpa = (f"Mean value of cgpa is {data['cgpa'].max()}")
    mean_cgpa : {mean_cgpa}
    std_cgpa : {std_cgpa}
    min_cgpa : {min_cgpa}
    max_cgpa : {max_cgpa}
""")

    st.write("Lets find boundry values of cgpa")
    high = data['cgpa'].mean() + 3* data['cgpa'].std()
    low = data['cgpa'].mean() - 3* data['cgpa'].std()

    st.code(f"""
    high = data['cgpa'].mean() + 3* data['cgpa'].std()
    low = data['cgpa'].mean() - 3* data['cgpa'].std()
    high : {high}
    low : {low}
""")
    
    st.write("to find bouondry outlier is high and low of cgpa (variable here)")
    outliers = data[(data['cgpa'] > 8.80) | (data['cgpa'] < 5.11 )]
    st.code(f"{outliers}")

    st.write("Dropping outliers")
    data['cgpa_z_score'] = (data['cgpa'] - data['cgpa'].mean()) / data['cgpa'].std()
    st.code("data['cgpa_z_score'] = (data['cgpa'] - data['cgpa'].mean()) / data['cgpa'].std()")
    st.code(f"{data.head(5)}")
    st.write("FInding outliers")
    st.code(data[(data['cgpa_z_score']>3 ) | (data['cgpa_z_score'] <-3)])
    st.write("**Capping**")
    data['cgpa'] = np.where(
        data['cgpa']>high, high, np.where(data['cgpa'] <low, low, data['cgpa'])
    )
    st.code("""
    st.code(data[(data['cgpa_z_score']>3 ) | (data['cgpa_z_score'] <-3)])
    st.write("**Capping**")
    data['cgpa'] = np.where(
        data['cgpa']>high, high, np.where(data['cgpa'] <low, low, data['cgpa'])
    )
    st.code
""")
    st.code(data.shape)



    st.write("### Outlier detection with IQR and box plot") 
    st.write("IQR (Interquartile Range) is a measure of statistical dispersion, which is the spread of the data points in a dataset. It is calculated as the difference between the first quartile (𝑄1) and the third quartile (𝑄3) of the data.")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=data, x=data['placement_exam_marks'])
    st.pyplot(fig=fig)

    st.write("First find IQR")
    percentile25 = data['placement_exam_marks'].quantile(0.25)
    percentile75 = data['placement_exam_marks'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 *iqr
    lower_limit = percentile25 - 1.5 *iqr  
    st.code("""
    percentile25 = data['placement_exam_marks'].quantile(0.25)
    percentile75 = data['placement_exam_marks'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 *iqr
    lower_limit = percentile25 - 1.5 *iqr  
""")
    st.code(f'''
    75 percentile: {percentile25}
    25 percentile: {percentile75}
    upper limit: {upper_limit}
    lower limit: {lower_limit}
''')
    
    st.write("**Trimming**")
    new_df = data[data['placement_exam_marks']< upper_limit]
    st.code(new_df.shape)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    sns.histplot(data['placement_exam_marks'], ax=axs[0, 0], kde=True)
    axs[0, 0].set_title('Distribution of Placement Exam Marks (Original Data)')

    sns.boxplot(data=data['placement_exam_marks'], ax=axs[0, 1], orient='h')
    axs[0, 1].set_title('Boxplot of Placement Exam Marks (Original Data)')

    sns.histplot(new_df['placement_exam_marks'], ax=axs[1, 0], kde=True)
    axs[1, 0].set_title('Distribution of Placement Exam Marks (New Data)')

    sns.boxplot(data=new_df['placement_exam_marks'], ax=axs[1, 1], orient='h')
    axs[1, 1].set_title('Boxplot of Placement Exam Marks (New Data)')

    plt.tight_layout()

    st.pyplot(fig)


    st.write("### Winsorization")
    st.write("Winsorization is a statistical technique used to reduce the influence of outliers in a dataset. It involves replacing extreme values with a specified percentile value. For example, you might replace values below the 5th percentile with the value at the 5th percentile and values above the 95th percentile with the value at the 95th percentile.")



    # Function to apply Winsorization
    def winsorize_data(data, lower_percentile=0.05, upper_percentile=0.95):
        lower_bound = data.quantile(lower_percentile)
        upper_bound = data.quantile(upper_percentile)
        return np.clip(data, lower_bound, upper_bound)

    # Sample DataFrame with outliers
    data = pd.DataFrame({
        'placement_exam_marks': [55, 60, 72, 68, 62, 71, 75, 64, 70, 100, 120]  # Contains outliers
    })

    # Streamlit app title
    st.title("Winsorization Technique for Outliers")

    # Display original data
    st.write("Original Data:")
    st.write(data)

    # Winsorize the data
    winsorized_data = winsorize_data(data['placement_exam_marks'])

    # Create DataFrames for comparison
    original_df = pd.DataFrame({'Original': data['placement_exam_marks'], 'Winsorized': winsorized_data})

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Boxplot for original data
    sns.boxplot(data=data, x='placement_exam_marks', ax=ax[0])
    ax[0].set_title('Original Data with Outliers')

    # Boxplot for winsorized data
    sns.boxplot(data=winsorized_data, ax=ax[1])
    ax[1].set_title('Winsorized Data (Outliers Reduced)')

    # Adjust layout
    plt.tight_layout()

    # Display the plots in Streamlit
    st.pyplot(fig)

    # Show the comparison DataFrame
    st.write("Comparison of Original and Winsorized Data:")
    st.write(original_df)
