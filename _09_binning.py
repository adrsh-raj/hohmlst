from libs import *

def main():
    st.header("Binning and Binarization")
    st.write("Binning refers to converting continuous numerical variables into discrete intervals or bins. This technique is used when we want to reduce the complexity of a dataset by grouping continuous variables into ranges or categories.")

    st.write("Binarization is a process of converting continuous numerical features into binary (0 or 1) values")
    st.write("The two techniques are Discreatization and binarization.")


    st.write("#### Binning")
    st.write("Discretization is the process of transforming continous variables into discrete variables by creating a set of contigunous intervals that span the range of the variable values.")

    st.write("*It has three types a) unsupervised (equal width, equal frequency, k-mean binning) b) decision tree binning c) custom binning")

    st.write("#### a) Equal width binning")
    
    n_points = st.slider("Select number of data points", 50, 500, 100)  # Slider to adjust the data size
    data = np.random.randn(n_points) * 10 + 50  # Normally distributed data

    # Select number of bins
    n_bins = st.slider("Number of bins", 2, 10, 5)

    # Show sample data
    col1 , col2 = st.columns(2)
    with col1:
        st.write("Sample Data:")
        st.write(pd.DataFrame(data, columns=["Data"]).head())
    with col2:
        st.write("Equal-Width Binning (Uniform Binning):")
        bin_labels = [f"Bin{i}" for i in range(1, n_bins + 1)]
        data_binned_equal_width = pd.cut(data, bins=n_bins, labels=bin_labels)
        df_equal_width = pd.DataFrame({'Data': data, 'Binned Data': data_binned_equal_width})
        st.write(df_equal_width.head())

    # Plot Equal-Width Binning
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data, bins=n_bins, kde=False, ax=ax)
    ax.set_title("Equal-Width Binning")
    st.pyplot(fig)

    # Equal-Frequency Binning
    st.write("### Equal-Frequency Binning (Quantile Binning):")
    data_binned_equal_freq = pd.qcut(data, q=n_bins, labels=bin_labels)

    # Create a DataFrame to show results
    df_equal_freq = pd.DataFrame({'Data': data, 'Binned Data': data_binned_equal_freq})
    st.write(df_equal_freq.head())

    # Plot Equal-Frequency Binning
    fig, ax = plt.subplots()
    sns.histplot(data_binned_equal_freq, kde=False, ax=ax)
    ax.set_title("Equal-Frequency Binning")
    st.pyplot(fig)

    st.write("### Explanation")
    st.write("""
    - **Equal-width binning** divides data into equal-sized intervals (bins). Some bins may contain more data points if the data is skewed.
    - **Equal-frequency binning** creates bins such that each bin has an equal number of data points, making it useful for data that is unevenly distributed.
    """)


    
    st.write("""
    - **K-Means binning** groups data points into `k` clusters by minimizing the variance within each cluster. 
    - This binning method is useful when you want to group data based on their inherent distribution rather than predefined intervals like in equal-width or equal-frequency binning.
    """)
    n_points = st.slider("Select number of data points", 50, 500, 100, key='1')  # Slider to adjust data size
    data = np.random.randn(n_points) * 10 + 50  # Normally distributed data

    # Number of bins (clusters)
    n_bins = st.slider("Number of bins (K-Means clusters)", 2, 10, 4)

    # Show sample data
    st.write("Data: ")
    col1, col2 = st.columns(2)
    with col1:
            
        st.write(pd.DataFrame(data, columns=["Data"]).head())

        # Reshaping the data for KMeans
        data_reshaped = data.reshape(-1, 1)
    with col2:
    # Applying K-Means clustering
        kmeans = KMeans(n_clusters=n_bins)
        kmeans.fit(data_reshaped)

        # Assigning each data point to a cluster (bin)
        labels = kmeans.labels_

        # Creating a DataFrame to show the binned data
        df_kmeans = pd.DataFrame({'Data': data, 'Cluster (Bin)': labels})
        st.write(df_kmeans.head())

    # Plot K-Means Binned Data
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data, bins=n_bins, kde=False, ax=ax)
    ax.set_title(f"K-Means Binning with {n_bins} Clusters")
    st.pyplot(fig)

    # Plot the clustering
    fig, ax = plt.subplots()
    plt.scatter(data, np.zeros_like(data), c=labels, cmap='viridis')
    plt.title(f"K-Means Clustering with {n_bins} Bins")
    st.pyplot(fig)
