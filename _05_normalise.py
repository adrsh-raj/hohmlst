from libs import *
def main():
    st.header("Normalzation")
    st.write("Normalization is a data prepration technique, the goal is to scale numeric value to use a common scale, without distorting a difference in ranges of values or losing information , there are several normalization techniques like *MinMaxScaling, mean normalizationm, max absolute, robust scaling*")

    st.write("##### Min Max scaling")
    # Display the Min-Max scaling formula using st.latex
    st.latex(r'X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}')
    st.latex(r'X_{\text{scaled}} \in [0, 1]')
    st.write("We compress the data in between [0, 1]")

    st.write("##### Mean Normalization")
    st.write("Mean normalization is a technique where you scale the data by subtracting the mean and dividing by the range. The formula for mean normalization is:")
    st.latex(r'X_{\text{normalized}} = \frac{X - \mu}{X_{\max} - X_{\min}}')
    st.latex(r'X_{\text{normalized}} \in [-1, 1]')


    st.write("##### Max Absolute scale")
    st.write("Max-Abs scaling is a technique where each feature is scaled by dividing by the maximum absolute value of that feature, ensuring the values range between -1 and 1, use it where sparse data ( 0 is too much)")
    st.latex(r'X_{\text{scaled}} = \frac{X}{|X_{\max}|}')
    st.latex(r'X_{\text{normalized}} \in [-1, 1]')


    st.write("##### Robust scale")
    st.write("Robust scaling is a technique that scales the data using the median and interquartile range (IQR), making it more robust to outliers. The formula for robust scaling is:")
    st.latex(r'X_{\text{scaled}} = \frac{X - \text{median}(X)}{IQR(75 PERCENTILE - 25 PERCENTILE)}')
    st.write("Doesnot have fixed range, it is roubust to outliers")
