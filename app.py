import streamlit as st
import numpy as np
import matplotlib.pylab as plt
import _01_tensors 
import _02_univariate, _03_biandmul, _04_feature_eng, _05_normalise, _06_encoding_cat_data, _07_pipeline, _08_functional_transformer, _09_binning, _10_oultliers, _11_pca



# Create a sidebar for navigation
page = st.sidebar.selectbox(
    "Select a Page",
    ["Home", "tensors", "univariate", "biandmultivariate", "feature_engineering", "Normalization", "Encoding Categorical", "Pipeline", "Functional Transormer", "Binning", "Missing data", "PCA"]
)



def main():
    if page == 'Home':
      
        st.title("Welcome to ML mathematics Page")
        st.write("Here you can learn and discover various topics of Mathematics and stats..with visualization like playing CS-GO")

        st.sidebar.markdown("## Controls")
        st.sidebar.markdown("You can **change** the values to change the *chart*.")
        x = st.sidebar.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
        y = st.sidebar.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)

        st.write(f"x={x} y={y}")
        values = np.cumprod(1 + np.random.normal(x, y, (100, 10)), axis=0)


        fig, ax = plt.subplots()
        for i in range(values.shape[1]):
                ax.plot(values[:, i])

        st.pyplot(fig=fig)

    elif page=='tensors':
        _01_tensors.main()

    elif page=='univariate':
        _02_univariate.main()

    elif page=='biandmultivariate':
        _03_biandmul.main()

    elif page=='feature_engineering':
        _04_feature_eng.main()
    elif page=='Normalization':
        _05_normalise.main()
    elif page=="Encoding Categorical":
        _06_encoding_cat_data.main()
    elif page=="Pipeline":
        _07_pipeline.main()

    elif page=="Functional Transormer":
        _08_functional_transformer.main()
    elif page=="Binning":
        _09_binning.main()

    elif page=="Missing data":
        _10_oultliers.main()
    elif page=="PCA":
        _11_pca.main()




if __name__ == '__main__':
     main()