import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def main():
    import numpy as np
    st.title('What are Tensors?')
    st.write('Basically Tensors are data structure, nothing much but it stores numbers rarely, you might have noticed alphabets or symbols because in deep learning machine learning you have to encode or vectorize them in numbers, dont worry I will make it intersting.',)

    st.write("So in back early 19's basically there was a problem with physicist and mathematician of considering n-dimensions so they called it as tensor it includes **scalar(0-D), vectors(1-D), matrices(3-D)** and ...")

    st.warning("Don't confuse between arrays and tensors. Arrays and tensors are both data structures used to represent and manipulate collections of numbers, but they differ in their dimensionality, usage, and certain functionalities. **You can call n-D array as tensors**.")

    st.write('## Types of tensors')
    st.write('Lets discuss types of tensors with visuals')
    data = {
        'CGPA': [8.0, 8.5, 6.9, 9.2],
        'IQ': [120, 122, 140, 136],
        'STATE': [0, 1, 1, 0]

    }

    df = pd.DataFrame(data, index=[1, 2, 3, 4])
    st.dataframe(df)
    one_d = df.loc[1]

    st.write("Consider above data, of **CGPA, IQ, STATES, here in states, columns 0 represent BIHAR and 1 represent as GOA, this technique is called one-hot encoding** where we have converted alphabets into numbers, it's easy for machine to learn number.")

    st.markdown(f'So back to tensors again, lets imagine we have 1000 students data with student 1: CGPA, IQ, STATE, student 2: CGPA, IQ, STATE and respectively.')

    st.dataframe(one_d)

    st.markdown('Above data is of student one and this is called **1-D TENSOR**, which consisting 3 vectors (cgps, iq, state), ok! if you select column of any CGPA or IQ or STATE its also a **1-D TENSOR**')
    st.warning(
        'Dont confuse between vectors and tensors, you will get clear view later in this page')
    iq = df.iloc[:, 1]
    state = df.iloc[:, 2]
    cgpa = df.iloc[:, 0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.dataframe(cgpa)
    with col2:
        st.dataframe(iq)
    with col3:
        st.dataframe(state)

    st.markdown(
        'Above examples are 1-D tensors too! with 3 vector of cgpa, iq, state')

    dataframe = pd.DataFrame(data)

    # Extract the first row
    row = dataframe.iloc[0]

    # Prepare data for 3D plot
    x = row['CGPA']
    y = row['IQ']
    z = row['STATE']

    fig = go.Figure(data=[go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode='markers+lines',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue', width=2)
    )])

    # Set axis labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='CGPA',
            yaxis_title='IQ',
            zaxis_title='STATE'
        ),
        title='3D Line Plot of First Row '
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write("#### What is 2-D Tensor?")
    st.write("""simply combining 2, 1-D tensors is 2-D tensors i.e data of student one and student 2, here we will add one more topic axis, A 2D tensor is essentially a matrix, and it has two axes:
    Axis 0 (Rows): This axis represents the rows of the matrix. In a tensor.
    Axis 1 (Columns): This axis represents the columns of the matrix. Here is an example below.""")

    two_d = df.loc[0:2]

    st.dataframe(two_d)

    st.latex('X = [[8, 120, 0], [8.5, 122, 1]]')
    st.write(
        'So basically this final matrix form is mathematical example of 2-D tensors')

    st.write("#### What is 3-D Tensors?")
    st.write('3-D tensors are combinations of 3, 2-D tesnors, For example in Natural language processing (NLP), we use to vectorize texts to numeric encoding called as vectorization.')

    three_d = {'Words': ['Hi Ram', 'Hi Mohan', 'Hi Dev']}
    series = pd.DataFrame(three_d, index=[1, 2, 3])

    mat_Data = {'Hi': [1, 0, 0, 0], 'Ram': [0, 1, 0, 0],
                'Mohan': [0, 0, 1, 0], 'Dev': [0, 0, 0, 1]}

    one_hot = pd.DataFrame(mat_Data, index=['Hi', 'Ram', 'Mohan', 'Dev'])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.dataframe(series)

    with col2:
        st.markdown("<h1 style='font-size:50px;'>&#8594;</h1>",
                    unsafe_allow_html=True)

    with col3:
        st.dataframe(mat_Data)

    st.markdown('So if you want to write how **Hi Ram** looks like see below:')
    st.latex("Hi  Ram =  [[1, 0, 0, 0] , [0, 1, 0, 0]]")
    st.latex("Hi  Mohan =  [[1, 0, 0, 0], [0, 0, 1, 0]]")
    st.latex("Hi  Dev =  [[1, 0, 0, 0], [0, 0, 0, 1]]")
    st.write('Here indivisual index are converted into 2-D tensors, if we collect them all and represent together, so now its called 3-D tensors with shape of ')
    st.latex(
        '3-D \ = \ [[[1, 0, 0, 0], [0, 1, 0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0 ], [0, 0, 0, 1]] ]')
    st.latex("Shape \  is \  = \ (3, 2, 3)")
    st.latex("Left \ 3:\  represent \ dimension \ of \  tensor \ i.e \ 3-D ")
    st.latex("Middle \ 2: \ represent\  number \ of \ matrix \  in \ each\  row ")
    st.latex(
        "Right \ 3: \ represent \ Number \ of \  vectors \ inside \ the\  each \ matrix")

    st.write("#### What is 4-D tensors looks like? ")
    st.write("In image processing there are pixels filled with channel of RGB, ",
             "consider pixel of image is 1200x800 of 3 channel and you have data of 3 samples of image")

    st.latex("(3(sample), 3(channels), 1200(height), 800(width))")
    st.image("images/4d tensor.png")
    st.write(
        'Here there are 3 sample of image, with 3 channel, and high and width as follows.')

    st.write("### How 5-D tensors looks like?")
    st.write("We will consider here example of videos, this is typically complex data structure, which is nothing but optical illusion of images against human limit, Persistence is an optical phenomenon in which the brain interprets numerous stationary images as one to generate the appearance of motion. The vision of a normal human eye lasts 1/16 second. here comes frames per second (fps) ")

    st.write('Imagine you have 4 videos, of 30FPS for 30sec, of 480 height 720 width with 3 channel (RGB), so the 5-D tensor shape will looks like this below')

    st.latex("(4(videos), (30fps* 60sec), 480(height), 720(width), 3(RGB))")

    st.write("**Storage require to store above data is 4x1800x480x720x3 in float32 is 238878720000 bits, 27 in GB , but we use encoders like mkv, mp4 which reduces file size**")

    st.write("### Here is code example for 1-D to 5-D tensors")
    st.write('#### 1-D tensor')
    code_1d = """
    import numpy as np
    tensor_1d = np.array([1, 2, 3, 4, 5])
    print(tensor_1d)
"""
    st.code(code_1d, language='python')
    st.write('###### output')
    st.code('[1 2 3 4 5]', language='python')

    st.write('#### 2-D tensor')
    code_2d = """
    tensor_2d = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    print(tensor_2d)
"""
    st.code(code_2d, language='python')
    st.write('###### output')
    st.code('[[1, 2, 3],[4, 5, 6],[7, 8, 9]]', language='python')

    st.write('#### 3-D tensor')
    code_3d = """
    tensor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                      [[1, 3, 5], [2, 4, 6], [7, 9, 8]]])

    print(tensor_3d)
    print(tensor_3d.shape)
"""
    st.code(code_3d, language='python')
    st.write('###### output')
    st.code("""[[[1 2 3]
  [4 5 6]
  [7 8 9]]

 [[9 8 7]
  [6 5 4]
  [3 2 1]]

 [[1 3 5]
  [2 4 6]
  [7 9 8]]]
            
shape (3, 3, 3)""", language='python')

    tensor_3d = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                          [[1, 3, 5], [2, 4, 6], [7, 9, 8]]])

    # Extract coordinates (x, y, z) from the 3D tensor
    x = tensor_3d[0].flatten()
    y = tensor_3d[1].flatten()
    z = tensor_3d[2].flatten()

    # Create 3D scatter plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.8
        )
    )])

    # Update layout for better visualization
    fig.update_layout(scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'),
        width=700,
        margin=dict(r=10, l=10, b=10, t=10))

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write('#### 4-D tensor')
    code_4d = """
 #Create a 4D tensor of shape (2, 3, 4, 5)
tensor_4d = np.random.rand(2, 3, 4, 5)

# Display the shape of the tensor
print("Shape of the tensor:", tensor_4d.shape)

# Display the 4D tensor
print(tensor_4d)


"""
    st.code(code_4d, language='python')

    st.write('#### 5-D tensor')
    code_5d = """

# Create a 5D tensor of shape (2, 3, 4, 5, 6)
tensor_5d = np.random.rand(2, 3, 4, 5, 6)

# Display the shape of the tensor
print("Shape of the tensor:", tensor_5d.shape)

# Display the 5D tensor
print(tensor_5d)


"""

