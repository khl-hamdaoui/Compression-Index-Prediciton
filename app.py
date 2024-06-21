import streamlit as st
import pandas as pd
import pickle

# Load the single model
model = pickle.load('catboost_cc.pkl')

# Set Streamlit page configuration
st.set_page_config(page_title='Compression Index Predictor', layout='wide')

# Title and description
st.title('Compression Index Predictor')
st.write("Input your own data for prediction.")

# Sidebar for user input features
st.sidebar.header('User Input Features')

def user_input_features():
    LL = st.sidebar.number_input('Liquid Limit (LL %)', value=50.0)
    PL = st.sidebar.number_input('Plastic Limit (PL %)', value=30.0)
    e0 = st.sidebar.number_input('Initial Void Ratio (e0)', value=0.5)
    w = st.sidebar.number_input('Water Content (w %)', value=20.0)
    
    data = {
        'LL': LL,
        'PL': PL,
        'e0': e0,
        'w': w
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Main header and input data
st.header('Prediction Input and Results')
input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)
    
    st.subheader('Prediction Results')
    st.write(f'Compression Index (Cc): **{prediction[0]:.2f}**')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)
