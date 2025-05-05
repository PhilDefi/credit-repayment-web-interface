import streamlit as st
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
#from IPython.display import Image, display
#import base64
#from PIL import Image
#import io

st.title('🎈 App App Name')

st.write('Hello world!')




# cd documents/python/projets/projet_7
# streamlit run streamlit_interface.py

# Load CSV modified test file with correct variable type
# with open('./dtypes_enriched.json', 'r') as f:
#     dtypes_loaded = json.load(f)
# X_test = pd.read_csv('./test_enriched.csv', dtype=dtypes_loaded)


### HEADER  ###################################################################
st.title("📊 Loan default prediction dashboard")
st.write("""
## Predict the credit repayment probability using a ML model
""")


### BODY 1 ###################################################################
row = st.number_input("Select the credit you want a prediction for :", min_value=0, max_value=100, value=42)

