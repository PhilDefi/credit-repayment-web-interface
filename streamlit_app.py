import streamlit as st
import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from IPython.display import Image, display
import base64
from PIL import Image
import io

st.title('🎈 App Happy Name')

st.write('Hello world!')




# cd documents/python/projets/projet_7
# streamlit run streamlit_interface.py

# Load CSV modified test file with correct variable type
with open('./dtypes_enriched.json', 'r') as f:
    dtypes_loaded = json.load(f)
X_test = pd.read_csv('./test_enriched.csv', dtype=dtypes_loaded)


### HEADER  ###################################################################
st.title("📊 Loan default prediction dashboard")
st.write("""
## Predict the credit repayment probability using a ML model
""")


### BODY 1 ###################################################################
row = st.number_input("Select the credit you want a prediction for :", min_value=0, max_value=100, value=42)


### BODY 2 ###################################################################
# Prepare payload for API request
X_sample = X_test.iloc[[row]].reset_index(drop=True)
X_sample = X_sample.astype(str)
payload = X_sample.to_dict(orient="split")
payload.pop('index', None)

# Heroku server API URL
url = "https://app-heroku-credit-p7-a25edceb2cf8.herokuapp.com/predict_with_explanation"

# POST API request
response = requests.post(url, json=payload)


# Create two tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction Chart", "🔍 Loan Details", "❓ Explanation"])