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


# Create three tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction Chart", "🔍 Loan Details", "❓ Explanation"])

# --- Tab 1: Pie Chart ---
with tab1:
    # Result from API
    if response.status_code == 200:
        probs = response.json()["probability_default"]
        print("Predictions:", probs)    
        
        # Pie chart
        fig, ax = plt.subplots(figsize=(6, 6))      
        labels = ['No Default', 'Default']        
        colors = ['green', 'red']
        ax.pie([1-probs, probs], labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax.set_title("Loan Default Probability")
        ax.axis('equal')  # cercle parfait
        
        # Display in Streamlit app
        st.pyplot(fig)
        
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
    

# --- Tab 2: Loan information ---
with tab2:
    st.write(f"Loan id : {X_test.loc[row, 'SK_ID_CURR']}")
    st.write(f"Credit amount : {X_test.loc[row, 'AMT_CREDIT']:,.0f}$")
    st.write(f"Annuity : {X_test.loc[row, 'AMT_ANNUITY']:,.0f}$")
    st.write(f"Good price amount : {X_test.loc[row, 'AMT_GOODS_PRICE']:,.0f}$")
    st.write(f"Total income : {X_test.loc[row, 'AMT_INCOME_TOTAL']:,.0f}$")    


# --- Tab 3: SHAP interpretation ---
with tab3:
    st.subheader("Explanation")
    st.info("This section provides SHAP interpretability to explain the model's prediction for the submitted loan application.")
    if response.status_code == 200:
        shap_img_base64 = response.json()["shap_waterfall_plot"]
        img_bytes = base64.b64decode(shap_img_base64)
        shap_image = Image.open(io.BytesIO(img_bytes))
        st.image(shap_image, caption="SHAP Waterfall Explanation")
