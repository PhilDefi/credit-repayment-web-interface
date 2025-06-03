import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import matplotlib.pyplot as plt
import json
import numpy as np
from IPython.display import Image, display
import base64
from PIL import Image
import io
import plotly.graph_objects as go
from scipy import stats

# To test locally :
# cd documents/python/projets/misc/credit-repayment-web-interface
# streamlit run app_test_streamlit.py

# Feature list
feat_list = ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'CNT_CHILDREN',
             'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
             'EXT_SOURCE_1_x', 'EXT_SOURCE_2_x', 'EXT_SOURCE_3_x']

# Load train set
df = pd.read_csv('train_reduced.csv')

# Load modified test set with correct variable type
with open('./dtypes_enriched.json', 'r') as f:
    dtypes_loaded = json.load(f)
X_test = pd.read_csv('./test_enriched.csv', dtype=dtypes_loaded)


### STREAMLIT INTERFACE HEADER  ################################################
st.title("📊 Loan default prediction dashboard")
st.write("""
## Predict the credit repayment probability using a ML model
""")


### BODY 1 ###################################################################
row = st.number_input("Select the credit you want a prediction for :", min_value=0, max_value=200, value=101)


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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Chart", "🔍 Loan Details", "❓ Explanation", "🎯 Benchmark Analysis"])

# --- Tab 1: Prediction Pie Chart ---
with tab1:
    st.header("Prediction")
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

        # WCAG: Provide text alternative for screen readers
        st.markdown("### 🔍 Visual Description (Screen Reader Alternative)")
        st.write(f"""
        **Loan Default Prediction Results :**
        - Probability of No Default: {(1-probs)*100:.1f}%
        - Probability of Default: {probs*100:.1f}%
        
        The chart shows a pie diagram with two sections representing the likelihood of loan default.
        """)
        
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
    

# --- Tab 2: Loan information ---
with tab2:
    st.header("Loan information")
    sex = "male" if X_test.loc[row, 'CODE_GENDER_M'] else "female"
    age = X_test.loc[row, 'DAYS_BIRTH'] / 365.25    
    st.write(f"Loan applicant : {sex}, {age:.0f} years old")    
    st.write(f"Loan id : {X_test.loc[row, 'SK_ID_CURR']}")
    st.write(f"Credit amount : {X_test.loc[row, 'AMT_CREDIT']:,.0f}$")
    st.write(f"Annuity : {X_test.loc[row, 'AMT_ANNUITY']:,.0f}$")
    st.write(f"Good price amount : {X_test.loc[row, 'AMT_GOODS_PRICE']:,.0f}$")
    st.write(f"Total income : {X_test.loc[row, 'AMT_INCOME_TOTAL']:,.0f}$")    


# --- Tab 3: SHAP interpretation ---
with tab3:
    st.header("Explanation")
    st.info("This section provides SHAP interpretability to explain the model's prediction for the submitted loan application.")
    if response.status_code == 200:
        shap_img_base64 = response.json()["shap_waterfall_plot"]
        img_bytes = base64.b64decode(shap_img_base64)
        shap_image = Image.open(io.BytesIO(img_bytes))
        st.image(shap_image, caption="SHAP Waterfall Explanation")

        # WCAG: Detailed text alternative for screen readers
        st.markdown("### 🔍 Visual Description (Screen Reader Alternative)")
        st.markdown("""
        **SHAP Waterfall Plot Description:**
        
        This chart shows how different features of the loan application contribute to the final prediction. 
        The waterfall plot displays:
        
        - **Baseline**: The average prediction across all loans in the dataset
        - **Feature contributions**: Each feature either pushes the prediction toward "Default" (positive values, typically red) 
          or toward "No Default" (negative values, typically blue)
        - **Final prediction**: The cumulative result after all feature contributions
        """)


# --- Tab 4: Feature distribution with loan under analysis highlight and feature market stats ---
with tab4:
    st.header("Individual vs Market Analysis")    
    # Dropdown menu
    selected_feature = st.selectbox("Choose a feature:", feat_list)    
    # Display distribution
    st.subheader(f"Distribution of {selected_feature}")    
    # Get the specific loan value
    loan_value = X_test[selected_feature][row]

    if np.isnan(loan_value) == False: # to avoid bug in cas of missing value

        ### FIRST GRAPH ###
        st.subheader(f"Individual feature value : {loan_value:.2f}")
        # Create histogram        
        fig = px.histogram(df, x=selected_feature)    
        # Add vertical line for the specific loan
        fig.add_vline(x=loan_value, line_dash="dash", line_color="red", 
                      annotation_text=f"Analyzed loan: {loan_value:.2f}")    
        st.plotly_chart(fig)
        # WCAG: Simple alternative text description
        st.markdown("### 🔍 Visual Description (Screen Reader Alternative)")        
        st.markdown(f"""
        **Chart Description:** Histogram showing the distribution of {selected_feature} across all loans in the dataset. 
        A red dashed vertical line indicates the analyzed loan's value of {loan_value:.2f} within this distribution.
""")

        ### SECOND GRAPH ###
        # Distribution by TARGET
        st.subheader(f"Distribution of {selected_feature} by TARGET")
        # Separate data by TARGET
        target_0_data = df[df['TARGET'] == 0][selected_feature].dropna()
        target_1_data = df[df['TARGET'] == 1][selected_feature].dropna()        
        # Create figure
        fig_target = go.Figure()
        
        # Calculate density for TARGET = 0
        if len(target_0_data) > 0:
            density_0 = stats.gaussian_kde(target_0_data)
            x_range = np.linspace(df[selected_feature].min(), df[selected_feature].max(), 200)
            y_0 = density_0(x_range)
            
            fig_target.add_trace(go.Scatter(
                x=x_range, y=y_0,
                mode='lines',
                name='No Default (0)',
                line=dict(color='blue', width=2),
                fill='tonexty' if len(target_1_data) == 0 else None
            ))
        
        # Calculate density for TARGET = 1
        if len(target_1_data) > 0:
            density_1 = stats.gaussian_kde(target_1_data)
            x_range = np.linspace(df[selected_feature].min(), df[selected_feature].max(), 200)
            y_1 = density_1(x_range)
            
            fig_target.add_trace(go.Scatter(
                x=x_range, y=y_1,
                mode='lines',
                name='Default (1)',
                line=dict(color='orange', width=2)
            ))
        
        # Add vertical line for the specific loan
        fig_target.add_vline(x=loan_value, line_dash="dash", line_color="red",
                           annotation_text=f"Analyzed loan: {loan_value:.2f}")
        
        # Update layout
        fig_target.update_layout(
            title=f"Density Distribution of {selected_feature} by TARGET",
            xaxis_title=selected_feature,
            yaxis_title="Density",
            legend=dict(title="Default Status", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_target)
        # WCAG: Simple alternative text description
        st.markdown("### 🔍 Visual Description (Screen Reader Alternative)")        
        st.markdown(f"""
        **Chart Description:** Density graph showing the distribution of the selected feature ({selected_feature}) for clients grouped by loan repayment status:
        - Blue curve represents clients **without default (TARGET = 0)**
        - Orange curve represents clients **with default (TARGET = 1)**

        The height of each curve reflects how frequently that feature value occurs in each group. 
        A red dashed vertical line indicates the analyzed loan's value of {loan_value:.2f} within this distribution.
""")        

        ### THIRD GRAPH ###
        st.subheader(f"Advanced market statistics - {selected_feature}")
        # Calculate key statistics
        feature_mean = df[selected_feature].mean()
        feature_median = df[selected_feature].median()
        feature_std = df[selected_feature].std()
        feature_q25 = df[selected_feature].quantile(0.25)
        feature_q75 = df[selected_feature].quantile(0.75)
        
        # Contextual information in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset Mean", f"{feature_mean:.2f}", 
                      delta=f"{loan_value - feature_mean:.2f}")
        
        with col2:
            st.metric("Dataset Median", f"{feature_median:.2f}",
                      delta=f"{loan_value - feature_median:.2f}")
        
        with col3:
            # Calculate percentile of the loan value
            percentile = (df[selected_feature] < loan_value).mean() * 100
            st.metric("Loan Percentile", f"{percentile:.1f}%")
        
        with col4:
            # Z-score (how many standard deviations from mean)
            z_score = (loan_value - feature_mean) / feature_std if feature_std > 0 else 0
            st.metric("Z-Score", f"{z_score:.2f}")
        
        # Additional context box
        with st.expander("📋 Feature Market Statistics Summary"):
            st.write(f"""
            **Basic Statistics : {selected_feature}**
            - Count: {df[selected_feature].count():,}
            - Missing: {df[selected_feature].isna().sum():,} ({df[selected_feature].isna().mean()*100:.1f}%)
            - Min: {df[selected_feature].min():.2f}
            - Max: {df[selected_feature].max():.2f}
            - Range: {df[selected_feature].max() - df[selected_feature].min():.2f}
            - Standard Deviation: {feature_std:.2f}
            - Coefficient of Variation: {(feature_std/feature_mean)*100 if feature_mean != 0 else 0:.1f}%
            
            **Quartiles:**
            - Q1 (25%): {feature_q25:.2f}
            - Q2 (50% - Median): {feature_median:.2f}
            - Q3 (75%): {feature_q75:.2f}
            - IQR: {feature_q75 - feature_q25:.2f}
            """)

    else:
        st.write("No data") 
