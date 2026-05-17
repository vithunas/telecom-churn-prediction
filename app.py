import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Page configuration
st.set_page_config(page_title="Telco Churn Prediction", layout="wide", page_icon="📡")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return df

st.title("📡 Telecom Customer Churn Prediction")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This app predicts telecom customer churn using an XGBoost model.")

try:
    df = load_data()
    pipeline = load_model()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    models_loaded = False

if models_loaded:
    tab1, tab2, tab3 = st.tabs(["📊 EDA Dashboard", "🔮 Prediction Form", "📈 Model Evaluation"])

    with tab1:
        st.header("Exploratory Data Analysis")
        
        # Standard Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", len(df))
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        c2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        avg_tenure = df['tenure'].mean()
        c3.metric("Average Tenure", f"{avg_tenure:.1f} mo")
        
        st.write("") # Spacing
        
        col1, col2 = st.columns(2)
        with col1:
            fig_churn = px.pie(df, names='Churn', title='Churn Distribution')
            st.plotly_chart(fig_churn, use_container_width=True)
            
            fig_tenure = px.histogram(df, x='tenure', color='Churn', barmode='group', title='Tenure Distribution by Churn')
            st.plotly_chart(fig_tenure, use_container_width=True)
            
        with col2:
            fig_charges = px.box(df, x='Churn', y='MonthlyCharges', color='Churn', title='Monthly Charges by Churn')
            st.plotly_chart(fig_charges, use_container_width=True)
            
            contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
            fig_contract = px.bar(contract_churn, x='Contract', y='Count', color='Churn', barmode='group', title='Contract Type vs Churn')
            st.plotly_chart(fig_contract, use_container_width=True)

    with tab2:
        st.header("Churn Prediction Form")
        
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 1)
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
                monthly_charges = st.number_input("Monthly Charges", value=50.0)
                
            with col_b:
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                total_charges = st.number_input("Total Charges", value=50.0)
                
            submitted = st.form_submit_button("Predict Churn")

        if submitted:
            input_data = pd.DataFrame({
                "gender": [gender], "SeniorCitizen": [senior_citizen], "Partner": [partner],
                "Dependents": [dependents], "tenure": [tenure], "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines], "InternetService": [internet_service],
                "OnlineSecurity": [online_security], "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection], "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv], "StreamingMovies": [streaming_movies],
                "Contract": [contract], "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method], "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges]
            })
            
            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"High Risk of Churn! (Probability: {probability:.1%})")
            else:
                st.success(f"Likely to Retain. (Probability of churning: {probability:.1%})")
                
            # Simple gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': "Churn Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "red" if prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab3:
        st.header("Feature Importance")
        
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        importance = classifier.feature_importances_
        
        categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                            "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                            "PaperlessBilling", "PaymentMethod"]
        ohe = preprocessor.named_transformers_['cat']
        cat_features = ohe.get_feature_names_out(categorical_cols)
        num_features = np.array(["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"])
        all_features = np.concatenate([num_features, cat_features])
        
        fi_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False).head(15)
        
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Top 15 Predictive Features')
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)
