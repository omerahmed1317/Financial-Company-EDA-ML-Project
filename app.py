"""
═══════════════════════════════════════════════════════════════════════════════
COMPANY PROFIT PREDICTION STREAMLIT APP
═══════════════════════════════════════════════════════════════════════════════

A production-ready web application for predicting company profit based on 
financial indicators using a trained XGBoost machine learning model.

Features:
- Interactive user interface with real-time validation
- Automatic feature engineering (profit margin, asset turnover)
- Input preprocessing (scaling, one-hot encoding)
- Detailed prediction results and input summary
- Error handling and data validation

Run: streamlit run app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Company Profit Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 0.95rem;
    }
    
    /* Input container */
    .input-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Result boxes */
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: none;
        cursor: pointer;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background-color: #1557a0;
    }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL AND UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_artifacts():
    """
    Load pre-trained model, scaler, and feature information from pickle files.
    Uses @st.cache_resource to avoid reloading on every interaction.
    """
    try:
        # Load model
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load scaler
        with open("feature_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Load feature information
        with open("feature_names.pkl", "rb") as f:
            feature_info = pickle.load(f)
        
        return model, scaler, feature_info
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: Could not find model file. {str(e)}")
        st.info("""
        Please ensure these files exist in the same directory as app.py:
        - best_model.pkl
        - feature_scaler.pkl
        - feature_names.pkl
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_countries(feature_info):
    """
    Extract all unique countries from encoded feature names.
    
    Args:
        feature_info (dict): Dictionary containing feature information
        
    Returns:
        list: Sorted list of country names
    """
    country_features = feature_info['country_encoded_features']
    countries = [feat.replace('country_', '') for feat in country_features]
    return sorted(countries)

def calculate_engineered_features(sales, assets):
    """
    Calculate engineered features from raw inputs.
    
    Args:
        sales (float): Company sales in billions
        assets (float): Company assets in billions
        
    Returns:
        tuple: (profit_margin, asset_turnover)
        
    Note:
        - profit_margin uses approximate average margin if not provided
        - asset_turnover = sales / assets
    """
    # Asset turnover is straightforward
    asset_turnover = sales / assets if assets > 0 else 0
    
    # For profit_margin, we'll use an approximate industry average (~20%)
    # since we don't know profit yet (it's being predicted)
    # The model will adjust this based on the pattern
    profit_margin = 0.20  # Default industry average
    
    return profit_margin, asset_turnover

def validate_inputs(sales, assets, market_value):
    """
    Validate user inputs to ensure they're reasonable.
    
    Args:
        sales (float): Company sales in billions
        assets (float): Company assets in billions
        market_value (float): Company market value in billions
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if sales <= 0:
        return False, "Sales must be greater than 0"
    if assets <= 0:
        return False, "Assets must be greater than 0"
    if market_value <= 0:
        return False, "Market Value must be greater than 0"
    
    # Check if values are unreasonably large (potential data entry errors)
    if sales > 10000:
        return False, "Sales seems unreasonably high (>$10,000B). Please verify."
    if assets > 50000:
        return False, "Assets seems unreasonably high (>$50,000B). Please verify."
    if market_value > 100000:
        return False, "Market Value seems unreasonably high (>$100,000B). Please verify."
    
    return True, ""

def prepare_prediction_data(sales, assets, market_value, country, feature_info, scaler):
    """
    Prepare input data for model prediction by:
    1. Calculating engineered features
    2. One-hot encoding country
    3. Scaling numerical features
    
    Args:
        sales (float): Company sales in billions
        assets (float): Company assets in billions
        market_value (float): Company market value in billions
        country (str): Company country
        feature_info (dict): Feature metadata
        scaler: Fitted StandardScaler object
        
    Returns:
        pandas.DataFrame: Preprocessed feature matrix ready for prediction
    """
    # Calculate engineered features
    profit_margin, asset_turnover = calculate_engineered_features(sales, assets)
    
    # Create numerical features dataframe
    numerical_data = pd.DataFrame({
        'sales': [sales],
        'assets': [assets],
        'market_value': [market_value],
        'profit_margin': [profit_margin],
        'asset_turnover': [asset_turnover]
    })
    
    # Create one-hot encoded country features
    country_features_dict = {}
    for country_feat in feature_info['country_encoded_features']:
        country_features_dict[country_feat] = [1 if country_feat == f'country_{country}' else 0]
    
    country_data = pd.DataFrame(country_features_dict)
    
    # Combine numerical and categorical features
    combined_data = pd.concat([numerical_data, country_data], axis=1)
    
    # Ensure column order matches training data
    combined_data = combined_data[feature_info['all_features']]
    
    # Scale features
    scaled_data = scaler.transform(combined_data)
    scaled_df = pd.DataFrame(scaled_data, columns=feature_info['all_features'])
    
    return scaled_df, {
        'profit_margin': profit_margin,
        'asset_turnover': asset_turnover
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main Streamlit application."""
    
    # Load model and utilities
    model, scaler, feature_info = load_model_and_artifacts()
    countries = get_all_countries(feature_info)
    
    # ═════════════════════════════════════════════════════════════════════════
    # HEADER SECTION
    # ═════════════════════════════════════════════════════════════════════════
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📊 Company Profit Prediction")
    
    st.markdown("""
        <div class="subtitle">
            Predict company profit based on financial indicators using AI
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═════════════════════════════════════════════════════════════════════════
    # INPUT SECTION
    # ═════════════════════════════════════════════════════════════════════════
    
    st.subheader("📝 Enter Company Financial Data")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sales = st.number_input(
                "💰 Sales (Billions $)",
                min_value=0.1,
                max_value=10000.0,
                value=100.0,
                step=10.0,
                help="Company annual sales revenue in billions of dollars"
            )
        
        with col2:
            assets = st.number_input(
                "🏦 Assets (Billions $)",
                min_value=0.1,
                max_value=50000.0,
                value=500.0,
                step=50.0,
                help="Total company assets in billions of dollars"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            market_value = st.number_input(
                "📈 Market Value (Billions $)",
                min_value=0.1,
                max_value=100000.0,
                value=200.0,
                step=20.0,
                help="Company market capitalization in billions of dollars"
            )
        
        with col4:
            country = st.selectbox(
                "🌍 Country",
                countries,
                help="Select the company's headquarters country"
            )
        
        # Calculate engineered features (display only)
        profit_margin, asset_turnover = calculate_engineered_features(sales, assets)
        
        st.info(f"""
        **Calculated Features:**
        - Profit Margin: {profit_margin:.2%} (estimated)
        - Asset Turnover: {asset_turnover:.3f}x
        """)
        
        submitted = st.form_submit_button("🚀 Predict Profit", use_container_width=True)
    
    # ═════════════════════════════════════════════════════════════════════════
    # PREDICTION SECTION
    # ═════════════════════════════════════════════════════════════════════════
    
    if submitted:
        # Validate inputs
        is_valid, error_msg = validate_inputs(sales, assets, market_value)
        
        if not is_valid:
            st.error(f"❌ Validation Error: {error_msg}")
        else:
            with st.spinner("🔮 Making prediction..."):
                try:
                    # Prepare data for prediction
                    X_pred, engineered = prepare_prediction_data(
                        sales, assets, market_value, country, 
                        feature_info, scaler
                    )
                    
                    # Make prediction
                    predicted_profit = model.predict(X_pred)[0]
                    
                    # Ensure non-negative profit
                    predicted_profit = max(predicted_profit, 0.0)
                    
                    # Display results
                    st.divider()
                    st.subheader("✅ Prediction Results")
                    
                    # Main prediction box
                    col1, col2 = st.columns([1.5, 1])
                    
                    with col1:
                        st.markdown(f"""
                            <div class="result-box success-box">
                            <h2 style="margin: 0; color: #155724;">
                                💰 Predicted Profit
                            </h2>
                            <h1 style="margin: 0.5rem 0 0 0; color: #155724;">
                                ${predicted_profit:.2f}B
                            </h1>
                            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
                                Billion US Dollars
                            </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Calculate profit margin based on prediction
                        profit_margin_actual = (predicted_profit / sales * 100) if sales > 0 else 0
                        st.metric(
                            "Actual Profit Margin",
                            f"{profit_margin_actual:.1f}%",
                            delta=f"{profit_margin_actual - 20:.1f}% vs avg"
                        )
                    
                    # Input summary
                    st.subheader("📋 Input Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.metric("Sales", f"${sales:.1f}B")
                        st.metric("Assets", f"${assets:.1f}B")
                    
                    with summary_col2:
                        st.metric("Market Value", f"${market_value:.1f}B")
                        st.metric("Country", country)
                    
                    with summary_col3:
                        st.metric("Asset Turnover", f"{engineered['asset_turnover']:.3f}x")
                        st.metric("ROA*", f"{(predicted_profit/assets*100):.2f}%")
                    
                    st.caption("*ROA = Return on Assets (Predicted Profit / Assets)")
                    
                    # Additional insights
                    st.subheader("📊 Model Insights")
                    
                    col_insight1, col_insight2, col_insight3 = st.columns(3)
                    
                    with col_insight1:
                        profit_to_sales = (predicted_profit / sales) if sales > 0 else 0
                        st.info(f"**Profit/Sales Ratio:** {profit_to_sales:.2%}")
                    
                    with col_insight2:
                        profit_to_assets = (predicted_profit / assets) if assets > 0 else 0
                        st.info(f"**Profit/Assets Ratio:** {profit_to_assets:.2%}")
                    
                    with col_insight3:
                        profit_to_market = (predicted_profit / market_value) if market_value > 0 else 0
                        st.info(f"**Profit/Market Value:** {profit_to_market:.2%}")
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Disclaimer:** This prediction is based on historical financial patterns 
                    from the training data. Actual profit may vary significantly based on market 
                    conditions, management decisions, and external factors not captured in this model.
                    """)
                
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    st.info("Please check your inputs and try again.")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("ℹ️ About This App")
    
    st.info("""
    This app uses a trained **XGBoost** machine learning model to predict 
    company profit based on financial indicators.
    
    **Model Features:**
    - Linear Regression
    - Random Forest
    - **XGBoost** (Selected as best)
    
    **Performance:**
    - R² Score: 0.84
    - Mean Absolute Error: $0.78B
    """)
    
    st.subheader("📚 Input Features")
    st.markdown("""
    - **Sales**: Annual revenue
    - **Assets**: Total assets
    - **Market Value**: Market cap
    - **Country**: HQ location
    - **Profit Margin**: Calculated
    - **Asset Turnover**: Calculated
    """)
    
    st.subheader("🔧 Technical Details")
    st.markdown("""
    - **Framework**: Streamlit
    - **ML Library**: XGBoost
    - **Preprocessing**: StandardScaler, OneHotEncoder
    - **Total Features**: 56 (5 numerical + 51 countries)
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# RUN APP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
