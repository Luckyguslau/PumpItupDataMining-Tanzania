import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import base64
from PIL import Image

# Set page config must be first command
st.set_page_config(
    layout="wide", 
    page_title="Water Pump Status Prediction in Tanzania",
    page_icon="🚰",
    initial_sidebar_state="collapsed"
)

# Fungsi untuk mengencode gambar ke base64 dengan error handling
def get_base64_of_bin_file(bin_file):
    try:
        possible_paths = [
            bin_file,
            os.path.join(os.path.dirname(__file__), bin_file),
            os.path.join(os.getcwd(), bin_file)
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                return base64.b64encode(data).decode()
        
        st.warning(f"Background image {bin_file} not found. Using default background.")
        return ""
        
    except Exception as e:
        st.warning(f"Could not load background image: {str(e)}")
        return ""

# Set paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'notebooks/random_forest_model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'notebooks/preprocessor.pkl')
PHOTOS_PATH = os.path.join(os.path.dirname(__file__), 'photos/')
WATER_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'WaterTanzania.png')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/training set values.csv')

# Encode background image
water_bg_base64 = get_base64_of_bin_file(WATER_IMAGE_PATH)

# Load model and preprocessor
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        
        required_keys = ['used_columns', 'label_mappings', 'scaler', 'numerical_cols', 'categorical_cols']
        if not all(key in preprocessor for key in required_keys):
            st.error("Invalid preprocessor structure!")
            return None, None
            
        return model, preprocessor
    except Exception as e:
        st.error(f"Failed to load model or preprocessor: {str(e)}")
        return None, None

model, preprocessor = load_model()

if model is None or preprocessor is None:
    st.stop()

# Load region mapping from training data
@st.cache_data
def load_region_mapping():
    try:
        # Read the training data
        training_data = pd.read_csv(TRAINING_DATA_PATH)
        
        # Create mapping from region to region_code
        region_mapping = {}
        
        # Get unique region-region_code pairs
        unique_pairs = training_data[['region', 'region_code']].drop_duplicates()
        
        for _, row in unique_pairs.iterrows():
            region = row['region']
            region_code = int(row['region_code'])
            region_mapping[region] = region_code
        
        # Create reverse mapping
        reverse_mapping = {}
        for region, code in region_mapping.items():
            reverse_mapping[code] = region
        
        return region_mapping, reverse_mapping
        
    except Exception as e:
        st.error(f"Could not load training data from {TRAINING_DATA_PATH}: {str(e)}")
        # Return empty mappings as fallback
        return {}, {}

# Get region mappings
region_mapping, reverse_region_mapping = load_region_mapping()

# Function to preprocess user input
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    
    # Ensure all required columns exist
    for col in preprocessor['used_columns']:
        if col not in df.columns:
            default_value = -1 if col in preprocessor['categorical_cols'] else 0.0
            df[col] = default_value
    
    # Label encoding for categorical features
    for col in preprocessor['categorical_cols']:
        if col in df.columns:
            mapping = preprocessor['label_mappings'].get(col, {})
            df[col] = df[col].apply(lambda x: mapping.get(x, -1))
    
    # Scaling for numerical features
    if preprocessor['numerical_cols']:
        df[preprocessor['numerical_cols']] = preprocessor['scaler'].transform(df[preprocessor['numerical_cols']])
    
    # Order columns as used during training
    df = df[preprocessor['used_columns']]
    
    return df

# Custom CSS
background_style = f"""
    background: linear-gradient(rgba(0, 0, 0, 0.50), rgba(0, 0, 0, 0.80)),
                url("data:image/jpg;base64,{water_bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
""" if water_bg_base64 else "background: #101010;"

st.markdown(f"""
<style>
    .stApp {{
        {background_style}
    }}
    .main {{
        background-color: transparent;
        color: #f0f0f0;
    }}
    h1 {{
        color: #f0f0f0;
        font-size: 3rem;
        font-weight: 300;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #ffcc00;
        padding-bottom: 1rem;
    }}
    .streamlit-expanderHeader {{
        background-color: rgba(26, 26, 26, 0.9);
        color: #f0f0f0;
        font-size: 1.1rem;
        font-weight: 400;
        padding: 1rem;
        border: none;
        border-radius: 4px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }}
    .streamlit-expanderHeader:hover {{
        background-color: rgba(37, 37, 37, 0.9);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .streamlit-expanderContent {{
        background-color: rgba(26, 26, 26, 0.9);
        border: none;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }}
    .stButton > button {{
        background-color: #ffcc00;
        color: #101010;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        border-radius: 4px;
        width: 100%;
        transition: all 0.3s ease;
        margin: 2rem 0;
    }}
    .stButton > button:hover {{
        background-color: #ffd633;
        box-shadow: 0 8px 25px rgba(255, 204, 0, 0.3);
        transform: translateY(-2px);
    }}
    .prediction-result {{
        background-color: rgba(26, 26, 26, 0.9);
        padding: 3rem;
        border-radius: 8px;
        text-align: center;
        margin: 2rem 0;
        opacity: 0;
        transform: scale(0.95);
        animation: scaleUp 0.6s ease-out 0.1s forwards;
        border-left: 4px solid #ffcc00;
        backdrop-filter: blur(10px);
    }}
    @keyframes scaleUp {{
        from {{ opacity: 0; transform: scale(0.95); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.header("Model Information")
    
    with st.expander("MODEL ARCHITECTURE", expanded=False):
        st.markdown(f"""
        <div style='color: #f0f0f0; font-size: 0.9rem; line-height: 1.6;'>
        <p><strong>Features Used:</strong> {len(preprocessor['used_columns'])}</p>
        <p><strong>Categorical Features:</strong> {len(preprocessor['categorical_cols'])}</p>
        <p><strong>Numerical Features:</strong> {len(preprocessor['numerical_cols'])}</p>
        </div>
        """, unsafe_allow_html=True)

# Main app content
st.title("Water Pump Status Prediction in Tanzania")

st.markdown("""
<div style='color: #888; font-size: 1.1rem; line-height: 1.6; margin-bottom: 3rem;'>
System analysis for water pump operational status assessment using optimized feature selection
</div>
""", unsafe_allow_html=True)

# Initialize session state for region and region_code
if 'region' not in st.session_state:
    # Set default to first available region
    available_regions = [r for r in preprocessor['label_mappings']['region'].keys() if r in region_mapping]
    st.session_state.region = available_regions[0] if available_regions else list(preprocessor['label_mappings']['region'].keys())[0]

if 'region_code' not in st.session_state:
    # Set default region_code based on default region
    st.session_state.region_code = region_mapping.get(st.session_state.region, 1)

# Input form
user_input = {}

# Define slider ranges
slider_ranges = {
    'amount_tsh': (0, 350000, 170000),
    'gps_height': (-90, 2650, 1300),
    'construction_year': (1990, 2013, 2002),
    'region_code': (1, 99, 50)
}

# Feature groups
feature_groups = {
    "WATER CHARACTERISTICS": ['quantity', 'quantity_group', 'source'],
    "LOCATION INFORMATION": ['lga', 'region_code', 'region', 'basin'],
    "TECHNICAL SPECIFICATIONS": ['construction_year', 'extraction_type_class', 'gps_height', 'amount_tsh'],
    "INSTALLATION DETAILS": ['installer', 'payment_type'],
    "PUMP CONFIGURATION": ['waterpoint_type', 'waterpoint_type_group']
}

# Create a form to prevent immediate updates
with st.form("prediction_form"):
    # Create input fields organized by categories
    for group_name, features in feature_groups.items():
        with st.expander(group_name, expanded=False):
            cols = st.columns(3)
            col_idx = 0
            
            for col in features:
                if col in preprocessor['used_columns']:
                    with cols[col_idx % 3]:
                        if col in preprocessor['categorical_cols']:
                            options = list(preprocessor['label_mappings'][col].keys())
                            
                            # Special handling for region and region_code synchronization
                            if col == 'region':
                                # Filter available regions to only those in our mapping
                                available_regions = [r for r in options if r in region_mapping]
                                
                                selected_region = st.selectbox(
                                    label=f"{col.replace('_', ' ').title()}",
                                    options=available_regions,
                                    index=available_regions.index(st.session_state.region) if st.session_state.region in available_regions else 0,
                                    key=f"{col}_select"
                                )
                                
                                user_input[col] = selected_region
                                
                            elif col == 'region_code':
                                # Get all available region codes from our mapping
                                available_codes = list(reverse_region_mapping.keys())
                                min_code = min(available_codes) if available_codes else 1
                                max_code = max(available_codes) if available_codes else 99
                                
                                selected_region_code = st.slider(
                                    label=f"{col.replace('_', ' ').title()}",
                                    min_value=min_code,
                                    max_value=max_code,
                                    value=st.session_state.region_code,
                                    key=f"{col}_slider"
                                )
                                
                                user_input[col] = selected_region_code
                                
                            else:
                                user_input[col] = st.selectbox(
                                    label=f"{col.replace('_', ' ').title()}",
                                    options=options,
                                    index=0,
                                    key=col
                                )
                        else:
                            if col in slider_ranges:
                                min_val, max_val, default_val = slider_ranges[col]
                                user_input[col] = st.slider(
                                    label=f"{col.replace('_', ' ').title()}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=default_val,
                                    key=col
                                )
                            else:
                                user_input[col] = st.number_input(
                                    label=f"{col.replace('_', ' ').title()}",
                                    value=0.0,
                                    key=col
                                )
                    col_idx += 1

    # Submit button
    submitted = st.form_submit_button("ANALYZE PUMP STATUS", type="primary", use_container_width=True)

# Handle synchronization after form submission
if submitted:
    # Update session state based on user input
    if 'region' in user_input and user_input['region'] != st.session_state.region:
        st.session_state.region = user_input['region']
        if st.session_state.region in region_mapping:
            st.session_state.region_code = region_mapping[st.session_state.region]
    
    if 'region_code' in user_input and user_input['region_code'] != st.session_state.region_code:
        st.session_state.region_code = user_input['region_code']
        if st.session_state.region_code in reverse_region_mapping:
            st.session_state.region = reverse_region_mapping[st.session_state.region_code]

# Display current synchronization status
st.markdown(f"""
<div style='background-color: rgba(26, 26, 26, 0.9); padding: 1rem; border-radius: 4px; margin: 1rem 0; border-left: 4px solid #ffcc00;'>
    <div style='color: #888; font-size: 0.9rem;'>Current Selection:</div>
    <div style='color: #ffcc00; font-size: 1.1rem;'>Region: <strong>{st.session_state.region}</strong> | Region Code: <strong>{st.session_state.region_code}</strong></div>
</div>
""", unsafe_allow_html=True)

# Handle prediction after form submission
if submitted:
    try:
        # Ensure region and region_code are synchronized in user_input
        user_input['region'] = st.session_state.region
        user_input['region_code'] = st.session_state.region_code
        
        # Preprocess input
        X_input = preprocess_input(user_input)
        
        # Prediction
        pred = model.predict(X_input)[0]
        
        # Map prediction to label
        status_map = {
            0: ("Non Functional", "#ff6b6b"),
            1: ("Functional", "#51cf66"),
            2: ("Functional Needs Repair", "#ffd43b")
        }
        
        status_text, status_color = status_map.get(pred, ("Unknown", "#74c0fc"))
        
        # Display result
        st.markdown(
            f"""
            <div class='prediction-result'>
                <div class='prediction-title'>SYSTEM ASSESSMENT</div>
                <div class='prediction-value' style='color: {status_color};'>{status_text}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Show confidence scores
        proba = model.predict_proba(X_input)[0]
        st.markdown("<div style='margin: 2rem 0 1rem 0; color: #888; font-size: 1rem;'>PREDICTION CONFIDENCE</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Functional", f"{proba[1]:.1%}")
        with col2:
            st.metric("Non Functional", f"{proba[0]:.1%}")
        with col3:
            st.metric("Needs Repair", f"{proba[2]:.1%}")
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        st.error("Please verify all input parameters are correctly configured.")
