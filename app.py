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
        # Coba beberapa path yang mungkin
        possible_paths = [
            bin_file,  # Path relatif
            os.path.join(os.path.dirname(__file__), bin_file),  # Path absolute dari app.py
            os.path.join(os.getcwd(), bin_file)  # Path dari working directory
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                return base64.b64encode(data).decode()
        
        # Jika file tidak ditemukan, return empty string
        st.warning(f"Background image {bin_file} not found. Using default background.")
        return ""
        
    except Exception as e:
        st.warning(f"Could not load background image: {str(e)}")
        return ""

# Set paths for model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'notebooks/random_forest_model.pkl')
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'notebooks/preprocessor.pkl')
PHOTOS_PATH = os.path.join(os.path.dirname(__file__), 'photos/')
WATER_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'WaterTanzania.png')

# Encode background image
water_bg_base64 = get_base64_of_bin_file(WATER_IMAGE_PATH)

# Load model and preprocessor
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Validate preprocessor structure
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

# Function to preprocess user input
def preprocess_input(user_input):
    # Create DataFrame from user input
    df = pd.DataFrame([user_input])
    
    # Ensure all required columns exist
    for col in preprocessor['used_columns']:
        if col not in df.columns:
            # Fill missing columns with default values
            default_value = -1 if col in preprocessor['categorical_cols'] else 0.0
            df[col] = default_value
    
    # Label encoding for categorical features
    for col in preprocessor['categorical_cols']:
        if col in df.columns:
            mapping = preprocessor['label_mappings'].get(col, {})
            # If value not in mapping, use default (unknown)
            df[col] = df[col].apply(lambda x: mapping.get(x, -1))
    
    # Scaling for numerical features
    if preprocessor['numerical_cols']:
        df[preprocessor['numerical_cols']] = preprocessor['scaler'].transform(df[preprocessor['numerical_cols']])
    
    # Order columns as used during training
    df = df[preprocessor['used_columns']]
    
    return df

# Custom CSS for ultra-minimal design dengan background waterTanzania.jpg
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
    /* BACKGROUND DENGAN GAMBAR WATERTANZANIA.png */
    .stApp {{
        {background_style}
    }}
    
    /* Base styling */
    .main {{
        background-color: transparent;
        color: #f0f0f0;
    }}
    
    /* Headers */
    h1 {{
        color: #f0f0f0;
        font-size: 3rem;
        font-weight: 300;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #ffcc00;
        padding-bottom: 1rem;
    }}
    
    h2 {{
        color: #f0f0f0;
        font-size: 1.5rem;
        font-weight: 400;
        margin-bottom: 2rem;
        opacity: 0;
        animation: fadeIn 0.8s ease-out 0.2s forwards;
    }}
    
    h3 {{
        color: #f0f0f0;
        font-size: 1.2rem;
        font-weight: 400;
        margin: 2rem 0 1rem 0;
    }}
    
    /* Expanders */
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
    
    /* Input fields */
    .stSelectbox, .stSlider, .stNumberInput {{
        margin-bottom: 1.5rem;
    }}
    
    label {{
        color: #f0f0f0 !important;
        font-size: 0.9rem !important;
        font-weight: 300 !important;
        margin-bottom: 0.5rem !important;
    }}
    
    .st-bb, .st-at, .st-ae {{
        background-color: rgba(37, 37, 37, 0.9);
        border: 1px solid #333;
        color: #f0f0f0;
    }}
    
    /* Button */
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
    
    /* Prediction result */
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
    
    .prediction-title {{
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 1rem;
        font-weight: 300;
    }}
    
    .prediction-value {{
        font-size: 2.5rem;
        font-weight: 300;
        color: #ffcc00;
        margin: 1rem 0;
    }}
    
    /* Confidence metrics */
    .stMetric {{
        background-color: rgba(26, 26, 26, 0.9);
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid rgba(37, 37, 37, 0.9);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }}
    
    .stMetric:hover {{
        border-color: #333;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    
    .stMetric label {{
        color: #888 !important;
        font-size: 0.8rem !important;
        font-weight: 300 !important;
    }}
    
    .stMetric div {{
        color: #f0f0f0 !important;
        font-size: 1.5rem !important;
        font-weight: 300 !important;
    }}
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {{
        background-color: rgba(26, 26, 26, 0.9);
        border-right: 1px solid rgba(37, 37, 37, 0.9);
    }}
    
    .sidebar .sidebar-content {{
        background-color: rgba(26, 26, 26, 0.9);
        backdrop-filter: blur(10px);
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes scaleUp {{
        from {{ opacity: 0; transform: scale(0.95); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    
    /* Team members */
    .team-member {{
        padding: 1rem 0;
        border-bottom: 1px solid rgba(37, 37, 37, 0.9);
        opacity: 0;
        animation: fadeIn 0.6s ease-out forwards;
    }}
    
    .team-member:last-child {{
        border-bottom: none;
    }}
    
    /* Utility classes */
    .fade-in {{
        opacity: 0;
        animation: fadeIn 0.8s ease-out forwards;
    }}
    
    .delayed-1 {{ animation-delay: 0.3s; }}
    .delayed-2 {{ animation-delay: 0.6s; }}
    .delayed-3 {{ animation-delay: 0.9s; }}
    
    /* Scroll behavior */
    html {{
        scroll-behavior: smooth;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.header("Model Information")
    
    with st.expander("MODEL ARCHITECTURE", expanded=False):
        st.markdown(f"""
        <div style='color: #f0f0f0; font-size: 0.9rem; line-height: 1.6;'>
        <p><strong>Features Used:</strong> {len(preprocessor['used_columns'])}</p>
        <p><strong>Categorical Features:</strong> {len(preprocessor['categorical_cols'])}</p>
        <p><strong>Numerical Features:</strong> {len(preprocessor['numerical_cols'])}</p>
        <p style='margin-top: 1rem; color: #888;'>
        Predictive model analyzing water pump operational status using optimized feature selection for maximum accuracy and efficiency.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("USAGE GUIDE", expanded=False):
        st.markdown("""
        <div style='color: #f0f0f0; font-size: 0.9rem; line-height: 1.6;'>
        <p>1. Complete all input fields</p>
        <p>2. Initiate prediction analysis</p>
        <p>3. Review system assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("DEVELOPMENT TEAM", expanded=False):
        team_members = [
            {"name": "Lucky Bima Bahari Guslau", "NIM": "00000107738", "photo": "Lucky.jpg"},
            {"name": "Kenneth Edbert Aliwarga", "NIM": "00000080925", "photo": "Kenneth.png"},
            {"name": "Muhammad Faiq Hakim Ulinnuha", "NIM": "00000110782", "photo": "Hakim.png"},
            {"name": "Quenessa Salamintargo", "NIM": "00000089201", "photo": "Quenessa.png"}
        ]
        
        for i, member in enumerate(team_members):
            st.markdown(f"<div class='team-member delayed-{i%3+1}'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    photo_path = os.path.join(PHOTOS_PATH, member["photo"])
                    img = Image.open(photo_path)
                    img.thumbnail((80, 80))
                    st.image(img, width=60, use_container_width=True, output_format='PNG')
                except Exception as e:
                    st.image("https://via.placeholder.com/60", width=60, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div style='color: #f0f0f0; font-size: 0.8rem; margin-top: 0.5rem;'>
                <strong>{member['name']}</strong><br>
                <span style='color: #888;'>{member['NIM']}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main app content
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

st.title("Water Pump Status Prediction in Tanzania")

st.markdown("""
<div style='color: #888; font-size: 1.1rem; line-height: 1.6; margin-bottom: 3rem;'>
System analysis for water pump operational status assessment using optimized feature selection
</div>
""", unsafe_allow_html=True)

# Input form
user_input = {}

# Define slider ranges for specific numerical features
slider_ranges = {
    'amount_tsh': (0, 350000, 170000),
    'gps_height': (-90, 2650, 1300),
    'construction_year': (1990, 2013, 2002),
    'region_code': (1, 99, 50)
}

# Group features by category based on the 15 important features
feature_groups = {
    "WATER CHARACTERISTICS": ['quantity', 'quantity_group', 'source'],
    "LOCATION INFORMATION": ['lga', 'region_code', 'region', 'basin'],
    "TECHNICAL SPECIFICATIONS": ['construction_year', 'extraction_type_class', 'gps_height', 'amount_tsh'],
    "INSTALLATION DETAILS": ['installer', 'payment_type'],
    "PUMP CONFIGURATION": ['waterpoint_type', 'waterpoint_type_group']
}

# Create input fields organized by categories
for group_name, features in feature_groups.items():
    with st.expander(group_name, expanded=False):
        cols = st.columns(3)
        col_idx = 0
        
        for col in features:
            if col in preprocessor['used_columns']:
                with cols[col_idx % 3]:
                    if col in preprocessor['categorical_cols']:
                        # For categorical features
                        options = list(preprocessor['label_mappings'][col].keys())
                        user_input[col] = st.selectbox(
                            label=f"{col.replace('_', ' ').title()}",
                            options=options,
                            index=0,
                            key=col
                        )
                    else:
                        # For numerical features
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

# Prediction button
if st.button("ANALYZE PUMP STATUS", type="primary", use_container_width=True):
    try:
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
        
        # Display result with minimal styling
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

st.markdown("</div>", unsafe_allow_html=True)
