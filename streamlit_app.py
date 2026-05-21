<<<<<<< HEAD
import streamlit as st
import numpy as np
from PIL import Image
from cnnClassifier.pipeline.prediction import PredictionPipeline
import os
import io
import base64
import pandas as pd
import matplotlib
from groq import Groq
try:
    import pydicom
except ImportError:
    pydicom = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress extraneous TensorFlow console logs

import warnings
warnings.filterwarnings("ignore")

# --- Application Page Configuration: Set up the Streamlit page title, icon, and layout ---
st.set_page_config(
    page_title="Renal Vision",
    page_icon="🩺",
    layout="wide"
)

# --- Initial Model Download if missing ---
@st.cache_resource(show_spinner="Downloading Kidney Disease Classification Model from Google Drive. Please wait...")
def download_model_if_missing():
    import os
    from pathlib import Path
    try:
        import gdown
    except ImportError:
        st.error("Please install gdown (`pip install gdown`) to download the model automatically.")
        return
    
    preferred = Path("artifacts/training/model.h5")
    fallback = Path("model/model.h5")
    
    if not preferred.exists() and not fallback.exists():
        os.makedirs("model", exist_ok=True)
        file_id = "10AzyxdAYIkA0MT5ITLKYEse0xrINX_r3"
        prefix = "https://drive.google.com/uc?/export=download&id="
        gdown.download(prefix + file_id, str(fallback), quiet=False)

download_model_if_missing()

# --- Initialize Streamlit Session State Variables: Ensure all required state variables are defined to maintain state across reruns ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'latest_label' not in st.session_state:
    st.session_state.latest_label = None
if 'latest_confidence' not in st.session_state:
    st.session_state.latest_confidence = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_clear_confirm' not in st.session_state:
    st.session_state.show_clear_confirm = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'processed_file_id' not in st.session_state:
    st.session_state.processed_file_id = None
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_pdf_bytes' not in st.session_state:
    st.session_state.batch_pdf_bytes = None
if 'batch_csv_bytes' not in st.session_state:
    st.session_state.batch_csv_bytes = None
if 'batch_file_ids' not in st.session_state:
    st.session_state.batch_file_ids = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'chat_input_box' not in st.session_state:
    st.session_state.chat_input_box = ""
if 'prediction_active' not in st.session_state:
    st.session_state.prediction_active = False
if 'persisted_label' not in st.session_state:
    st.session_state.persisted_label = None
if 'persisted_confidence' not in st.session_state:
    st.session_state.persisted_confidence = None
if 'persisted_heatmap' not in st.session_state:
    st.session_state.persisted_heatmap = None
if 'persisted_img' not in st.session_state:
    st.session_state.persisted_img = None
if 'persisted_preprocessed_img' not in st.session_state:
    st.session_state.persisted_preprocessed_img = None
if 'persisted_color_indicator' not in st.session_state:
    st.session_state.persisted_color_indicator = None
if 'persisted_animation_class' not in st.session_state:
    st.session_state.persisted_animation_class = None
if 'model_prewarming_started' not in st.session_state:
    st.session_state.model_prewarming_started = False
if 'manual_clear' not in st.session_state:
    st.session_state.manual_clear = False

# --- Inject Custom Medical Theme CSS: Apply custom styling based on the active theme (light or dark) ---
if st.session_state.theme == 'dark':
    bg_color = "#0B1120"           # Deep Radiology Black-Blue
    grid_color = "rgba(255, 255, 255, 0.03)" # Very faint white grid
    text_color = "#F8FAFC"         # Crisp Off-White Text
    container_bg = "#1E293B"       # Slate Blue Container
    container_border = "#334155"   # Subtle Dark Border
    sidebar_bg = "#0F172A"         # Deep Slate for Sidebar
    sidebar_border = "#1E293B"     # Dark Border
    sidebar_text = "#F8FAFC"       # Light Text
    assistant_text = "#D1FAE5"     # Soft Emerald Text for AI
    title_color = "#38BDF8"        # Bright Cyan for Headers
    val_color = "#E0F2FE"          # Very Light Blue for Values
    info_bg = "rgba(56, 189, 248, 0.15)"    # Soft Cyan for Info
    success_bg = "rgba(80, 200, 120, 0.15)" # Soft Green for Success
    warning_bg = "rgba(217, 83, 79, 0.15)"  # Soft Red for Warning
    expander_bg = "#1E293B"                 # Slate Blue for Dark Mode Expander
    expander_border = "#334155"             # Subtle Dark Border
    expander_text = "#F8FAFC"               # Crisp Off-White Text
    footer_bg = "#FFFFFF"                   # White Footer Background for Dark Theme
    footer_text = "#000000"                 # Black Text for White Footer
    divider_color = "#FFFFFF"               # White Dividers for Dark Theme
    user_chat_bg = "rgba(56, 189, 248, 0.05)"
    assistant_chat_bg = "rgba(80, 200, 120, 0.05)"
    header_line_color = "#FFFFFF"           # White line for Dark Theme
    header_shadow = "rgba(255, 255, 255, 0.15)" # Soft white glow for Dark Theme
    scrollbar_color = "rgba(255, 255, 255, 0.25)"
    scrollbar_hover = "rgba(255, 255, 255, 0.5)"
    sidebar_icon_color = "#FFFFFF"          # Pure white for high contrast
    sidebar_icon_bg = "rgba(56, 189, 248, 0.25)" # Bright cyan button background for visibility
else:
    bg_color = "#F4F6F8"           # Soft Clinical Slate Background
    grid_color = "rgba(0, 0, 0, 0.03)"       # Very faint black grid
    text_color = "#0C0E11"         # Rich Dark Slate for maximum readability
    container_bg = "#FFFFFF"       # Pure White Containers
    container_border = "#E2E8F0"   # Soft Slate Border
    sidebar_bg = "#FFFFFF"         # Pure White Sidebar
    sidebar_border = "#E2E8F0"     # Soft Slate Border
    sidebar_text = "#1E282D"       # Dark Slate Text
    assistant_text = "#065F46"     # Deep Forest Green Text for AI
    title_color = "#1E282D"        # Clinical Cerulean Blue Headers
    val_color = "#4B6878"          # Deep Medical Blue for Values
    info_bg = "rgba(2, 132, 199, 0.1)"      # Soft Blue for Info
    success_bg = "rgba(80, 200, 120, 0.15)" # Soft Green for Success
    warning_bg = "rgba(217, 83, 79, 0.15)"  # Soft Red for Warning
    expander_bg = "#AEC6CF"                 # Pastel Blue for Expander
    expander_border = "#779ECB"             # Darker Pastel Blue Border
    expander_text = "#0F172A"               # Dark Slate Text
    footer_bg = "#000000"                   # Black Footer Background for Light Theme
    footer_text = "#FFFFFF"                 # White Text for Black Footer
    divider_color = "#000000"               # Black Dividers for Light Theme
    user_chat_bg = "rgba(2, 132, 199, 0.05)"
    assistant_chat_bg = "rgba(80, 200, 120, 0.05)"
    header_line_color = "#000000"           # Black line for Light Theme
    header_shadow = "rgba(0, 0, 0, 0.25)"   # Soft dark shadow for Light Theme
    scrollbar_color = "rgba(0, 0, 0, 0.2)"
    scrollbar_hover = "rgba(0, 0, 0, 0.4)"
    sidebar_icon_color = "#1E282D"          # Dark slate
    sidebar_icon_bg = "rgba(0, 0, 0, 0.05)" # Soft grey background

st.markdown(f"""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body {{
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}

    /* Safely apply modern font to typography without breaking Streamlit's native icon ligatures (like the green tick) */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, [data-testid="stMetricValue"], summary, input, textarea, button {{
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif !important;
    }}

    /* Custom Modern Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px !important;
        height: 6px !important;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent !important;
    }}
    ::-webkit-scrollbar-thumb {{
        background-color: {scrollbar_color} !important;
        border-radius: 10px !important;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background-color: {scrollbar_hover} !important;
    }}
    * {{
        scrollbar-width: thin;
        scrollbar-color: {scrollbar_color} transparent;
    }}

    /* Base Streamlit App overrides */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
        background-image: 
            linear-gradient({grid_color} 1px, transparent 1px),
            linear-gradient(90deg, {grid_color} 1px, transparent 1px);
        background-size: 30px 30px;
        background-position: center center;
        color: {text_color};
    }}
    /* Force all dividers to match theme color */
    hr {{
        border-bottom: 2px dashed {divider_color} !important;
    }}
    /* Modern Medical Dashboard Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        background-image: linear-gradient(180deg, {sidebar_bg} 70%, rgba(0, 123, 255, 0.05) 100%) !important;
        border-right: 1px solid {sidebar_border} !important;
        box-shadow: 8px 0 24px rgba(0, 123, 255, 0.12) !important;
        transition: transform 0.4s ease-in-out, box-shadow 0.4s ease-in-out !important;
    }}
    [data-testid="stSidebarResizer"] {{
        background-color: {sidebar_border} !important;
        width: 3px !important;
        opacity: 1 !important;
        transition: background-color 0.3s ease !important;
    }}
    [data-testid="stSidebarResizer"]:hover {{
        background-color: #007BFF !important; /* Blue on hover */
    }}
    [data-testid="stSidebar"] .stMarkdownContainer p, [data-testid="stSidebar"] .stMarkdownContainer div {{
        color: {sidebar_text};
    }}
    [data-testid="stSidebarUserContent"] {{
        padding-top: 2rem !important;
    }}
    
    /* Sidebar Toggle Pulse Animation to attract attention on mobile */
    @keyframes pulseSidebarToggle {{
        0% {{ box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.6); }}
        70% {{ box-shadow: 0 0 0 12px rgba(0, 123, 255, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0, 123, 255, 0); }}
    }}

    /* Ensure Sidebar Toggle Icons are clearly visible and distinguishable */
    [data-testid="collapsedControl"] {{
        background-color: rgba(0, 123, 255, 0.1) !important;
        border: 2px solid #007BFF !important;
        border-radius: 50% !important;
        padding: 6px !important;
        animation: pulseSidebarToggle 2.5s infinite !important;
        transition: all 0.2s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    /* Ensure the Closed Sidebar toggle hides fallback text and properly handles clicks */
    [data-testid="collapsedControl"] {{
        color: transparent !important;
    }}

    /* Hide Default Streamlit Arrow SVGs & Font Icons completely for Closed state so they don't overlap */
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] span {{
        display: none !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    /* Inject Custom Hamburger Menu Icon (☰) ONLY when sidebar is closed */
    [data-testid="collapsedControl"]::before {{
        content: "☰" !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        color: #007BFF !important;
        pointer-events: none !important; /* Ensure the button remains fully clickable */
        transition: transform 0.2s ease-in-out !important;
    }}
    
    [data-testid="collapsedControl"]:hover {{
        background-color: rgba(0, 123, 255, 0.2) !important;
        transform: scale(1.05) !important;
    }}
    /* Apply Medical Blue hover color to toggle icons */
    [data-testid="collapsedControl"]:hover::before {{
        transform: scale(1.05) !important;
    }}

    /* Ensure Native Close Icon inside the Open Sidebar is highlighted by default */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] button[kind="header"] {{
        background-color: rgba(0, 123, 255, 0.1) !important;
        border: 2px solid #007BFF !important;
        border-radius: 50% !important;
        transition: all 0.2s ease-in-out !important;
    }}
    
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stSidebar"] button[kind="header"]:hover {{
        background-color: rgba(0, 123, 255, 0.2) !important;
        transform: scale(1.05) !important;
    }}

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebar"] button[kind="header"] svg {{
        color: #007BFF !important;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent;
        border-top: 5px solid {header_line_color} !important;
        box-shadow: inset 0px 8px 15px -5px {header_shadow} !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {title_color} !important;
        text-align: center !important;
    }}
    .stMarkdownContainer p, label {{
        color: {text_color} !important;
    }}
    
    /* Custom Metric Container */
    .metric-container {{
        background-color: {container_bg};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px {container_border};
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid {container_border};
    }}

    /* Smooth Metric Fade-In Animation */
    @keyframes fadeInScale {{
        0% {{ opacity: 0; transform: scale(0.95) translateY(10px); }}
        100% {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    .metric-animate {{
        animation: fadeInScale 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }}

    .metric-title {{
        font-size: 1.2rem;
        color: {title_color};
        margin-bottom: 10px;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {val_color};
    }}

    /* Animate DataFrames globally */
    [data-testid="stDataFrame"] {{
        animation: fadeInScale 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }}

    /* Professional Dashboard Main Container */
    [data-testid="block-container"] {{
        background-color: {container_bg};
        padding: 3rem 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px {container_border};
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid {container_border};
    }}
    
    /* Center all button wrappers */
    [data-testid="stButton"], [data-testid="stDownloadButton"], [data-testid="stFormSubmitButton"] {{
        display: flex !important;
        justify-content: center !important;
    }}

    /* Medical Blue Buttons */
    .stButton > button[kind="secondary"], .stDownloadButton > button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {{
        background-color: #0056b3 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
    }}
    .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover, [data-testid="stDownloadButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {{
        background-color: #004494 !important;
    }}
    .stButton > button[kind="secondary"]:active, .stDownloadButton > button:active, [data-testid="stDownloadButton"] button:active, [data-testid="stFormSubmitButton"] button:active {{
        background-color: #5C4033 !important; /* Darker brown for click state */
        transform: scale(0.96) !important; /* Subtle scale-down on click */
    }}

    /* Destructive Red Buttons (Primary) */
    [data-testid="stButton"] > button[kind="primary"] {{
        background-color: #D9534F !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:hover {{
        background-color: #C9302C !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:active {{
        background-color: #AC2925 !important;
        transform: scale(0.96) !important;
    }}

    /* Custom Stand-out Cancel Button */
    div.element-container:has(#cancel-btn-highlight) {{
        display: none; /* Hide the anchor container */
    }}
    div.element-container:has(#cancel-btn-highlight) + div.element-container [data-testid="stButton"] > button {{
        background-color: #334155 !important; /* Dark Slate Background */
        color: #FFC107 !important; /* Vibrant Amber Text */
        border: 1px solid #FFC107 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
    }}
    div.element-container:has(#cancel-btn-highlight) + div.element-container [data-testid="stButton"] > button:hover {{
        background-color: #475569 !important;
        color: #FFD54F !important;
    }}
    div.element-container:has(#cancel-btn-highlight) + div.element-container [data-testid="stButton"] > button:active {{
        background-color: #1E293B !important;
        transform: scale(0.96) !important;
    }}

    /* Expander Styling (Session History & Metadata) */
    [data-testid="stExpander"] {{
        background-color: {expander_bg} !important;
        border: 2px solid {expander_border} !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }}
    [data-testid="stExpander"] summary p {{
        color: {expander_text} !important;
        font-weight: bold;
    }}
    [data-testid="stExpander"] svg {{
        color: {expander_text} !important;
    }}

    /* Chatbot Animations */
    @keyframes fadeInUp {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .chat-animate {{
        animation: fadeInUp 0.4s ease-out forwards;
    }}

    /* Bouncing Dots Animation */
    @keyframes bounce {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    .bouncing-dots > div {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #50C878;
        border-radius: 100%;
        animation: bounce 1.4s infinite ease-in-out both;
        margin-right: 4px;
    }}
    .bouncing-dots .dot1 {{ animation-delay: -0.32s; }}
    .bouncing-dots .dot2 {{ animation-delay: -0.16s; }}
    .bouncing-dots .dot3 {{ animation-delay: 0s; }}

    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {{
        border: 2px dashed #000000 !important;
        border-radius: 12px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease-in-out !important;
        position: relative !important;
    }}
    
    /* Hover & Drag-over effect for Dropzone */
    [data-testid="stFileUploadDropzone"]:hover {{
        border-color: #007BFF !important; /* Medical Blue */
        background-color: rgba(0, 123, 255, 0.05) !important; /* Very subtle blue tint */
        box-shadow: 0 0 10px rgba(0, 123, 255, 0.2) !important; /* Soft glow */
    }}

    /* Hide default content on hover to make room for custom text */
    [data-testid="stFileUploadDropzone"]:hover > * {{
        opacity: 0 !important;
        transition: opacity 0.2s ease-in-out !important;
    }}

    /* Add custom text and icon when hovering/dragging */
    [data-testid="stFileUploadDropzone"]:hover::before {{
        content: '📥 Release to Upload Scan...';
        position: absolute;
        font-size: 1.3rem;
        font-weight: 700;
        color: #007BFF;
        pointer-events: none;
        animation: fadeInScale 0.3s ease-out forwards;
    }}

    /* Integrated 'Browse files' Button */
    [data-testid="stFileUploadDropzone"] button {{
        background-color: rgba(0, 123, 255, 0.1) !important;
        color: #007BFF !important;
        border: 1px solid #007BFF !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        padding: 4px 16px !important;
        transition: all 0.2s ease-in-out !important;
    }}
    [data-testid="stFileUploadDropzone"] button:hover {{
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2) !important;
    }}

    /* Danger Animation for Tumor Detection */
    @keyframes pulseDanger {{
        0% {{ box-shadow: 0 0 0 0 rgba(217, 83, 79, 0.7); }}
        70% {{ box-shadow: 0 0 0 15px rgba(217, 83, 79, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(217, 83, 79, 0); }}
    }}
    .danger-animate {{
        animation: pulseDanger 1.5s infinite;
    }}

    /* Calm Success Animation for Normal Detection */
    @keyframes pulseSuccess {{
        0% {{ box-shadow: 0 0 0 0 rgba(80, 200, 120, 0.7); }}
        70% {{ box-shadow: 0 0 0 15px rgba(80, 200, 120, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(80, 200, 120, 0); }}
    }}
    .success-animate {{
        animation: pulseSuccess 2s infinite;
    }}

    /* Professional Tab Bar Styling - Button Style */
    .stTabs {{
        margin-top: 2rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        justify-content: space-between; /* Spread buttons evenly */
        gap: 15px; /* Spacing between buttons */
        border-bottom: none !important; /* Remove full-width line */
        padding-bottom: 10px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        letter-spacing: 0.5px !important;
        font-weight: 600 !important;
        color: #6C757D !important;
        background-color: {container_bg} !important;
        border: 1px solid {container_border} !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        flex: 1 !important; /* Force all tabs to be exactly the same uniform width */
        justify-content: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02) !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(0, 123, 255, 0.05) !important;
        color: #007BFF !important;
        border-color: rgba(0, 123, 255, 0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.1) !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: #007BFF !important;
        background-color: rgba(0, 123, 255, 0.1) !important;
        border-color: #007BFF !important;
        /* Replicate the blue indicator via box-shadow to prevent sliding glitches */
        box-shadow: inset 0 -4px 0 0 #007BFF, 0 4px 8px rgba(0, 123, 255, 0.15) !important;
    }}
    /* Disable the native sliding tab highlight to fix sidebar toggle resize lag */
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    /* Text Input Active/Focus State */
    [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {{
        border-color: #007BFF !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }}

    /* Custom Chat Input Styling */
    [data-testid="stTextInput"] div[data-baseweb="base-input"] {{
        border-radius: 25px !important;
        background-color: {container_bg} !important;
        border: 2px solid #007BFF !important;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }}
    [data-testid="stTextInput"] div[data-baseweb="input"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 4px 16px !important;
    }}
    [data-testid="stTextInput"] input {{
        font-size: 1.05rem !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        background-color: transparent !important;
    }}
    [data-testid="stTextInput"] input::placeholder {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        opacity: 0.5 !important;
        font-weight: 500 !important;
    }}

    /* Alter Progress Bar Color for Batch Processing */
    [data-testid="stProgress"] > div > div > div > div {{
        background-color: #50C878 !important; /* Soft Emerald Green */
    }}

    /* Change Top-Right Processing Spinner Color */
    [data-testid="stStatusWidget"] * {{
        color: #007BFF !important;
    }}
    
    /* Change standard st.spinner border color */
    .stSpinner > div > div {{
        border-top-color: #007BFF !important;
    }}
    </style>
""", unsafe_allow_html=True)
# --- Define Helper Functions for Data Processing and Model Inference ---

def _render_chat_history_txt_bytes(chat_history, prediction=None, confidence=None):
    """
    Convert the provided chat history into a formatted TXT byte string for downloading.
    Supports both English and Hindi characters natively using utf-8-sig.
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = "Renal Vision - Medical Advisory Session History\n"
    content += f"Generated on: {timestamp}\n"
    content += "="*60 + "\n\n"
    
    if prediction is not None and confidence is not None:
        content += "--- Analyzed Medical Scan Results ---\n"
        content += f"Prediction: {prediction}\n"
        content += f"Confidence: {confidence:.2f}%\n"
        content += "-"*60 + "\n\n"
        
    for msg in chat_history or []:
        role = "Patient/User" if msg.get("role") == "user" else "Nephrology AI Assistant"
        text = msg.get("content", "")
        content += f"{role}:\n{text}\n"
        content += "-"*60 + "\n\n"
        
    return content.encode('utf-8-sig'), "text/plain", "txt"

def load_dicom_image(file_bytes) -> Image.Image:
    """
    Safely parse and convert a DICOM file format into a standard PIL Image.
    
    Args:
        file_bytes (bytes): The raw byte data of the uploaded DICOM file.
        
    Returns:
        Image.Image: The processed image converted into a PIL object.
        
    Raises:
        ImportError: If the 'pydicom' library is not installed in the environment.
    """
    if pydicom is None:
        raise ImportError("pydicom library is not installed. Unable to process DICOM files.")
    dicom = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = dicom.pixel_array
    # Normalize the DICOM pixel array to an 8-bit (0-255) format so it can be converted into a standard PIL Image.
    pixel_array = pixel_array - np.min(pixel_array)
    pixel_array = (pixel_array / np.max(pixel_array) * 255).astype(np.uint8)
    return Image.fromarray(pixel_array)

# --- Define the Main Streamlit Application Layout: Setup the sidebar, main title, and layout containers ---
st.sidebar.markdown(f"""
    <h2 style='color: {title_color} !important; font-weight: 800; text-align: left !important; font-size: 1.4rem; padding-bottom: 0.5rem; border-bottom: 1px solid {sidebar_border}; margin-bottom: 1.5rem;'>
    🏥 Control Center
    </h2>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background-color: rgba(80, 200, 120, 0.1); padding: 10px 15px; border-radius: 8px; border-left: 4px solid #50C878; margin-bottom: 20px;">
    <strong style="color: #50C878; font-size: 0.95rem;">🟢 System Status: Online</strong>
</div>

<div style="background-color: {info_bg}; padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border};">
    <strong style="color: {title_color}; font-size: 1rem;">📌 Mandatory Instructions</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>1. Data Acquisition:</strong><br>Upload high-resolution scans (DICOM, JPEG, PNG).<br><br>
        <strong>2. Initialization:</strong><br>AI Assistant unlocks upon successful valid scan analysis.<br><br>
        <strong>3. Validation Layer:</strong><br>System rejects chromatic (RGB) or non-medical images to prevent bias.
    </div>
</div>

<div style="background-color: rgba(139, 92, 246, 0.1); padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border}; border-left: 4px solid #8B5CF6; margin-top: 20px;">
    <strong style="color: #8B5CF6; font-size: 1rem;">🧠 Explainable AI (XAI)</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>Grad-CAM Heatmap Analysis:</strong><br>
        To ensure diagnostic transparency, the model generates heatmaps over the uploaded scans.<br><br>
        🔴 <strong>Warm (Red/Yellow):</strong> High clinical significance regions driving the AI prediction.<br>
        🔵 <strong>Cool (Blue/Cyan):</strong> Lower impact regions.
    </div>
</div>

<div style="background-color: rgba(245, 158, 11, 0.1); padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border}; border-left: 4px solid #F59E0B; margin-top: 20px; margin-bottom: 20px;">
    <strong style="color: #F59E0B; font-size: 1rem;">🌟 App Overview & Features</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>📑 Batch Processing:</strong><br>Upload multiple CT scans at once to generate a consolidated, downloadable diagnostic report.<br><br>
        <strong>🤖 Renal AI Assistant:</strong><br>Context-aware nephrology chatbot providing supportive advisory based on the latest scan predictions.<br><br>
        <strong>✨ Key Capabilities:</strong><br>
        • High-accuracy Tumor vs. Normal classification.<br>
        • Automated image quality & CT verification.<br>
        • Bilingual AI support (English & Hindi).
    </div>
</div>
""", unsafe_allow_html=True)

col_title, col_toggle = st.columns([9, 1])
with col_title:
    st.markdown(f"<h1 style='color: {title_color}; font-weight: 800; letter-spacing: 1px;'>🩺 Renal Vision <span style='font-size: 0.5em; color: #6C757D; vertical-align: middle; font-weight: 600;'>| Diagnostic Dashboard</span></h1>", unsafe_allow_html=True)
with col_toggle:
    st.write("") # Add empty space to vertically align the theme toggle button with the main title.
    if st.session_state.theme == "light":
        if st.button("🌜 Dark"):
            st.session_state.theme = "dark"
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
    else:
        if st.button("🌞 Light"):
            st.session_state.theme = "light"
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

st.markdown("##### *DIAGNOSTIC ADVISORY: This AI-powered tool is designed for preliminary screening and decision support only. The results generated by the VGG16 model must be correlated with clinical findings by a certified Radiologist or Nephrologist. This interface does not provide a final medical diagnosis.*")
st.divider()

st.markdown("### AI-Powered Classification for Normal vs Tumor kidney Scans")

tab1, tab2, tab3 = st.tabs(["🔬 SINGLE SCAN ANALYSIS", "📑 BATCH MEDICAL REPORT", "🤖 RENAL AI"])

with tab1:
    uploaded_file = st.file_uploader("Upload Medical Scan (JPEG, PNG, DICOM)",type=['jpg', 'jpeg', 'png', 'dcm'], key="single_scan_uploader")
    
    if uploaded_file is not None:
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("File size exceeds the 2MB limit. Please upload a smaller file.")
        else:
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            # Verify if the uploaded file is new. This prevents redundant model inference on the same file during UI reruns.
            if st.session_state.processed_file_id != current_file_id:
                st.session_state.prediction_active = False
                st.session_state.manual_clear = False
                st.session_state.processed_file_id = current_file_id
                
                temp_filename = "temp_scan.png" # Use a consistent temp name
                try:
                    # --- Image Loading and Conversion ---
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'dcm':
                        img = load_dicom_image(uploaded_file.getvalue())
                    else:
                        img = Image.open(uploaded_file)
                    
                    # Save the PIL image to a temporary file for the pipeline
                    img.convert("RGB").save(temp_filename, "PNG")

                    # --- Prediction using the unified pipeline ---
                    with st.status("⚙️ Initializing AI Diagnostic Pipeline...", expanded=True) as status:
                        step1 = st.empty()
                        step1.markdown("⏳ **Phase 1/6:** Loading model and medical scan...")
                        pipeline = PredictionPipeline(filename=temp_filename)
                        step1.markdown("✅ **Phase 1/6:** Loading model and medical scan...")
                        
                        # Run validations from the pipeline
                        step2 = st.empty()
                        step2.markdown("⏳ **Phase 2/6:** Validating image quality & integrity...")
                        pipeline.validate_image_quality()
                        step2.markdown("✅ **Phase 2/6:** Validating image quality & integrity...")
                        
                        step3 = st.empty()
                        step3.markdown("⏳ **Phase 3/6:** Verifying renal CT characteristics...")
                        pipeline.verify_ct_scan()
                        step3.markdown("✅ **Phase 3/6:** Verifying renal CT characteristics...")

                        # Get prediction, Grad-CAM, and Preprocessing previews
                        step4 = st.empty()
                        step4.markdown("⏳ **Phase 4/6:** Running deep learning prediction...")
                        prediction_result = pipeline.predict_detailed()
                        step4.markdown("✅ **Phase 4/6:** Running deep learning prediction...")
                        
                        step5 = st.empty()
                        step5.markdown("⏳ **Phase 5/6:** Generating Grad-CAM visualization...")
                        gradcam_b64 = pipeline.make_gradcam_overlay_base64()
                        step5.markdown("✅ **Phase 5/6:** Generating Grad-CAM visualization...")
                        
                        step6 = st.empty()
                        step6.markdown("⏳ **Phase 6/6:** Finalizing diagnostic previews...")
                        previews_b64 = pipeline.make_preprocess_previews_base64()
                        step6.markdown("✅ **Phase 6/6:** Finalizing diagnostic previews...")

                        status.update(label="Scan Analysis Complete!", state="complete", expanded=False)

                        predicted_label = prediction_result["prediction"]
                        confidence = prediction_result["confidence"]
                        
                        # --- Update Session State ---
                        st.session_state.latest_label = predicted_label
                        st.session_state.latest_confidence = confidence * 100
                        
                        new_entry = {
                            "File Name": uploaded_file.name,
                            "Predicted Label": predicted_label,
                            "Confidence Score": f"{confidence * 100:.2f}%"
                        }
                        st.session_state.history.insert(0, new_entry)
                        st.session_state.history = st.session_state.history[:5]
                        
                        color_indicator = '#D9534F' if predicted_label == 'Tumor' else '#50C878'
                        animation_class = 'danger-animate' if predicted_label == 'Tumor' else 'success-animate'

                        st.session_state.prediction_active = True
                        st.session_state.persisted_label = predicted_label
                        st.session_state.persisted_confidence = confidence
                        st.session_state.persisted_heatmap = gradcam_b64
                        st.session_state.persisted_img = img
                        st.session_state.persisted_preprocessed_img = previews_b64["resized"]
                        st.session_state.persisted_color_indicator = color_indicator
                        st.session_state.persisted_animation_class = animation_class
                        
                        if predicted_label == 'Normal':
                            st.snow()

                except ValueError as ve:
                    # Catch specific validation errors from the pipeline
                    st.error(str(ve))
                    st.session_state.processed_file_id = None
                except Exception as e:
                    if pydicom and isinstance(e, pydicom.errors.InvalidDicomError):
                        st.error("Error: The uploaded DICOM file is corrupted or invalid. Please provide a valid DICOM file.")
                    elif isinstance(e, FileNotFoundError):
                        st.error(f"System Error: {str(e)}")
                    else:
                        st.error(f"An unexpected error occurred: {str(e)}")
                    st.session_state.processed_file_id = None
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
    
            # Render the prediction results and metrics conditionally, relying on the active prediction state.
            if st.session_state.prediction_active:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="metric-container metric-animate {st.session_state.persisted_animation_class}">
                        <div class="metric-title">Diagnostic Prediction</div>
                        <div class="metric-value" style="color: {st.session_state.persisted_color_indicator}">{st.session_state.persisted_label}</div>
                        <div style="margin-top: 10px; font-size: 1.1rem; color: {text_color};">Confidence: <strong style="color: {st.session_state.persisted_color_indicator};">{st.session_state.persisted_confidence * 100:.1f}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f'<h3 style="color: {title_color}; font-weight: bold; text-align: center;">Imaging Analysis</h3>', unsafe_allow_html=True)
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    st.image(st.session_state.persisted_img, width="stretch")
                    st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Original Uploaded Scan</strong></p>', unsafe_allow_html=True)
                with img_col2:
                    st.image(st.session_state.persisted_preprocessed_img, width="stretch")
                    st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Preprocessed (224x224)</strong></p>', unsafe_allow_html=True)
                with img_col3:
                    st.image(st.session_state.persisted_heatmap, width="stretch")
                    st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Grad-CAM Heatmap Analysis</strong></p>', unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # The persisted_heatmap is now a data URL: "data:image/png;base64,..."
                b64_string = st.session_state.persisted_heatmap.split(',')[1]
                image_bytes = base64.b64decode(b64_string)
                
                def clear_single_scan_results():
                    st.session_state.prediction_active = False
                    st.session_state.manual_clear = True
                    st.session_state.latest_label = None
                    st.session_state.latest_confidence = None

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="📥 Download Grad-CAM Overlay",
                        data=image_bytes,
                        file_name="gradcam_overlay.png",
                        mime="image/png"
                    )
                with btn_col2:
                    st.button("🧹 Clear Results", key="clear_results_btn", type="primary", on_click=clear_single_scan_results)
    else:
        # Reset the session state variables when the user clears or removes the uploaded file to prepare for a new session.
        st.session_state.processed_file_id = None
        st.session_state.prediction_active = False
        st.session_state.manual_clear = False
        st.session_state.latest_label = None
        st.session_state.latest_confidence = None
        st.session_state.persisted_preprocessed_img = None

    # --- Render Session History & Metadata Section: Display a table of past predictions within an expander ---
    st.markdown("---")
    with st.expander("📋 Session History "):
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            history_df.index = history_df.index + 1
            st.dataframe(history_df, width="stretch")
        else:
            st.info("No session records available yet.")

with tab2:
    st.markdown(f"""
    <div class="metric-animate" style="background-color: {info_bg}; padding: 15px; border-radius: 10px; border: 1px solid {title_color}; margin-bottom: 20px;">
        <strong style="color: {title_color}; font-size: 1.1rem;">BATCH PROCESSING PROTOCOL:</strong><br><br>
        <span style="color: {text_color};">
        You may upload multiple CT scans simultaneously for high-throughput screening. The system will process each scan sequentially and generate a consolidated diagnostic summary table.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    batch_files = st.file_uploader("Drop multiple renal scans here (JPEG, PNG, DICOM)", type=['jpg', 'jpeg', 'png', 'dcm'], accept_multiple_files=True, key="batch_scan_uploader")
    
    # Compare the current batch of files with the previous one. Clear cached results if a new batch is detected to avoid mixing data.
    current_batch_ids = [f"{b.name}_{b.size}" for b in batch_files] if batch_files else []
    if st.session_state.batch_file_ids != current_batch_ids:
        st.session_state.batch_results = None
        st.session_state.batch_pdf_bytes = None
        st.session_state.batch_csv_bytes = None
        st.session_state.batch_file_ids = current_batch_ids
    
    if batch_files:
        if st.button("Start Batch Processing"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_results = []
            temp_filename = "temp_batch_scan.png"

            for i, b_file in enumerate(batch_files):
                status_text.markdown(f"<span style='color: #007BFF; font-weight: 500;'>⏳ Processing {b_file.name} ({i+1}/{len(batch_files)})...</span>", unsafe_allow_html=True)
                try:
                    # --- Load image (including DICOM) ---
                    file_ext = b_file.name.split('.')[-1].lower()
                    if file_ext == 'dcm':
                        img = load_dicom_image(b_file.getvalue())
                    else:
                        img = Image.open(b_file)
                    
                    # Save to temp file for pipeline
                    img.convert("RGB").save(temp_filename, "PNG")

                    # --- Instantiate and run pipeline ---
                    # The model is loaded once and cached inside the pipeline instance
                    pipeline = PredictionPipeline(filename=temp_filename)
                    pipeline.validate_image_quality() # Run validation per image
                    
                    prediction_result = pipeline.predict_detailed()
                    predicted_label = prediction_result["prediction"]
                    confidence = prediction_result["confidence"]
                    
                    batch_results.append({
                        "File Name": b_file.name,
                        "Detection Label": predicted_label,
                        "Confidence (%)": f"{confidence * 100:.2f}"
                    })
                    st.session_state.latest_label = predicted_label
                    st.session_state.latest_confidence = confidence * 100
                except ValueError as ve:
                    st.warning(f"Skipped {b_file.name}: {str(ve)}")
                except Exception as e:
                    st.warning(f"Failed to process {b_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            # Clean up temp file after loop
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            status_text.markdown("<span style='color: #50C878; font-weight: bold;'>✅ Batch processing complete.</span>", unsafe_allow_html=True)
            
            if batch_results:
                results_df = pd.DataFrame(batch_results)
                st.session_state.batch_results = results_df
                
                # Provide an option to generate and download a comprehensive batch report in PDF format using the 'reportlab' library.
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas
                    from reportlab.lib import colors
                    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    elements = []
                    styles = getSampleStyleSheet()
                    
                    elements.append(Paragraph("Batch Medical Report", styles['Title']))
                    elements.append(Spacer(1, 12))
                    
                    data = [["File Name", "Detection Label", "Confidence (%)"]]
                    for res in batch_results:
                        data.append([res["File Name"], res["Detection Label"], res["Confidence (%)"]])
                        
                    t = Table(data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                        ('GRID', (0,0), (-1,-1), 1, colors.black)
                    ]))
                    elements.append(t)
                    doc.build(elements)

                    st.session_state.batch_pdf_bytes = buffer.getvalue()
                    st.session_state.batch_csv_bytes = None
                    
                except ImportError:
                    st.session_state.batch_csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                    st.session_state.batch_pdf_bytes = None

        # Ensure the batch results dataframe and download buttons remain visible across Streamlit reruns by rendering them from the session state.
        if st.session_state.batch_results is not None:
            st.dataframe(st.session_state.batch_results, width="stretch")
            
            if st.session_state.batch_pdf_bytes is not None:
                st.download_button(
                    label="Download Batch Report as PDF",
                    data=st.session_state.batch_pdf_bytes,
                    file_name="batch_medical_report.pdf",
                    mime="application/pdf"
                )
            elif st.session_state.batch_csv_bytes is not None:
                st.warning("To generate PDF reports, please install 'reportlab' (pip install reportlab). For now, you can download the results as CSV.")
                st.download_button(
                    label="Download Batch Report as CSV",
                    data=st.session_state.batch_csv_bytes,
                    file_name="batch_medical_report.csv",
                    mime="text/csv"
                )

# --- Configure the Medical Advisory Chatbot Assistant: Set up API keys and environment variables required by the Groq SDK ---
# Attempt to load the Groq API key from Streamlit secrets and set it as an environment variable for the SDK.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # Silently handle the case where secrets.toml does not exist

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "YOUR_API_KEY_HERE"

CHATBOT_API_KEY = os.environ["GROQ_API_KEY"]
CHATBOT_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Define the Integrated Medical Advisory Chatbot UI and Logic: Setup the tab interface, warning messages, and chat interactions ---
with tab3:
    st.markdown("<h5 style='text-align: center; color: #6C757D;'>Active Clinical Session: Context-Aware Nephrology Assistant</h5>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Language Support Instruction Box
    st.markdown(f"""
    <div class="metric-animate" style="background-color: {info_bg}; padding: 15px 20px; border-radius: 8px; border-left: 5px solid {title_color}; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <strong style="color: {title_color}; font-size: 1.05rem;">🗣️ Supported Languages / भाषा समर्थन</strong><br>
        <span style="color: {text_color}; font-size: 0.95rem;">
        This Medical Assistant fluently supports both <strong>English</strong> and <strong>Hindi</strong>. Feel free to ask your questions in your preferred language.<br>
        यह मेडिकल असिस्टेंट <strong>अंग्रेजी</strong> और <strong>हिंदी</strong> दोनों भाषाओं का समर्थन करता है। बेझिझक अपनी पसंदीदा भाषा में प्रश्न पूछें।
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💬 Medical Advisory Assistant")

    latest_label = st.session_state.latest_label

    if latest_label == 'Tumor':
        st.markdown(f"""
        <div class="metric-animate danger-animate" style="background-color: {warning_bg}; padding: 15px; border-radius: 10px; border: 1px solid #D9534F; margin-bottom: 20px;">
            <strong style="color: #D9534F; font-size: 1.1rem;">⚠️ Clinical Advisory (Tumor Detected):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Consult a Nephrologist or Oncologist</strong> immediately for a formal diagnosis.<br>
            2. <strong>Recommended further tests</strong>: Consider ordering a biopsy, contrast-enhanced MRI, or PET scan.<br>
            3. Please ensure all medical imaging is formally reviewed by a certified radiologist.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif latest_label == 'Normal':
        st.markdown(f"""
        <div class="metric-animate" style="background-color: {success_bg}; padding: 15px; border-radius: 10px; border: 2px solid #50C878; margin-bottom: 20px;">
            <strong style="color: #50C878; font-size: 1.1rem;">✅ Clinical Advisory (Normal Scan):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Wellness Instructions</strong>: Maintain a healthy diet, stay hydrated, and monitor kidney function annually if the patient is high risk.<br>
            2. Routine follow-up is recommended as per the primary care physician's schedule.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-animate" style="background-color: {info_bg}; padding: 15px; border-radius: 10px; border: 1px solid {title_color}; margin-bottom: 20px;">
                <strong style="color: {title_color}; font-size: 1.1rem;">🔒 System Status: Chatbot Locked</strong><br><br>
            <span style="color: {text_color};">
                No renal scan has been uploaded yet. Please provide the image first in the Analysis tabs to unlock personalized context-aware advisory.
            </span>
        </div>
        """, unsafe_allow_html=True)

    def submit_chat():
        if st.session_state.chat_input_box.strip():
            st.session_state.user_query = st.session_state.chat_input_box
            st.session_state.chat_input_box = ""

    st.markdown("<h5 style='color: #6C757D; margin-top: 15px;'>Feel free to ask your doubts to Renal Vision AI.</h5>", unsafe_allow_html=True)
    st.text_input("Query", key="chat_input_box", placeholder="Type your message here and press Enter...", label_visibility="collapsed", on_change=submit_chat)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 0.85rem; margin-top: 5px;'><em>Renal AI can make mistakes, please double-check it.</em></p>", unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid {title_color}; background-color: {user_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <strong style="color: {title_color};">👤 You:</strong><br><span style="color: {text_color};">{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #50C878; background-color: {assistant_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <strong style="color: #50C878;">🩺 Assistant:</strong><br><span style="color: {assistant_text};">{message['content']}</span>
                """, unsafe_allow_html=True)

        active_chat_placeholder = st.container()

    prompt = st.session_state.user_query

    if prompt:
        st.session_state.user_query = "" # Immediately clear the user query from the session state to prevent duplicate processing on subsequent interactions.
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Inject CSS to create a shimmering loading animation inside the input box
        # and disable further input while the model generates the response.
        st.markdown(f"""
        <style>
        @keyframes shimmerInput {{
            0% {{ background-position: -1000px 0; }}
            100% {{ background-position: 1000px 0; }}
        }}
        [data-testid="stTextInput"] div[data-baseweb="base-input"] {{
            background: linear-gradient(90deg, {container_bg} 25%, rgba(0, 123, 255, 0.2) 50%, {container_bg} 75%) !important;
            background-size: 1000px 100% !important;
            animation: shimmerInput 2s infinite linear !important;
            border-color: #007BFF !important;
            pointer-events: none !important;
        }}
        [data-testid="stTextInput"] input {{
            opacity: 0.5 !important;
            pointer-events: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        with active_chat_placeholder:
            st.markdown(f"""
            <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid {title_color}; background-color: {user_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <strong style="color: {title_color};">👤 You:</strong><br><span style="color: {text_color};">{prompt}</span>
            </div>
            """, unsafe_allow_html=True)
            
            assistant_placeholder = st.empty()
            
            # Display a temporary 'thinking' animation to provide visual feedback to the user while waiting for the LLM API response.
            assistant_placeholder.markdown(f"""
            <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #50C878; background-color: {assistant_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <strong style="color: #50C878;">🩺 Assistant:</strong><br>
                <div class="bouncing-dots" style="margin-top: 8px;">
                    <div class="dot1"></div>
                    <div class="dot2"></div>
                    <div class="dot3"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Handle the AI API integration: Verify if a scan has been processed first, then call the Groq LLM API if a valid key is provided.
        if not latest_label:
            response = "⚠️ Please Provide the Image first. The Medical Advisory Chatbot remains locked until a valid scan is analyzed."
            assistant_placeholder.markdown(f"""
            <div style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #D9534F; background-color: {warning_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <strong style="color: #D9534F;">🔒 Assistant:</strong><br><span style="color: {text_color};">{response}</span>
            </div>
            """, unsafe_allow_html=True)
        elif CHATBOT_API_KEY != "YOUR_API_KEY_HERE":
            try:
                # Initialize the Groq client. The SDK automatically utilizes the 'GROQ_API_KEY' environment variable configured earlier.
                client = Groq()
                
                system_prompt = f"""You are a specialized Nephrology AI Assistant. 
Current Patient Context:
- Diagnostic Prediction: {st.session_state.latest_label}
- Model Confidence: {st.session_state.latest_confidence:.2f}%

Instructions: Answer the user's query based strictly on this clinical context. If the prediction is 'Tumor', tailor your advice around next-step staging, oncological consultations, and lifestyle precautions, while maintaining a supportive but objective medical tone. Do not give a final medical prescription; emphasize clinical correlation.

IMPORTANT: Always conclude your response with the exact following disclaimer:
"Disclaimer: Renal AI can make mistakes, please double-check it." """

                messages = [
                    {"role": "system", "content": system_prompt}
                ]
                for msg in st.session_state.chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
                completion = client.chat.completions.create(
                    model=CHATBOT_MODEL_NAME,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                response = ""
                for chunk in completion:
                    response += chunk.choices[0].delta.content or ""
                    assistant_placeholder.markdown(f"""
                    <div style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #50C878; background-color: {assistant_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <strong style="color: #50C878;">🩺 Assistant:</strong><br><span style="color: {assistant_text};">{response}✚</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                response = f"Error calling Groq API: {e}"
        else:
            response = "I have provided the standard medical advisory above. Please configure your Groq API key securely in Streamlit Secrets to enable advanced interactive chat capabilities."

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.balloons()

    # --- Implement Chat History Export Functionality: Provide a UI and logic for users to clear or download their chat history ---
    if st.session_state.chat_history:
        st.markdown("---")
        chat_export = "Medical Advisory Assistant - Session Chat History\n"
        chat_export += "="*50 + "\n\n"
        for msg in st.session_state.chat_history:
            role = "Patient/User" if msg["role"] == "user" else "Nephrology AI Assistant"
            chat_export += f"{role}:\n{msg['content']}\n\n"
            
        def confirm_clear():
            st.session_state.show_clear_confirm = True
            
        def execute_clear():
            st.session_state.chat_history = []
            st.session_state.show_clear_confirm = False
            
        def cancel_clear():
            st.session_state.show_clear_confirm = False
            
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.show_clear_confirm:
                st.warning("Are you sure you want to clear the chat?")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.button("✔️ Yes, Clear", key="yes_clear", type="primary", on_click=execute_clear)
                with sc2:
                    st.markdown("<span id='cancel-btn-highlight'></span>", unsafe_allow_html=True)
                    st.button("❌ Cancel", key="cancel_clear", on_click=cancel_clear)
            else:
                st.button("Clear Chat History", on_click=confirm_clear)
        with col2:
            export_bytes, mime_type, file_ext = _render_chat_history_txt_bytes(
                st.session_state.chat_history, 
                prediction=st.session_state.get('latest_label'),
                confidence=st.session_state.get('latest_confidence')
            )
            if export_bytes:
                st.download_button(
                    label="📥 Download Conversation History",
                    data=export_bytes,
                    file_name=f"renal_ai_chat_history.{file_ext}",
                    mime=mime_type
                )

# --- Render Application Footer: Display the developer credits at the bottom of the page ---
st.markdown(f"""
<div style="display: flex; justify-content: center; margin-top: 50px; margin-bottom: 20px;">
    <div style="background-color: {footer_bg}; color: {footer_text}; padding: 10px 30px; border-radius: 20px; font-weight: bold; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Made with ❤️ by Rohit.
    </div>
</div>
=======
import streamlit as st
import numpy as np
from PIL import Image
from cnnClassifier.pipeline.prediction import PredictionPipeline
import os
import io
import base64
import pandas as pd
import matplotlib
from groq import Groq
try:
    import pydicom
except ImportError:
    pydicom = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress extraneous TensorFlow console logs

import warnings
warnings.filterwarnings("ignore")

# --- Application Page Configuration: Set up the Streamlit page title, icon, and layout ---
st.set_page_config(
    page_title="Renal Vision",
    page_icon="🩺",
    layout="wide"
)

# --- Initial Model Download if missing ---
@st.cache_resource(show_spinner="Downloading Kidney Disease Classification Model from Google Drive. Please wait...")
def download_model_if_missing():
    import os
    from pathlib import Path
    try:
        import gdown
    except ImportError:
        st.error("Please install gdown (`pip install gdown`) to download the model automatically.")
        return
    
    preferred = Path("artifacts/training/model.h5")
    fallback = Path("model/model.h5")
    
    if not preferred.exists() and not fallback.exists():
        os.makedirs("model", exist_ok=True)
        file_id = "10AzyxdAYIkA0MT5ITLKYEse0xrINX_r3"
        prefix = "https://drive.google.com/uc?/export=download&id="
        gdown.download(prefix + file_id, str(fallback), quiet=False)

download_model_if_missing()

# --- Initialize Streamlit Session State Variables: Ensure all required state variables are defined to maintain state across reruns ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'latest_label' not in st.session_state:
    st.session_state.latest_label = None
if 'latest_confidence' not in st.session_state:
    st.session_state.latest_confidence = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_clear_confirm' not in st.session_state:
    st.session_state.show_clear_confirm = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'processed_file_id' not in st.session_state:
    st.session_state.processed_file_id = None
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_pdf_bytes' not in st.session_state:
    st.session_state.batch_pdf_bytes = None
if 'batch_csv_bytes' not in st.session_state:
    st.session_state.batch_csv_bytes = None
if 'batch_file_ids' not in st.session_state:
    st.session_state.batch_file_ids = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'chat_input_box' not in st.session_state:
    st.session_state.chat_input_box = ""
if 'prediction_active' not in st.session_state:
    st.session_state.prediction_active = False
if 'persisted_label' not in st.session_state:
    st.session_state.persisted_label = None
if 'persisted_confidence' not in st.session_state:
    st.session_state.persisted_confidence = None
if 'persisted_heatmap' not in st.session_state:
    st.session_state.persisted_heatmap = None
if 'persisted_img' not in st.session_state:
    st.session_state.persisted_img = None
if 'persisted_preprocessed_img' not in st.session_state:
    st.session_state.persisted_preprocessed_img = None
if 'persisted_color_indicator' not in st.session_state:
    st.session_state.persisted_color_indicator = None
if 'persisted_animation_class' not in st.session_state:
    st.session_state.persisted_animation_class = None
if 'model_prewarming_started' not in st.session_state:
    st.session_state.model_prewarming_started = False
if 'manual_clear' not in st.session_state:
    st.session_state.manual_clear = False

# --- Inject Custom Medical Theme CSS: Apply custom styling based on the active theme (light or dark) ---
if st.session_state.theme == 'dark':
    bg_color = "#0B1120"           # Deep Radiology Black-Blue
    grid_color = "rgba(255, 255, 255, 0.03)" # Very faint white grid
    text_color = "#F8FAFC"         # Crisp Off-White Text
    container_bg = "#1E293B"       # Slate Blue Container
    container_border = "#334155"   # Subtle Dark Border
    sidebar_bg = "#0F172A"         # Deep Slate for Sidebar
    sidebar_border = "#1E293B"     # Dark Border
    sidebar_text = "#F8FAFC"       # Light Text
    assistant_text = "#D1FAE5"     # Soft Emerald Text for AI
    title_color = "#38BDF8"        # Bright Cyan for Headers
    val_color = "#E0F2FE"          # Very Light Blue for Values
    info_bg = "rgba(56, 189, 248, 0.15)"    # Soft Cyan for Info
    success_bg = "rgba(80, 200, 120, 0.15)" # Soft Green for Success
    warning_bg = "rgba(217, 83, 79, 0.15)"  # Soft Red for Warning
    expander_bg = "#1E293B"                 # Slate Blue for Dark Mode Expander
    expander_border = "#334155"             # Subtle Dark Border
    expander_text = "#F8FAFC"               # Crisp Off-White Text
    footer_bg = "#FFFFFF"                   # White Footer Background for Dark Theme
    footer_text = "#000000"                 # Black Text for White Footer
    divider_color = "#FFFFFF"               # White Dividers for Dark Theme
    user_chat_bg = "rgba(56, 189, 248, 0.05)"
    assistant_chat_bg = "rgba(80, 200, 120, 0.05)"
    header_line_color = "#FFFFFF"           # White line for Dark Theme
    header_shadow = "rgba(255, 255, 255, 0.15)" # Soft white glow for Dark Theme
    scrollbar_color = "rgba(255, 255, 255, 0.25)"
    scrollbar_hover = "rgba(255, 255, 255, 0.5)"
    sidebar_icon_color = "#FFFFFF"          # Pure white for high contrast
    sidebar_icon_bg = "rgba(56, 189, 248, 0.25)" # Bright cyan button background for visibility
else:
    bg_color = "#F4F6F8"           # Soft Clinical Slate Background
    grid_color = "rgba(0, 0, 0, 0.03)"       # Very faint black grid
    text_color = "#0C0E11"         # Rich Dark Slate for maximum readability
    container_bg = "#FFFFFF"       # Pure White Containers
    container_border = "#E2E8F0"   # Soft Slate Border
    sidebar_bg = "#FFFFFF"         # Pure White Sidebar
    sidebar_border = "#E2E8F0"     # Soft Slate Border
    sidebar_text = "#1E282D"       # Dark Slate Text
    assistant_text = "#065F46"     # Deep Forest Green Text for AI
    title_color = "#1E282D"        # Clinical Cerulean Blue Headers
    val_color = "#4B6878"          # Deep Medical Blue for Values
    info_bg = "rgba(2, 132, 199, 0.1)"      # Soft Blue for Info
    success_bg = "rgba(80, 200, 120, 0.15)" # Soft Green for Success
    warning_bg = "rgba(217, 83, 79, 0.15)"  # Soft Red for Warning
    expander_bg = "#AEC6CF"                 # Pastel Blue for Expander
    expander_border = "#779ECB"             # Darker Pastel Blue Border
    expander_text = "#0F172A"               # Dark Slate Text
    footer_bg = "#000000"                   # Black Footer Background for Light Theme
    footer_text = "#FFFFFF"                 # White Text for Black Footer
    divider_color = "#000000"               # Black Dividers for Light Theme
    user_chat_bg = "rgba(2, 132, 199, 0.05)"
    assistant_chat_bg = "rgba(80, 200, 120, 0.05)"
    header_line_color = "#000000"           # Black line for Light Theme
    header_shadow = "rgba(0, 0, 0, 0.25)"   # Soft dark shadow for Light Theme
    scrollbar_color = "rgba(0, 0, 0, 0.2)"
    scrollbar_hover = "rgba(0, 0, 0, 0.4)"
    sidebar_icon_color = "#1E282D"          # Dark slate
    sidebar_icon_bg = "rgba(0, 0, 0, 0.05)" # Soft grey background

st.markdown(f"""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body {{
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}

    /* Safely apply modern font to typography without breaking Streamlit's native icon ligatures (like the green tick) */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, [data-testid="stMetricValue"], summary, input, textarea, button {{
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif !important;
    }}

    /* Custom Modern Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px !important;
        height: 6px !important;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent !important;
    }}
    ::-webkit-scrollbar-thumb {{
        background-color: {scrollbar_color} !important;
        border-radius: 10px !important;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background-color: {scrollbar_hover} !important;
    }}
    * {{
        scrollbar-width: thin;
        scrollbar-color: {scrollbar_color} transparent;
    }}

    /* Base Streamlit App overrides */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
        background-image: 
            linear-gradient({grid_color} 1px, transparent 1px),
            linear-gradient(90deg, {grid_color} 1px, transparent 1px);
        background-size: 30px 30px;
        background-position: center center;
        color: {text_color};
    }}
    /* Force all dividers to match theme color */
    hr {{
        border-bottom: 2px dashed {divider_color} !important;
    }}
    /* Modern Medical Dashboard Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        background-image: linear-gradient(180deg, {sidebar_bg} 70%, rgba(0, 123, 255, 0.05) 100%) !important;
        border-right: 1px solid {sidebar_border} !important;
        box-shadow: 8px 0 24px rgba(0, 123, 255, 0.12) !important;
        transition: transform 0.4s ease-in-out, box-shadow 0.4s ease-in-out !important;
    }}
    [data-testid="stSidebarResizer"] {{
        background-color: {sidebar_border} !important;
        width: 3px !important;
        opacity: 1 !important;
        transition: background-color 0.3s ease !important;
    }}
    [data-testid="stSidebarResizer"]:hover {{
        background-color: #007BFF !important; /* Blue on hover */
    }}
    [data-testid="stSidebar"] .stMarkdownContainer p, [data-testid="stSidebar"] .stMarkdownContainer div {{
        color: {sidebar_text};
    }}
    [data-testid="stSidebarUserContent"] {{
        padding-top: 2rem !important;
    }}
    
    /* Sidebar Toggle Pulse Animation to attract attention on mobile */
    @keyframes pulseSidebarToggle {{
        0% {{ box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.6); }}
        70% {{ box-shadow: 0 0 0 12px rgba(0, 123, 255, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0, 123, 255, 0); }}
    }}

    /* Ensure Sidebar Toggle Icons are clearly visible and distinguishable */
    [data-testid="collapsedControl"] {{
        background-color: rgba(0, 123, 255, 0.1) !important;
        border: 2px solid #007BFF !important;
        border-radius: 50% !important;
        padding: 6px !important;
        animation: pulseSidebarToggle 2.5s infinite !important;
        transition: all 0.2s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    /* Ensure the Closed Sidebar toggle hides fallback text and properly handles clicks */
    [data-testid="collapsedControl"] {{
        color: transparent !important;
    }}

    /* Hide Default Streamlit Arrow SVGs & Font Icons completely for Closed state so they don't overlap */
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] span {{
        display: none !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    /* Inject Custom Hamburger Menu Icon (☰) ONLY when sidebar is closed */
    [data-testid="collapsedControl"]::before {{
        content: "☰" !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        color: #007BFF !important;
        pointer-events: none !important; /* Ensure the button remains fully clickable */
        transition: transform 0.2s ease-in-out !important;
    }}
    
    [data-testid="collapsedControl"]:hover {{
        background-color: rgba(0, 123, 255, 0.2) !important;
        transform: scale(1.05) !important;
    }}
    /* Apply Medical Blue hover color to toggle icons */
    [data-testid="collapsedControl"]:hover::before {{
        transform: scale(1.05) !important;
    }}

    /* Ensure Native Close Icon inside the Open Sidebar is highlighted by default */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] button[kind="header"] {{
        background-color: rgba(0, 123, 255, 0.1) !important;
        border: 2px solid #007BFF !important;
        border-radius: 50% !important;
        transition: all 0.2s ease-in-out !important;
    }}
    
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stSidebar"] button[kind="header"]:hover {{
        background-color: rgba(0, 123, 255, 0.2) !important;
        transform: scale(1.05) !important;
    }}

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebar"] button[kind="header"] svg {{
        color: #007BFF !important;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent;
        border-top: 5px solid {header_line_color} !important;
        box-shadow: inset 0px 8px 15px -5px {header_shadow} !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {title_color} !important;
        text-align: center !important;
    }}
    .stMarkdownContainer p, label {{
        color: {text_color} !important;
    }}
    
    /* Custom Metric Container */
    .metric-container {{
        background-color: {container_bg};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px {container_border};
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid {container_border};
    }}

    /* Smooth Metric Fade-In Animation */
    @keyframes fadeInScale {{
        0% {{ opacity: 0; transform: scale(0.95) translateY(10px); }}
        100% {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    .metric-animate {{
        animation: fadeInScale 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }}

    .metric-title {{
        font-size: 1.2rem;
        color: {title_color};
        margin-bottom: 10px;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {val_color};
    }}

    /* Animate DataFrames globally */
    [data-testid="stDataFrame"] {{
        animation: fadeInScale 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }}

    /* Professional Dashboard Main Container */
    [data-testid="block-container"] {{
        background-color: {container_bg};
        padding: 3rem 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px {container_border};
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid {container_border};
    }}
    
    /* Center all button wrappers */
    [data-testid="stButton"], [data-testid="stDownloadButton"], [data-testid="stFormSubmitButton"] {{
        display: flex !important;
        justify-content: center !important;
    }}

    /* Medical Blue Buttons */
    .stButton > button[kind="secondary"], .stDownloadButton > button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {{
        background-color: #0056b3 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
    }}
    .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover, [data-testid="stDownloadButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {{
        background-color: #004494 !important;
    }}
    .stButton > button[kind="secondary"]:active, .stDownloadButton > button:active, [data-testid="stDownloadButton"] button:active, [data-testid="stFormSubmitButton"] button:active {{
        background-color: #5C4033 !important; /* Darker brown for click state */
        transform: scale(0.96) !important; /* Subtle scale-down on click */
    }}

    /* Destructive Red Buttons (Primary) */
    [data-testid="stButton"] > button[kind="primary"] {{
        background-color: #D9534F !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:hover {{
        background-color: #C9302C !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:active {{
        background-color: #AC2925 !important;
        transform: scale(0.96) !important;
    }}

    /* Custom Stand-out Cancel Button */
    div.element-container:has(#cancel-btn-highlight) {{
        display: none; /* Hide the anchor container */
    }}
    div.element-container:has(#cancel-btn-highlight) + div.element-container [data-testid="stButton"] > button {{
        background-color: #334155 !important; /* Dark Slate Background */
        color: #FFC107 !important; /* Vibrant Amber Text */
        border: 1px solid #FFC107 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
    }}
    div.element-container:has(#cancel-btn-highlight) + div.element-container [data-testid="stButton"] > button:hover {{
        background-color: #475569 !important;
        color: #FFD54F !important;
    }}
    div.element-container:has(#cancel-btn-highlight) + div.element-container [data-testid="stButton"] > button:active {{
        background-color: #1E293B !important;
        transform: scale(0.96) !important;
    }}

    /* Expander Styling (Session History & Metadata) */
    [data-testid="stExpander"] {{
        background-color: {expander_bg} !important;
        border: 2px solid {expander_border} !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }}
    [data-testid="stExpander"] summary p {{
        color: {expander_text} !important;
        font-weight: bold;
    }}
    [data-testid="stExpander"] svg {{
        color: {expander_text} !important;
    }}

    /* Chatbot Animations */
    @keyframes fadeInUp {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .chat-animate {{
        animation: fadeInUp 0.4s ease-out forwards;
    }}

    /* Bouncing Dots Animation */
    @keyframes bounce {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    .bouncing-dots > div {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #50C878;
        border-radius: 100%;
        animation: bounce 1.4s infinite ease-in-out both;
        margin-right: 4px;
    }}
    .bouncing-dots .dot1 {{ animation-delay: -0.32s; }}
    .bouncing-dots .dot2 {{ animation-delay: -0.16s; }}
    .bouncing-dots .dot3 {{ animation-delay: 0s; }}

    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {{
        border: 2px dashed #000000 !important;
        border-radius: 12px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease-in-out !important;
        position: relative !important;
    }}
    
    /* Hover & Drag-over effect for Dropzone */
    [data-testid="stFileUploadDropzone"]:hover {{
        border-color: #007BFF !important; /* Medical Blue */
        background-color: rgba(0, 123, 255, 0.05) !important; /* Very subtle blue tint */
        box-shadow: 0 0 10px rgba(0, 123, 255, 0.2) !important; /* Soft glow */
    }}

    /* Hide default content on hover to make room for custom text */
    [data-testid="stFileUploadDropzone"]:hover > * {{
        opacity: 0 !important;
        transition: opacity 0.2s ease-in-out !important;
    }}

    /* Add custom text and icon when hovering/dragging */
    [data-testid="stFileUploadDropzone"]:hover::before {{
        content: '📥 Release to Upload Scan...';
        position: absolute;
        font-size: 1.3rem;
        font-weight: 700;
        color: #007BFF;
        pointer-events: none;
        animation: fadeInScale 0.3s ease-out forwards;
    }}

    /* Integrated 'Browse files' Button */
    [data-testid="stFileUploadDropzone"] button {{
        background-color: rgba(0, 123, 255, 0.1) !important;
        color: #007BFF !important;
        border: 1px solid #007BFF !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        padding: 4px 16px !important;
        transition: all 0.2s ease-in-out !important;
    }}
    [data-testid="stFileUploadDropzone"] button:hover {{
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2) !important;
    }}

    /* Danger Animation for Tumor Detection */
    @keyframes pulseDanger {{
        0% {{ box-shadow: 0 0 0 0 rgba(217, 83, 79, 0.7); }}
        70% {{ box-shadow: 0 0 0 15px rgba(217, 83, 79, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(217, 83, 79, 0); }}
    }}
    .danger-animate {{
        animation: pulseDanger 1.5s infinite;
    }}

    /* Calm Success Animation for Normal Detection */
    @keyframes pulseSuccess {{
        0% {{ box-shadow: 0 0 0 0 rgba(80, 200, 120, 0.7); }}
        70% {{ box-shadow: 0 0 0 15px rgba(80, 200, 120, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(80, 200, 120, 0); }}
    }}
    .success-animate {{
        animation: pulseSuccess 2s infinite;
    }}

    /* Professional Tab Bar Styling - Button Style */
    .stTabs {{
        margin-top: 2rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        justify-content: space-between; /* Spread buttons evenly */
        gap: 15px; /* Spacing between buttons */
        border-bottom: none !important; /* Remove full-width line */
        padding-bottom: 10px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        letter-spacing: 0.5px !important;
        font-weight: 600 !important;
        color: #6C757D !important;
        background-color: {container_bg} !important;
        border: 1px solid {container_border} !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        flex: 1 !important; /* Force all tabs to be exactly the same uniform width */
        justify-content: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02) !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(0, 123, 255, 0.05) !important;
        color: #007BFF !important;
        border-color: rgba(0, 123, 255, 0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.1) !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: #007BFF !important;
        background-color: rgba(0, 123, 255, 0.1) !important;
        border-color: #007BFF !important;
        /* Replicate the blue indicator via box-shadow to prevent sliding glitches */
        box-shadow: inset 0 -4px 0 0 #007BFF, 0 4px 8px rgba(0, 123, 255, 0.15) !important;
    }}
    /* Disable the native sliding tab highlight to fix sidebar toggle resize lag */
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    /* Text Input Active/Focus State */
    [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {{
        border-color: #007BFF !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }}

    /* Custom Chat Input Styling */
    [data-testid="stTextInput"] div[data-baseweb="base-input"] {{
        border-radius: 25px !important;
        background-color: {container_bg} !important;
        border: 2px solid #007BFF !important;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }}
    [data-testid="stTextInput"] div[data-baseweb="input"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 4px 16px !important;
    }}
    [data-testid="stTextInput"] input {{
        font-size: 1.05rem !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        background-color: transparent !important;
    }}
    [data-testid="stTextInput"] input::placeholder {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        opacity: 0.5 !important;
        font-weight: 500 !important;
    }}

    /* Alter Progress Bar Color for Batch Processing */
    [data-testid="stProgress"] > div > div > div > div {{
        background-color: #50C878 !important; /* Soft Emerald Green */
    }}

    /* Change Top-Right Processing Spinner Color */
    [data-testid="stStatusWidget"] * {{
        color: #007BFF !important;
    }}
    
    /* Change standard st.spinner border color */
    .stSpinner > div > div {{
        border-top-color: #007BFF !important;
    }}
    </style>
""", unsafe_allow_html=True)
# --- Define Helper Functions for Data Processing and Model Inference ---

def _render_chat_history_txt_bytes(chat_history, prediction=None, confidence=None):
    """
    Convert the provided chat history into a formatted TXT byte string for downloading.
    Supports both English and Hindi characters natively using utf-8-sig.
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = "Renal Vision - Medical Advisory Session History\n"
    content += f"Generated on: {timestamp}\n"
    content += "="*60 + "\n\n"
    
    if prediction is not None and confidence is not None:
        content += "--- Analyzed Medical Scan Results ---\n"
        content += f"Prediction: {prediction}\n"
        content += f"Confidence: {confidence:.2f}%\n"
        content += "-"*60 + "\n\n"
        
    for msg in chat_history or []:
        role = "Patient/User" if msg.get("role") == "user" else "Nephrology AI Assistant"
        text = msg.get("content", "")
        content += f"{role}:\n{text}\n"
        content += "-"*60 + "\n\n"
        
    return content.encode('utf-8-sig'), "text/plain", "txt"

def load_dicom_image(file_bytes) -> Image.Image:
    """
    Safely parse and convert a DICOM file format into a standard PIL Image.
    
    Args:
        file_bytes (bytes): The raw byte data of the uploaded DICOM file.
        
    Returns:
        Image.Image: The processed image converted into a PIL object.
        
    Raises:
        ImportError: If the 'pydicom' library is not installed in the environment.
    """
    if pydicom is None:
        raise ImportError("pydicom library is not installed. Unable to process DICOM files.")
    dicom = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = dicom.pixel_array
    # Normalize the DICOM pixel array to an 8-bit (0-255) format so it can be converted into a standard PIL Image.
    pixel_array = pixel_array - np.min(pixel_array)
    pixel_array = (pixel_array / np.max(pixel_array) * 255).astype(np.uint8)
    return Image.fromarray(pixel_array)

# --- Define the Main Streamlit Application Layout: Setup the sidebar, main title, and layout containers ---
st.sidebar.markdown(f"""
    <h2 style='color: {title_color} !important; font-weight: 800; text-align: left !important; font-size: 1.4rem; padding-bottom: 0.5rem; border-bottom: 1px solid {sidebar_border}; margin-bottom: 1.5rem;'>
    🏥 Control Center
    </h2>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background-color: rgba(80, 200, 120, 0.1); padding: 10px 15px; border-radius: 8px; border-left: 4px solid #50C878; margin-bottom: 20px;">
    <strong style="color: #50C878; font-size: 0.95rem;">🟢 System Status: Online</strong>
</div>

<div style="background-color: {info_bg}; padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border};">
    <strong style="color: {title_color}; font-size: 1rem;">📌 Mandatory Instructions</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>1. Data Acquisition:</strong><br>Upload high-resolution scans (DICOM, JPEG, PNG).<br><br>
        <strong>2. Initialization:</strong><br>AI Assistant unlocks upon successful valid scan analysis.<br><br>
        <strong>3. Validation Layer:</strong><br>System rejects chromatic (RGB) or non-medical images to prevent bias.
    </div>
</div>

<div style="background-color: rgba(139, 92, 246, 0.1); padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border}; border-left: 4px solid #8B5CF6; margin-top: 20px;">
    <strong style="color: #8B5CF6; font-size: 1rem;">🧠 Explainable AI (XAI)</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>Grad-CAM Heatmap Analysis:</strong><br>
        To ensure diagnostic transparency, the model generates heatmaps over the uploaded scans.<br><br>
        🔴 <strong>Warm (Red/Yellow):</strong> High clinical significance regions driving the AI prediction.<br>
        🔵 <strong>Cool (Blue/Cyan):</strong> Lower impact regions.
    </div>
</div>

<div style="background-color: rgba(245, 158, 11, 0.1); padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border}; border-left: 4px solid #F59E0B; margin-top: 20px; margin-bottom: 20px;">
    <strong style="color: #F59E0B; font-size: 1rem;">🌟 App Overview & Features</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>📑 Batch Processing:</strong><br>Upload multiple CT scans at once to generate a consolidated, downloadable diagnostic report.<br><br>
        <strong>🤖 Renal AI Assistant:</strong><br>Context-aware nephrology chatbot providing supportive advisory based on the latest scan predictions.<br><br>
        <strong>✨ Key Capabilities:</strong><br>
        • High-accuracy Tumor vs. Normal classification.<br>
        • Automated image quality & CT verification.<br>
        • Bilingual AI support (English & Hindi).
    </div>
</div>
""", unsafe_allow_html=True)

col_title, col_toggle = st.columns([9, 1])
with col_title:
    st.markdown(f"<h1 style='color: {title_color}; font-weight: 800; letter-spacing: 1px;'>🩺 Renal Vision <span style='font-size: 0.5em; color: #6C757D; vertical-align: middle; font-weight: 600;'>| Diagnostic Dashboard</span></h1>", unsafe_allow_html=True)
with col_toggle:
    st.write("") # Add empty space to vertically align the theme toggle button with the main title.
    if st.session_state.theme == "light":
        if st.button("🌜 Dark"):
            st.session_state.theme = "dark"
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
    else:
        if st.button("🌞 Light"):
            st.session_state.theme = "light"
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

st.markdown("##### *DIAGNOSTIC ADVISORY: This AI-powered tool is designed for preliminary screening and decision support only. The results generated by the VGG16 model must be correlated with clinical findings by a certified Radiologist or Nephrologist. This interface does not provide a final medical diagnosis.*")
st.divider()

st.markdown("### AI-Powered Classification for Normal vs Tumor kidney Scans")

tab1, tab2, tab3 = st.tabs(["🔬 SINGLE SCAN ANALYSIS", "📑 BATCH MEDICAL REPORT", "🤖 RENAL AI"])

with tab1:
    uploaded_file = st.file_uploader("Upload Medical Scan (JPEG, PNG, DICOM)",type=['jpg', 'jpeg', 'png', 'dcm'], key="single_scan_uploader")
    
    if uploaded_file is not None:
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("File size exceeds the 2MB limit. Please upload a smaller file.")
        else:
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            # Verify if the uploaded file is new. This prevents redundant model inference on the same file during UI reruns.
            if st.session_state.processed_file_id != current_file_id:
                st.session_state.prediction_active = False
                st.session_state.manual_clear = False
                st.session_state.processed_file_id = current_file_id
                
                temp_filename = "temp_scan.png" # Use a consistent temp name
                try:
                    # --- Image Loading and Conversion ---
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'dcm':
                        img = load_dicom_image(uploaded_file.getvalue())
                    else:
                        img = Image.open(uploaded_file)
                    
                    # Save the PIL image to a temporary file for the pipeline
                    img.convert("RGB").save(temp_filename, "PNG")

                    # --- Prediction using the unified pipeline ---
                    with st.status("⚙️ Initializing AI Diagnostic Pipeline...", expanded=True) as status:
                        step1 = st.empty()
                        step1.markdown("⏳ **Phase 1/6:** Loading model and medical scan...")
                        pipeline = PredictionPipeline(filename=temp_filename)
                        step1.markdown("✅ **Phase 1/6:** Loading model and medical scan...")
                        
                        # Run validations from the pipeline
                        step2 = st.empty()
                        step2.markdown("⏳ **Phase 2/6:** Validating image quality & integrity...")
                        pipeline.validate_image_quality()
                        step2.markdown("✅ **Phase 2/6:** Validating image quality & integrity...")
                        
                        step3 = st.empty()
                        step3.markdown("⏳ **Phase 3/6:** Verifying renal CT characteristics...")
                        pipeline.verify_ct_scan()
                        step3.markdown("✅ **Phase 3/6:** Verifying renal CT characteristics...")

                        # Get prediction, Grad-CAM, and Preprocessing previews
                        step4 = st.empty()
                        step4.markdown("⏳ **Phase 4/6:** Running deep learning prediction...")
                        prediction_result = pipeline.predict_detailed()
                        step4.markdown("✅ **Phase 4/6:** Running deep learning prediction...")
                        
                        step5 = st.empty()
                        step5.markdown("⏳ **Phase 5/6:** Generating Grad-CAM visualization...")
                        gradcam_b64 = pipeline.make_gradcam_overlay_base64()
                        step5.markdown("✅ **Phase 5/6:** Generating Grad-CAM visualization...")
                        
                        step6 = st.empty()
                        step6.markdown("⏳ **Phase 6/6:** Finalizing diagnostic previews...")
                        previews_b64 = pipeline.make_preprocess_previews_base64()
                        step6.markdown("✅ **Phase 6/6:** Finalizing diagnostic previews...")

                        status.update(label="Scan Analysis Complete!", state="complete", expanded=False)

                        predicted_label = prediction_result["prediction"]
                        confidence = prediction_result["confidence"]
                        
                        # --- Update Session State ---
                        st.session_state.latest_label = predicted_label
                        st.session_state.latest_confidence = confidence * 100
                        
                        new_entry = {
                            "File Name": uploaded_file.name,
                            "Predicted Label": predicted_label,
                            "Confidence Score": f"{confidence * 100:.2f}%"
                        }
                        st.session_state.history.insert(0, new_entry)
                        st.session_state.history = st.session_state.history[:5]
                        
                        color_indicator = '#D9534F' if predicted_label == 'Tumor' else '#50C878'
                        animation_class = 'danger-animate' if predicted_label == 'Tumor' else 'success-animate'

                        st.session_state.prediction_active = True
                        st.session_state.persisted_label = predicted_label
                        st.session_state.persisted_confidence = confidence
                        st.session_state.persisted_heatmap = gradcam_b64
                        st.session_state.persisted_img = img
                        st.session_state.persisted_preprocessed_img = previews_b64["resized"]
                        st.session_state.persisted_color_indicator = color_indicator
                        st.session_state.persisted_animation_class = animation_class
                        
                        if predicted_label == 'Normal':
                            st.snow()

                except ValueError as ve:
                    # Catch specific validation errors from the pipeline
                    st.error(str(ve))
                    st.session_state.processed_file_id = None
                except Exception as e:
                    if pydicom and isinstance(e, pydicom.errors.InvalidDicomError):
                        st.error("Error: The uploaded DICOM file is corrupted or invalid. Please provide a valid DICOM file.")
                    elif isinstance(e, FileNotFoundError):
                        st.error(f"System Error: {str(e)}")
                    else:
                        st.error(f"An unexpected error occurred: {str(e)}")
                    st.session_state.processed_file_id = None
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
    
            # Render the prediction results and metrics conditionally, relying on the active prediction state.
            if st.session_state.prediction_active:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="metric-container metric-animate {st.session_state.persisted_animation_class}">
                        <div class="metric-title">Diagnostic Prediction</div>
                        <div class="metric-value" style="color: {st.session_state.persisted_color_indicator}">{st.session_state.persisted_label}</div>
                        <div style="margin-top: 10px; font-size: 1.1rem; color: {text_color};">Confidence: <strong style="color: {st.session_state.persisted_color_indicator};">{st.session_state.persisted_confidence * 100:.1f}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f'<h3 style="color: {title_color}; font-weight: bold; text-align: center;">Imaging Analysis</h3>', unsafe_allow_html=True)
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    st.image(st.session_state.persisted_img, width="stretch")
                    st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Original Uploaded Scan</strong></p>', unsafe_allow_html=True)
                with img_col2:
                    st.image(st.session_state.persisted_preprocessed_img, width="stretch")
                    st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Preprocessed (224x224)</strong></p>', unsafe_allow_html=True)
                with img_col3:
                    st.image(st.session_state.persisted_heatmap, width="stretch")
                    st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Grad-CAM Heatmap Analysis</strong></p>', unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # The persisted_heatmap is now a data URL: "data:image/png;base64,..."
                b64_string = st.session_state.persisted_heatmap.split(',')[1]
                image_bytes = base64.b64decode(b64_string)
                
                def clear_single_scan_results():
                    st.session_state.prediction_active = False
                    st.session_state.manual_clear = True
                    st.session_state.latest_label = None
                    st.session_state.latest_confidence = None

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="📥 Download Grad-CAM Overlay",
                        data=image_bytes,
                        file_name="gradcam_overlay.png",
                        mime="image/png"
                    )
                with btn_col2:
                    st.button("🧹 Clear Results", key="clear_results_btn", type="primary", on_click=clear_single_scan_results)
    else:
        # Reset the session state variables when the user clears or removes the uploaded file to prepare for a new session.
        st.session_state.processed_file_id = None
        st.session_state.prediction_active = False
        st.session_state.manual_clear = False
        st.session_state.latest_label = None
        st.session_state.latest_confidence = None
        st.session_state.persisted_preprocessed_img = None

    # --- Render Session History & Metadata Section: Display a table of past predictions within an expander ---
    st.markdown("---")
    with st.expander("📋 Session History "):
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            history_df.index = history_df.index + 1
            st.dataframe(history_df, width="stretch")
        else:
            st.info("No session records available yet.")

with tab2:
    st.markdown(f"""
    <div class="metric-animate" style="background-color: {info_bg}; padding: 15px; border-radius: 10px; border: 1px solid {title_color}; margin-bottom: 20px;">
        <strong style="color: {title_color}; font-size: 1.1rem;">BATCH PROCESSING PROTOCOL:</strong><br><br>
        <span style="color: {text_color};">
        You may upload multiple CT scans simultaneously for high-throughput screening. The system will process each scan sequentially and generate a consolidated diagnostic summary table.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    batch_files = st.file_uploader("Drop multiple renal scans here (JPEG, PNG, DICOM)", type=['jpg', 'jpeg', 'png', 'dcm'], accept_multiple_files=True, key="batch_scan_uploader")
    
    # Compare the current batch of files with the previous one. Clear cached results if a new batch is detected to avoid mixing data.
    current_batch_ids = [f"{b.name}_{b.size}" for b in batch_files] if batch_files else []
    if st.session_state.batch_file_ids != current_batch_ids:
        st.session_state.batch_results = None
        st.session_state.batch_pdf_bytes = None
        st.session_state.batch_csv_bytes = None
        st.session_state.batch_file_ids = current_batch_ids
    
    if batch_files:
        if st.button("Start Batch Processing"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_results = []
            temp_filename = "temp_batch_scan.png"

            for i, b_file in enumerate(batch_files):
                status_text.markdown(f"<span style='color: #007BFF; font-weight: 500;'>⏳ Processing {b_file.name} ({i+1}/{len(batch_files)})...</span>", unsafe_allow_html=True)
                try:
                    # --- Load image (including DICOM) ---
                    file_ext = b_file.name.split('.')[-1].lower()
                    if file_ext == 'dcm':
                        img = load_dicom_image(b_file.getvalue())
                    else:
                        img = Image.open(b_file)
                    
                    # Save to temp file for pipeline
                    img.convert("RGB").save(temp_filename, "PNG")

                    # --- Instantiate and run pipeline ---
                    # The model is loaded once and cached inside the pipeline instance
                    pipeline = PredictionPipeline(filename=temp_filename)
                    pipeline.validate_image_quality() # Run validation per image
                    
                    prediction_result = pipeline.predict_detailed()
                    predicted_label = prediction_result["prediction"]
                    confidence = prediction_result["confidence"]
                    
                    batch_results.append({
                        "File Name": b_file.name,
                        "Detection Label": predicted_label,
                        "Confidence (%)": f"{confidence * 100:.2f}"
                    })
                    st.session_state.latest_label = predicted_label
                    st.session_state.latest_confidence = confidence * 100
                except ValueError as ve:
                    st.warning(f"Skipped {b_file.name}: {str(ve)}")
                except Exception as e:
                    st.warning(f"Failed to process {b_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            # Clean up temp file after loop
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            status_text.markdown("<span style='color: #50C878; font-weight: bold;'>✅ Batch processing complete.</span>", unsafe_allow_html=True)
            
            if batch_results:
                results_df = pd.DataFrame(batch_results)
                st.session_state.batch_results = results_df
                
                # Provide an option to generate and download a comprehensive batch report in PDF format using the 'reportlab' library.
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas
                    from reportlab.lib import colors
                    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    elements = []
                    styles = getSampleStyleSheet()
                    
                    elements.append(Paragraph("Batch Medical Report", styles['Title']))
                    elements.append(Spacer(1, 12))
                    
                    data = [["File Name", "Detection Label", "Confidence (%)"]]
                    for res in batch_results:
                        data.append([res["File Name"], res["Detection Label"], res["Confidence (%)"]])
                        
                    t = Table(data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                        ('GRID', (0,0), (-1,-1), 1, colors.black)
                    ]))
                    elements.append(t)
                    doc.build(elements)

                    st.session_state.batch_pdf_bytes = buffer.getvalue()
                    st.session_state.batch_csv_bytes = None
                    
                except ImportError:
                    st.session_state.batch_csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                    st.session_state.batch_pdf_bytes = None

        # Ensure the batch results dataframe and download buttons remain visible across Streamlit reruns by rendering them from the session state.
        if st.session_state.batch_results is not None:
            st.dataframe(st.session_state.batch_results, width="stretch")
            
            if st.session_state.batch_pdf_bytes is not None:
                st.download_button(
                    label="Download Batch Report as PDF",
                    data=st.session_state.batch_pdf_bytes,
                    file_name="batch_medical_report.pdf",
                    mime="application/pdf"
                )
            elif st.session_state.batch_csv_bytes is not None:
                st.warning("To generate PDF reports, please install 'reportlab' (pip install reportlab). For now, you can download the results as CSV.")
                st.download_button(
                    label="Download Batch Report as CSV",
                    data=st.session_state.batch_csv_bytes,
                    file_name="batch_medical_report.csv",
                    mime="text/csv"
                )

# --- Configure the Medical Advisory Chatbot Assistant: Set up API keys and environment variables required by the Groq SDK ---
# Attempt to load the Groq API key from Streamlit secrets and set it as an environment variable for the SDK.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # Silently handle the case where secrets.toml does not exist

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "YOUR_API_KEY_HERE"

CHATBOT_API_KEY = os.environ["GROQ_API_KEY"]
CHATBOT_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Define the Integrated Medical Advisory Chatbot UI and Logic: Setup the tab interface, warning messages, and chat interactions ---
with tab3:
    st.markdown("<h5 style='text-align: center; color: #6C757D;'>Active Clinical Session: Context-Aware Nephrology Assistant</h5>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Language Support Instruction Box
    st.markdown(f"""
    <div class="metric-animate" style="background-color: {info_bg}; padding: 15px 20px; border-radius: 8px; border-left: 5px solid {title_color}; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <strong style="color: {title_color}; font-size: 1.05rem;">🗣️ Supported Languages / भाषा समर्थन</strong><br>
        <span style="color: {text_color}; font-size: 0.95rem;">
        This Medical Assistant fluently supports both <strong>English</strong> and <strong>Hindi</strong>. Feel free to ask your questions in your preferred language.<br>
        यह मेडिकल असिस्टेंट <strong>अंग्रेजी</strong> और <strong>हिंदी</strong> दोनों भाषाओं का समर्थन करता है। बेझिझक अपनी पसंदीदा भाषा में प्रश्न पूछें।
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💬 Medical Advisory Assistant")

    latest_label = st.session_state.latest_label

    if latest_label == 'Tumor':
        st.markdown(f"""
        <div class="metric-animate danger-animate" style="background-color: {warning_bg}; padding: 15px; border-radius: 10px; border: 1px solid #D9534F; margin-bottom: 20px;">
            <strong style="color: #D9534F; font-size: 1.1rem;">⚠️ Clinical Advisory (Tumor Detected):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Consult a Nephrologist or Oncologist</strong> immediately for a formal diagnosis.<br>
            2. <strong>Recommended further tests</strong>: Consider ordering a biopsy, contrast-enhanced MRI, or PET scan.<br>
            3. Please ensure all medical imaging is formally reviewed by a certified radiologist.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif latest_label == 'Normal':
        st.markdown(f"""
        <div class="metric-animate" style="background-color: {success_bg}; padding: 15px; border-radius: 10px; border: 2px solid #50C878; margin-bottom: 20px;">
            <strong style="color: #50C878; font-size: 1.1rem;">✅ Clinical Advisory (Normal Scan):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Wellness Instructions</strong>: Maintain a healthy diet, stay hydrated, and monitor kidney function annually if the patient is high risk.<br>
            2. Routine follow-up is recommended as per the primary care physician's schedule.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-animate" style="background-color: {info_bg}; padding: 15px; border-radius: 10px; border: 1px solid {title_color}; margin-bottom: 20px;">
                <strong style="color: {title_color}; font-size: 1.1rem;">🔒 System Status: Chatbot Locked</strong><br><br>
            <span style="color: {text_color};">
                No renal scan has been uploaded yet. Please provide the image first in the Analysis tabs to unlock personalized context-aware advisory.
            </span>
        </div>
        """, unsafe_allow_html=True)

    def submit_chat():
        if st.session_state.chat_input_box.strip():
            st.session_state.user_query = st.session_state.chat_input_box
            st.session_state.chat_input_box = ""

    st.markdown("<h5 style='color: #6C757D; margin-top: 15px;'>Feel free to ask your doubts to Renal Vision AI.</h5>", unsafe_allow_html=True)
    st.text_input("Query", key="chat_input_box", placeholder="Type your message here and press Enter...", label_visibility="collapsed", on_change=submit_chat)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 0.85rem; margin-top: 5px;'><em>Renal AI can make mistakes, please double-check it.</em></p>", unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid {title_color}; background-color: {user_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <strong style="color: {title_color};">👤 You:</strong><br><span style="color: {text_color};">{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #50C878; background-color: {assistant_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <strong style="color: #50C878;">🩺 Assistant:</strong><br><span style="color: {assistant_text};">{message['content']}</span>
                """, unsafe_allow_html=True)

        active_chat_placeholder = st.container()

    prompt = st.session_state.user_query

    if prompt:
        st.session_state.user_query = "" # Immediately clear the user query from the session state to prevent duplicate processing on subsequent interactions.
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Inject CSS to create a shimmering loading animation inside the input box
        # and disable further input while the model generates the response.
        st.markdown(f"""
        <style>
        @keyframes shimmerInput {{
            0% {{ background-position: -1000px 0; }}
            100% {{ background-position: 1000px 0; }}
        }}
        [data-testid="stTextInput"] div[data-baseweb="base-input"] {{
            background: linear-gradient(90deg, {container_bg} 25%, rgba(0, 123, 255, 0.2) 50%, {container_bg} 75%) !important;
            background-size: 1000px 100% !important;
            animation: shimmerInput 2s infinite linear !important;
            border-color: #007BFF !important;
            pointer-events: none !important;
        }}
        [data-testid="stTextInput"] input {{
            opacity: 0.5 !important;
            pointer-events: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        with active_chat_placeholder:
            st.markdown(f"""
            <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid {title_color}; background-color: {user_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <strong style="color: {title_color};">👤 You:</strong><br><span style="color: {text_color};">{prompt}</span>
            </div>
            """, unsafe_allow_html=True)
            
            assistant_placeholder = st.empty()
            
            # Display a temporary 'thinking' animation to provide visual feedback to the user while waiting for the LLM API response.
            assistant_placeholder.markdown(f"""
            <div class="chat-animate" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #50C878; background-color: {assistant_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <strong style="color: #50C878;">🩺 Assistant:</strong><br>
                <div class="bouncing-dots" style="margin-top: 8px;">
                    <div class="dot1"></div>
                    <div class="dot2"></div>
                    <div class="dot3"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Handle the AI API integration: Verify if a scan has been processed first, then call the Groq LLM API if a valid key is provided.
        if not latest_label:
            response = "⚠️ Please Provide the Image first. The Medical Advisory Chatbot remains locked until a valid scan is analyzed."
            assistant_placeholder.markdown(f"""
            <div style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #D9534F; background-color: {warning_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <strong style="color: #D9534F;">🔒 Assistant:</strong><br><span style="color: {text_color};">{response}</span>
            </div>
            """, unsafe_allow_html=True)
        elif CHATBOT_API_KEY != "YOUR_API_KEY_HERE":
            try:
                # Initialize the Groq client. The SDK automatically utilizes the 'GROQ_API_KEY' environment variable configured earlier.
                client = Groq()
                
                system_prompt = f"""You are a specialized Nephrology AI Assistant. 
Current Patient Context:
- Diagnostic Prediction: {st.session_state.latest_label}
- Model Confidence: {st.session_state.latest_confidence:.2f}%

Instructions: Answer the user's query based strictly on this clinical context. If the prediction is 'Tumor', tailor your advice around next-step staging, oncological consultations, and lifestyle precautions, while maintaining a supportive but objective medical tone. Do not give a final medical prescription; emphasize clinical correlation.

IMPORTANT: Always conclude your response with the exact following disclaimer:
"Disclaimer: Renal AI can make mistakes, please double-check it." """

                messages = [
                    {"role": "system", "content": system_prompt}
                ]
                for msg in st.session_state.chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
                completion = client.chat.completions.create(
                    model=CHATBOT_MODEL_NAME,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                response = ""
                for chunk in completion:
                    response += chunk.choices[0].delta.content or ""
                    assistant_placeholder.markdown(f"""
                    <div style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #50C878; background-color: {assistant_chat_bg}; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <strong style="color: #50C878;">🩺 Assistant:</strong><br><span style="color: {assistant_text};">{response}✚</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                response = f"Error calling Groq API: {e}"
        else:
            response = "I have provided the standard medical advisory above. Please configure your Groq API key securely in Streamlit Secrets to enable advanced interactive chat capabilities."

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.balloons()

    # --- Implement Chat History Export Functionality: Provide a UI and logic for users to clear or download their chat history ---
    if st.session_state.chat_history:
        st.markdown("---")
        chat_export = "Medical Advisory Assistant - Session Chat History\n"
        chat_export += "="*50 + "\n\n"
        for msg in st.session_state.chat_history:
            role = "Patient/User" if msg["role"] == "user" else "Nephrology AI Assistant"
            chat_export += f"{role}:\n{msg['content']}\n\n"
            
        def confirm_clear():
            st.session_state.show_clear_confirm = True
            
        def execute_clear():
            st.session_state.chat_history = []
            st.session_state.show_clear_confirm = False
            
        def cancel_clear():
            st.session_state.show_clear_confirm = False
            
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.show_clear_confirm:
                st.warning("Are you sure you want to clear the chat?")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.button("✔️ Yes, Clear", key="yes_clear", type="primary", on_click=execute_clear)
                with sc2:
                    st.markdown("<span id='cancel-btn-highlight'></span>", unsafe_allow_html=True)
                    st.button("❌ Cancel", key="cancel_clear", on_click=cancel_clear)
            else:
                st.button("Clear Chat History", on_click=confirm_clear)
        with col2:
            export_bytes, mime_type, file_ext = _render_chat_history_txt_bytes(
                st.session_state.chat_history, 
                prediction=st.session_state.get('latest_label'),
                confidence=st.session_state.get('latest_confidence')
            )
            if export_bytes:
                st.download_button(
                    label="📥 Download Conversation History",
                    data=export_bytes,
                    file_name=f"renal_ai_chat_history.{file_ext}",
                    mime=mime_type
                )

# --- Render Application Footer: Display the developer credits at the bottom of the page ---
st.markdown(f"""
<div style="display: flex; justify-content: center; margin-top: 50px; margin-bottom: 20px;">
    <div style="background-color: {footer_bg}; color: {footer_text}; padding: 10px 30px; border-radius: 20px; font-weight: bold; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Made with ❤️ by Rohit.
    </div>
</div>
>>>>>>> e3ae6845b1340cd8a1abe4e46f4aab4f0dcbdb8c
""", unsafe_allow_html=True)