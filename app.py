"""
🥔 Potato Chip AI — Quality Monitoring System
==============================================
A Streamlit dashboard for potato chip quality control with:
  - Tab 1: CNN-based defect detection from chip images
  - Tab 2: Random Forest process risk prediction from fryer parameters

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import os
from PIL import Image

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Potato Chip AI — Quality Monitor",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Dark UI ──────────────────────────────────────────

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 107, 53, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #FF6B35;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    }

    /* Result boxes */
    .result-box {
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .result-pass {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
    }
    .result-fail {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
    }
    .result-box h2 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .result-box p {
        font-size: 1.2rem;
        color: #e2e8f0;
    }

    /* Risk gauge */
    .risk-low {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
    }
    .risk-medium {
        background: linear-gradient(135deg, #78350f, #92400e);
        border: 2px solid #f59e0b;
    }
    .risk-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }

    /* Slider labels */
    .slider-label {
        font-size: 0.95rem;
        color: #a0aec0;
        margin-bottom: 0.2rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🥔 Potato Chip AI</h1>
    <p>Intelligent Quality Monitoring System — Powered by Deep Learning & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏭 About")
    st.markdown("""
    This system uses AI to monitor potato chip quality at two levels:

    **🔍 Defect Detection**
    A CNN analyzes chip images to detect visual defects like burns, cracks, and discoloration.

    **⚙️ Process Risk**
    A Random Forest model predicts defect rates from fryer process parameters.
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    | Model | Type |
    |---|---|
    | CNN | Binary Classifier |
    | Random Forest | Regressor |
    """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
        "Built with ❤️ using TensorFlow & scikit-learn"
        "</p>",
        unsafe_allow_html=True
    )

# ─── Load Models ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_cnn_model():
    """Load the trained CNN model for defect detection."""
    model_path = os.path.join("models", "cnn_model.h5")
    if not os.path.exists(model_path):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

@st.cache_resource
def load_rf_model():
    """Load the trained Random Forest model for process risk prediction."""
    model_path = os.path.join("models", "rf_model.pkl")
    if not os.path.exists(model_path):
        return None
    try:
        import joblib
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading RF model: {e}")
        return None

cnn_model = load_cnn_model()
rf_model = load_rf_model()

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Image Check", "📷 Live Webcam", "⚙️ Process Risk"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CNN Defect Detection
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 🔍 Chip Defect Detection")
    st.markdown("Upload a potato chip image and the CNN model will classify it as **defective** or **non-defective**.")

    if cnn_model is None:
        st.warning(
            "⚠️ **CNN model not found.** Please run `notebooks/01_cnn_model.ipynb` first "
            "to train and save the model to `models/cnn_model.h5`."
        )
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a potato chip image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="chip_uploader",
            help="Upload a clear image of a potato chip for quality analysis"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Chip Image", width="stretch")

    with col2:
        st.markdown("#### 📊 Analysis Result")

        if uploaded_file is not None and cnn_model is not None:
            # Preprocess the image to match training input
            import tensorflow as tf

            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0      # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict
            with st.spinner("🔄 Analyzing chip quality..."):
                prediction = cnn_model.predict(img_array, verbose=0)
                confidence = float(prediction[0][0])

            # The CNN outputs sigmoid probability:
            # Values closer to 1 → class index 1, closer to 0 → class index 0
            # Class mapping depends on alphabetical folder order:
            #   typically 0 = "defective", 1 = "non-defective" (or vice versa)
            # We check if "defect" maps to 0 (most common alphabetical order)

            is_defective = confidence < 0.5
            conf_score = (1 - confidence) if is_defective else confidence

            if is_defective:
                st.markdown(f"""
                <div class="result-box result-fail">
                    <h2>❌ Defective</h2>
                    <p>This chip has been identified as <strong>defective</strong></p>
                    <p style="font-size: 2.5rem; font-weight: 800; color: #fca5a5;">
                        {conf_score:.1%}
                    </p>
                    <p>Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-pass">
                    <h2>✅ Non-Defective</h2>
                    <p>This chip passes quality inspection</p>
                    <p style="font-size: 2.5rem; font-weight: 800; color: #6ee7b7;">
                        {conf_score:.1%}
                    </p>
                    <p>Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown("**Confidence Breakdown:**")
            st.progress(conf_score, text=f"{'Defective' if is_defective else 'Non-Defective'}: {conf_score:.1%}")

        elif uploaded_file is not None and cnn_model is None:
            st.info("Load the CNN model to see predictions.")
        else:
            st.markdown(
                "<div class='metric-card'>"
                "<p style='color: #64748b; font-size: 1.1rem;'>👆 Upload an image to get started</p>"
                "</div>",
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Live Webcam Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 📷 Live Webcam Analysis")
    st.markdown("Use your MacBook's webcam to analyze potato chips in real-time.")
    
    if cnn_model is None:
        st.warning("⚠️ **CNN model not found.** Please run `notebooks/01_cnn_model.ipynb` first.")
    else:
        # Metrics layout
        m_col1, m_col2, m_col3 = st.columns(3)
        metrics_total = m_col1.empty()
        metrics_defective = m_col2.empty()
        metrics_rate = m_col3.empty()
        
        # Session state initialization
        if 'total_chips' not in st.session_state:
            st.session_state.total_chips = 0
        if 'defective_chips' not in st.session_state:
            st.session_state.defective_chips = 0
            
        # Initialize metrics display
        rate = (st.session_state.defective_chips / max(1, st.session_state.total_chips)) * 100 if st.session_state.total_chips > 0 else 0.0
        metrics_total.metric("Total Analyzed", st.session_state.total_chips)
        metrics_defective.metric("Defective", st.session_state.defective_chips)
        metrics_rate.metric("Defect Rate", f"{rate:.1f}%")
        
        st.markdown("---")
        
        # Streamlit's built in camera input makes capturing perfect images manual and robust
        st.markdown("**Instructions:** Hold the chip up to the camera. The AI will attempt to automatically find the chip and ignore everything else!")
        camera_photo = st.camera_input("Take Photo")

        if camera_photo is not None:
            # Load the image from the camera
            image = Image.open(camera_photo).convert("RGB")
            
            # --- CRITICAL FIX: Smart Auto-Crop ---
            import cv2
            cv_image = np.array(image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # 1. Black out human faces so we don't accidentally track them!
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                cv_image[fy:fy+fh, fx:fx+fw] = 0  # Erase the face from the image
                
            # 2. Find potato-chip-colored objects (yellow/gold/brown)
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_chip = np.array([10, 50, 50])
            upper_chip = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_chip, upper_chip)
            
            # Clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            extracted = False
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 800:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    padding = 50
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.width, x + w + padding)
                    y2 = min(image.height, y + h + padding)
                    cropped_image = image.crop((x1, y1, x2, y2))
                    extracted = True
            
            if not extracted:
                # Fallback Center Crop
                width, height = image.size
                crop_size = int(min(width, height) * 0.4)
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2
                right = (width + crop_size) // 2
                bottom = (height + crop_size) // 2
                cropped_image = image.crop((left, top, right, bottom))
            
            # Preprocess the cropped image to match CNN training
            img = cropped_image.resize((224, 224))
            img_array = np.array(img) / 255.0      # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict
            with st.spinner("🔄 Analyzing chip quality..."):
                prediction = cnn_model.predict(img_array, verbose=0)
                confidence = float(prediction[0][0])
            
            is_defective = confidence < 0.5
            conf_score = (1 - confidence) if is_defective else confidence
            
            st.session_state.total_chips += 1
            if is_defective:
                st.session_state.defective_chips += 1
                st.markdown(f"""
                <div class="result-box result-fail">
                    <h2>❌ Defective</h2>
                    <p>Confidence: {conf_score:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-pass">
                    <h2>✅ Non-Defective (Pass)</h2>
                    <p>Confidence: {conf_score:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Update metrics directly
            rate = (st.session_state.defective_chips / st.session_state.total_chips) * 100
            metrics_total.metric("Total Analyzed", st.session_state.total_chips)
            metrics_defective.metric("Defective", st.session_state.defective_chips)
            metrics_rate.metric("Defect Rate", f"{rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Random Forest Process Risk Predictor
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### ⚙️ Process Risk Predictor")
    st.markdown("Adjust fryer process parameters to predict the expected **defect percentage**.")

    if rf_model is None:
        st.warning(
            "⚠️ **Random Forest model not found.** Please run `notebooks/02_rf_model.ipynb` first "
            "to train and save the model to `models/rf_model.pkl`."
        )

    col_sliders, col_result = st.columns([1, 1], gap="large")

    with col_sliders:
        st.markdown("#### 🎛️ Process Parameters")

        st.markdown('<p class="slider-label">🌡️ Fryer Temperature (°C)</p>', unsafe_allow_html=True)
        fryer_temp = st.slider(
            "Fryer Temperature",
            min_value=150.0, max_value=200.0, value=175.0, step=0.5,
            key="fryer_temp",
            label_visibility="collapsed"
        )

        st.markdown('<p class="slider-label">⏱️ Frying Time (minutes)</p>', unsafe_allow_html=True)
        frying_time = st.slider(
            "Frying Time",
            min_value=2.0, max_value=6.0, value=4.0, step=0.1,
            key="frying_time",
            label_visibility="collapsed"
        )

        st.markdown('<p class="slider-label">💧 Moisture Content (%)</p>', unsafe_allow_html=True)
        moisture = st.slider(
            "Moisture Content",
            min_value=1.0, max_value=8.0, value=4.5, step=0.1,
            key="moisture",
            label_visibility="collapsed"
        )

        st.markdown('<p class="slider-label">📏 Slice Thickness (mm)</p>', unsafe_allow_html=True)
        thickness = st.slider(
            "Slice Thickness",
            min_value=1.0, max_value=3.0, value=2.0, step=0.1,
            key="thickness",
            label_visibility="collapsed"
        )

        st.markdown('<p class="slider-label">🛢️ Oil Quality Index (0-1)</p>', unsafe_allow_html=True)
        oil_quality = st.slider(
            "Oil Quality Index",
            min_value=0.5, max_value=1.0, value=0.75, step=0.01,
            key="oil_quality",
            label_visibility="collapsed"
        )

        predict_btn = st.button(
            "🚀 Predict Defect Rate",
            use_container_width=True,
            type="primary"
        )

    with col_result:
        st.markdown("#### 📊 Prediction Result")

        if predict_btn and rf_model is not None:
            import pandas as pd

            # Prepare input as DataFrame (matches training feature order)
            input_data = pd.DataFrame({
                'fryer_temperature': [fryer_temp],
                'frying_time': [frying_time],
                'moisture_content': [moisture],
                'slice_thickness': [thickness],
                'oil_quality_index': [oil_quality]
            })

            # Predict
            predicted_defect = float(rf_model.predict(input_data)[0])
            predicted_defect = np.clip(predicted_defect, 0, 100)

            # Determine risk level
            if predicted_defect < 20:
                risk_level = "Low"
                risk_class = "risk-low"
                risk_emoji = "🟢"
                risk_color = "#10b981"
            elif predicted_defect < 50:
                risk_level = "Medium"
                risk_class = "risk-medium"
                risk_emoji = "🟡"
                risk_color = "#f59e0b"
            else:
                risk_level = "High"
                risk_class = "risk-high"
                risk_emoji = "🔴"
                risk_color = "#ef4444"

            # Display result
            st.markdown(f"""
            <div class="result-box {risk_class}">
                <h2>{risk_emoji} {risk_level} Risk</h2>
                <p style="font-size: 3rem; font-weight: 800; color: {risk_color};">
                    {predicted_defect:.1f}%
                </p>
                <p>Predicted Defect Rate</p>
            </div>
            """, unsafe_allow_html=True)

            # Show input summary
            st.markdown("**Input Summary:**")
            summary_data = {
                "Parameter": ["🌡️ Temperature", "⏱️ Frying Time", "💧 Moisture",
                              "📏 Thickness", "🛢️ Oil Quality"],
                "Value": [f"{fryer_temp}°C", f"{frying_time} min", f"{moisture}%",
                          f"{thickness} mm", f"{oil_quality}"],
            }
            st.table(summary_data)

            # Recommendations
            st.markdown("**💡 Recommendations:**")
            if fryer_temp > 185:
                st.warning("🌡️ Temperature is high — consider reducing to below 185°C to lower defect rate.")
            if frying_time > 4.5:
                st.info("⏱️ Frying time is above optimal — shorter frying may improve quality.")
            if oil_quality < 0.65:
                st.warning("🛢️ Oil quality is low — consider replacing fryer oil.")
            if moisture < 2.5:
                st.info("💧 Low moisture content — raw material may need conditioning.")
            if predicted_defect < 15:
                st.success("✅ Process parameters look great! Low defect rate expected.")

        elif predict_btn and rf_model is None:
            st.info("Load the RF model to see predictions.")
        else:
            st.markdown(
                "<div class='metric-card'>"
                "<p style='color: #64748b; font-size: 1.1rem;'>👈 Adjust parameters and click <strong>Predict</strong></p>"
                "</div>",
                unsafe_allow_html=True
            )

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #475569; font-size: 0.85rem;'>"
    "🥔 Potato Chip AI v1.0 — Quality Monitoring System"
    "</p>",
    unsafe_allow_html=True
)
