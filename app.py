import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI PAGE ---
st.set_page_config(
    page_title="Job Change Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS INJECTION: ULTIMATE AURORA GLASSMORPHISM (ANTI DARK/LIGHT MODE) ---
def apply_custom_css():
    st.markdown("""
    <style>
        /* 1. BACKGROUND AURORA (Gradient Halus - MEMAKSA TETAP TERANG) */
        .stApp {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }
        header {background-color: transparent !important;}
        footer {visibility: hidden !important;}

        /* 2. MAIN HEADER STYLING */
        .main-header {
            font-size: 3.5rem;
            font-weight: 900;
            background: -webkit-linear-gradient(45deg, #1e293b, #4f46e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: -15px;
            letter-spacing: -1px;
            text-shadow: 2px 4px 10px rgba(0,0,0,0.05);
        }
        .sub-header {
            font-size: 1.15rem;
            color: #334155 !important;
            font-weight: 600;
            margin-bottom: 2rem;
        }

        /* 3. FORM CONTAINER (Kaca Buram Asli) */
        [data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.4) !important;
            border-radius: 30px !important;
            border: 1px solid rgba(255, 255, 255, 0.7) !important;
            box-shadow: 0 20px 50px rgba(31, 38, 135, 0.15) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            padding: 3rem !important;
        }

        /* 4. INPUT FIELDS (Selectbox & NumberInput bening) */
        /* Target spesifik untuk menembus Dark Mode BaseWeb UI */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.5) !important;
            border-radius: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            color: #0f172a !important; /* Memaksa teks selalu gelap */
            fill: #0f172a !important; /* Warna icon panah dropdown */
        }
        
        /* Memaksa teks di dalam input selalu terbaca (Anti-hitam) */
        div[data-baseweb="select"] span, div[data-baseweb="input"] input {
            color: #0f172a !important;
            font-weight: 700 !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        /* Efek Hover Input */
        div[data-baseweb="select"] > div:hover, 
        div[data-baseweb="input"] > div:hover {
            background-color: rgba(255, 255, 255, 0.8) !important;
            border-color: #ffffff !important;
        }

        /* 5. DROPDOWN MENU POPOVER (Daftar saat diklik) */
        ul[role="listbox"], div[data-baseweb="popover"] > div {
            background-color: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(25px) !important;
            -webkit-backdrop-filter: blur(25px) !important;
            border: 1px solid rgba(255, 255, 255, 1) !important;
            border-radius: 15px !important;
        }
        ul[role="listbox"] li {
            color: #1e293b !important;
            font-weight: 600 !important;
            background-color: transparent !important;
        }
        ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.2) !important;
            color: #4f46e5 !important;
        }

        /* 6. LABELS & TEXT (Memaksa warna label selalu kontras) */
        .stSelectbox label p, .stNumberInput label p, .stSlider label p, .stMarkdown p {
            color: #1e293b !important;
            font-weight: 800 !important;
            font-size: 0.95rem !important;
        }
        .section-title {
            color: #0f172a !important;
            font-size: 1.4rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            border-bottom: 3px solid rgba(255, 255, 255, 0.6);
            display: inline-block;
            padding-bottom: 5px;
        }

        /* 7. ELEGANT GRADIENT BUTTON (Menimpa Light Mode Glitch) */
        div.stButton > button, div.stButton > button:focus {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            border-radius: 50px !important;
            border: none !important;
            font-weight: 800 !important;
            font-size: 1.2rem !important;
            padding: 0.75rem 2rem !important;
            box-shadow: 0 10px 20px rgba(118, 75, 162, 0.3) !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        div.stButton > button:hover {
            transform: translateY(-3px) scale(1.01);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            box-shadow: 0 15px 30px rgba(118, 75, 162, 0.5) !important;
            color: #ffffff !important;
            border: none !important;
        }
        
        /* 8. FIX SLIDER UI (Mengubah garis merah menjadi gradien ungu) */
        div[data-testid="stSliderTickBarMin"] {
            background: linear-gradient(90deg, #667eea, #764ba2) !important;
        }
        div[data-testid="stSliderThumb"] {
            background-color: #764ba2 !important;
            border-color: #ffffff !important;
            box-shadow: 0 0 10px rgba(118, 75, 162, 0.5) !important;
        }
        /* Hapus background default slider */
        .stSlider [data-baseweb="slider"] {
            background-color: transparent !important;
        }
        .stSlider div[data-testid="stSliderTickBar"] {
            background-color: rgba(255,255,255, 0.5) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD ARTEFAK ---
@st.cache_resource
def load_artifacts():
    model_path = os.path.join('assets', 'model_svm.pkl') 
    scaler_path = os.path.join('assets', 'scaler.pkl')
    encoder_path = os.path.join('assets', 'encoders.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
        
    return model, scaler, encoders

# --- MAPPING TEKS MENTAH ---
OPTIONS = {
    "city": [f"city_{i}" for i in [103, 40, 21, 115, 162, 176, 160, 46, 61, 114, 13, 159, 102, 67, 100, 16, 71, 104, 64, 101, 83, 105, 73, 75, 41, 11, 93, 90, 36, 20, 57, 152, 19, 65, 74, 173, 136, 98, 97, 50, 138, 82, 157, 89, 150, 70, 175, 94, 28, 59, 165, 145, 142, 26, 12, 37, 43, 116, 23, 99, 149, 10, 45, 80, 128, 158, 123, 7, 72, 106, 143, 78, 109, 24, 134, 48, 144, 91, 146, 133, 126, 118, 9, 167, 27, 84, 54, 39, 79, 76, 77, 81, 131, 44, 117, 155, 33, 141, 127, 62, 53, 25, 2, 69, 120, 111, 30, 1, 140, 179, 55, 14, 42, 107, 18, 139, 180, 166, 121, 129, 8, 31, 171]],
    "company_size": ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"],
    "company_type": ["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Other", "Public Sector", "NGO"],
    "gender": ["Male", "Female", "Other"],
    "enrolled_university": ["no_enrollment", "Full time course", "Part time course"],
    "education_level": ["Graduate", "Masters", "High School", "Phd", "Primary School"],
    "relevent_experience": ["Has relevent experience", "No relevent experience"],
    "last_new_job": ["1", "2", "3", "4", ">4", "never"],
    "experience": ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20"],
    "major_discipline": ["STEM", "Humanities", "Other", "Business Degree", "Arts", "No Major"]
}

@st.cache_resource
def build_string_encoders():
    string_encoders = {}
    for col, vals in OPTIONS.items():
        le = LabelEncoder()
        le.fit(sorted(vals))
        string_encoders[col] = le
    return string_encoders

def main():
    # Suntikkan CSS Masterpiece
    apply_custom_css()

    st.markdown('<div class="main-header">Predict Candidate Movement</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered SVM analysis to predict employee retention.</div>', unsafe_allow_html=True)
    st.write("") 

    try:
        model, scaler, encoders = load_artifacts()
        string_encoders = build_string_encoders()
    except FileNotFoundError as e:
        st.error(f"⚠️ File artefak ML tidak ditemukan. Pastikan assets/model_svm.pkl, dll tersedia.")
        return

    with st.form("prediction_form"):
        col1, col_spacer, col2 = st.columns([1, 0.1, 1])
        
        with col1:
            st.markdown('<div class="section-title">📊 Demographics & Company</div>', unsafe_allow_html=True)
            
            city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.8)
            training_hours = st.number_input("Training Hours", min_value=1, max_value=500, value=50)
            city = st.selectbox("City", OPTIONS["city"])
            company_size = st.selectbox("Company Size", OPTIONS["company_size"])
            company_type = st.selectbox("Company Type", OPTIONS["company_type"])
            gender = st.selectbox("Gender", OPTIONS["gender"])
            
        with col2:
            st.markdown('<div class="section-title">🎓 Experience & Education</div>', unsafe_allow_html=True)
            
            enrolled_university = st.selectbox("Enrolled University", OPTIONS["enrolled_university"])
            education_level = st.selectbox("Education Level", OPTIONS["education_level"])
            relevent_experience = st.selectbox("Relevant Experience", OPTIONS["relevent_experience"])
            last_new_job = st.selectbox("Last New Job", OPTIONS["last_new_job"])
            experience = st.selectbox("Years of Experience", OPTIONS["experience"])
            major_discipline = st.selectbox("Major Discipline", OPTIONS["major_discipline"])

        st.write("") 
        submitted = st.form_submit_button("Analyze Candidate ✨", use_container_width=True)

    if submitted:
        input_dict = {
            "city_development_index": city_development_index, "training_hours": training_hours,
            "city": city, "company_size": company_size, "company_type": company_type, "gender": gender,
            "enrolled_university": enrolled_university, "education_level": education_level,
            "relevent_experience": relevent_experience, "last_new_job": last_new_job,
            "experience": experience, "major_discipline": major_discipline
        }
        
        input_df = pd.DataFrame([input_dict])
        
        try:
            categorical_cols = [
                "city", "company_size", "company_type", "gender", 
                "enrolled_university", "education_level", "relevent_experience", 
                "last_new_job", "experience", "major_discipline"
            ]
            
            for col in categorical_cols:
                val = str(input_df[col].iloc[0])
                encoded_val = None
                
                if col in encoders and hasattr(encoders[col], 'transform'):
                    try:
                        encoded_val = encoders[col].transform([val])[0]
                    except Exception:
                        pass
                
                if encoded_val is None:
                    encoded_val = string_encoders[col].transform([val])[0]
                
                input_df[col] = encoded_val
                
            input_df = input_df[scaler.feature_names_in_]
            input_scaled = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled, columns=scaler.feature_names_in_)
            
            selected_features = [
                "relevent_experience", "enrolled_university", "education_level", 
                "company_size", "company_type", "last_new_job", "city", 
                "city_development_index", "training_hours"
            ]
            final_input = input_scaled_df[selected_features]
            
            prediction = model.predict(final_input)
            
            st.write("")
            if prediction[0] == 1:
                st.error("### 🚨 High Risk of Leaving\nModel mendeteksi kandidat ini memiliki probabilitas tinggi untuk berpindah pekerjaan (Leave).")
            else:
                st.success("### ✅ Likely to Retain\nModel mendeteksi kandidat ini cenderung akan bertahan di perusahaan (Retain).")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pemrosesan: {e}")

if __name__ == "__main__":
    main()