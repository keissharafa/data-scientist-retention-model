import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI PAGE ---
st.set_page_config(
    page_title="Job Change Predictor",
    page_icon="💼",
    layout="wide"
)

# --- FUNGSI LOAD ARTEFAK & ENCODER FALLBACK ---
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

# --- MAPPING TEKS MENTAH DARI DATASET ---
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

# --- PRE-FIT STRING ENCODERS UNTUK MENJAMIN MAPPING TEKS -> ANGKA ---
@st.cache_resource
def build_string_encoders():
    string_encoders = {}
    for col, vals in OPTIONS.items():
        le = LabelEncoder()
        le.fit(sorted(vals))
        string_encoders[col] = le
    return string_encoders

def main():
    st.title("💼 Data Scientist Job Change Predictor")
    st.markdown("Aplikasi ini memprediksi apakah kandidat akan **bertahan (Retain)** atau **mencari pekerjaan baru (Leave)** menggunakan model Support Vector Machine (SVM).")
    st.divider()

    try:
        model, scaler, encoders = load_artifacts()
        string_encoders = build_string_encoders()
    except FileNotFoundError as e:
        st.error(f"⚠️ File tidak ditemukan: {e}. Pastikan file model_svm.pkl, scaler.pkl, dan encoders.pkl ada di folder assets.")
        return

    st.subheader("📝 Masukkan Data Kandidat")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**1. Data Numerik & Geografis**")
            city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.8)
            training_hours = st.number_input("Training Hours", min_value=1, max_value=500, value=50)
            
            # DITAMPILKAN DALAM FORMAT TEKS MENTAH DESKRIPTIF
            city = st.selectbox("City", OPTIONS["city"])
            company_size = st.selectbox("Company Size", OPTIONS["company_size"])
            company_type = st.selectbox("Company Type", OPTIONS["company_type"])
            gender = st.selectbox("Gender", OPTIONS["gender"])
            
        with col2:
            st.markdown("**2. Latar Belakang Pendidikan & Pekerjaan**")
            enrolled_university = st.selectbox("Enrolled University", OPTIONS["enrolled_university"])
            education_level = st.selectbox("Education Level", OPTIONS["education_level"])
            relevent_experience = st.selectbox("Relevant Experience", OPTIONS["relevent_experience"])
            last_new_job = st.selectbox("Last New Job", OPTIONS["last_new_job"])
            experience = st.selectbox("Experience", OPTIONS["experience"])
            major_discipline = st.selectbox("Major Discipline", OPTIONS["major_discipline"])

        submitted = st.form_submit_button("Jalankan Prediksi 🚀", type="primary", use_container_width=True)

    if submitted:
        input_dict = {
            "city_development_index": city_development_index,
            "training_hours": training_hours,
            "city": city,
            "company_size": company_size,
            "company_type": company_type,
            "gender": gender,
            "enrolled_university": enrolled_university,
            "education_level": education_level,
            "relevent_experience": relevent_experience,
            "last_new_job": last_new_job,
            "experience": experience,
            "major_discipline": major_discipline
        }
        
        input_df = pd.DataFrame([input_dict])
        
        try:
            categorical_cols = [
                "city", "company_size", "company_type", "gender", 
                "enrolled_university", "education_level", "relevent_experience", 
                "last_new_job", "experience", "major_discipline"
            ]
            
            # --- PROSES ENCODING: TEKS DESKRIPTIF -> ANGKA MODEL ---
            for col in categorical_cols:
                val = str(input_df[col].iloc[0])
                encoded_val = None
                
                # Coba encoder bawaan jika mendukung input string
                if col in encoders and hasattr(encoders[col], 'transform'):
                    try:
                        encoded_val = encoders[col].transform([val])[0]
                    except Exception:
                        pass
                
                # Fallback ke string_encoders jika encoders bawaan gagal/berisi angka
                if encoded_val is None:
                    encoded_val = string_encoders[col].transform([val])[0]
                
                input_df[col] = encoded_val
                
            # 1. Menyelaraskan urutan kolom dengan scaler (12 kolom)
            input_df = input_df[scaler.feature_names_in_]
            
            # 2. Scaling
            input_scaled = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled, columns=scaler.feature_names_in_)
            
            # 3. Filter 9 kolom fitur terbaik
            selected_features = [
                "relevent_experience", "enrolled_university", "education_level", 
                "company_size", "company_type", "last_new_job", "city", 
                "city_development_index", "training_hours"
            ]
            final_input = input_scaled_df[selected_features]
            
            # 4. Prediksi SVM
            prediction = model.predict(final_input)
            
            # Tampilkan Hasil Prediksi
            st.divider()
            st.subheader("Hasil Analisis SVM:")
            if prediction[0] == 1:
                st.error("🚨 **Kandidat Terindikasi Akan Mencari Pekerjaan Baru (Leave)**")
            else:
                st.success("✅ **Kandidat Kemungkinan Besar Akan Bertahan (Retain)**")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

if __name__ == "__main__":
    main()