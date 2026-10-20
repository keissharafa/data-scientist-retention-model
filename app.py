import streamlit as st
import pandas as pd
import pickle
import os

# --- KONFIGURASI TATA LETAK UI ---
st.set_page_config(
    page_title="Job Change Predictor",
    page_icon="💼",
    layout="wide"
)

# --- FUNGSI LOAD ARTEFAK ---
@st.cache_resource
def load_artifacts():
    # Nama file SVM sudah disesuaikan dengan milik Fachri
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

def main():
    st.title("💼 Data Scientist Job Change Predictor")
    st.markdown("Aplikasi ini memprediksi apakah kandidat akan **bertahan (Retain)** atau **mencari pekerjaan baru (Leave)** menggunakan model Support Vector Machine (SVM).")
    st.divider()

    # Memuat Artefak
    try:
        model, scaler, encoders = load_artifacts()
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
            
            city = st.selectbox("City", encoders['city'].classes_)
            company_size = st.selectbox("Company Size", encoders['company_size'].classes_)
            company_type = st.selectbox("Company Type", encoders['company_type'].classes_)
            
            # --- FITUR YANG DITAMBAHKAN ---
            gender = st.selectbox("Gender", encoders['gender'].classes_) 
            
        with col2:
            st.markdown("**2. Latar Belakang Pendidikan & Pekerjaan**")
            enrolled_university = st.selectbox("Enrolled University", encoders['enrolled_university'].classes_)
            education_level = st.selectbox("Education Level", encoders['education_level'].classes_)
            relevent_experience = st.selectbox("Relevant Experience", encoders['relevent_experience'].classes_)
            last_new_job = st.selectbox("Last New Job", encoders['last_new_job'].classes_)
            
            # --- FITUR YANG DITAMBAHKAN ---
            experience = st.selectbox("Experience", encoders['experience'].classes_)
            major_discipline = st.selectbox("Major Discipline", encoders['major_discipline'].classes_)

        submitted = st.form_submit_button("Jalankan Prediksi 🚀", type="primary", use_container_width=True)

    if submitted:
        # Masukkan SELURUH 12 fitur (termasuk 3 yang baru) ke dalam Dictionary
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
            # Tambahkan 3 fitur baru ke dalam proses Encoding
            categorical_cols = [
                "city", "company_size", "company_type", "gender", 
                "enrolled_university", "education_level", "relevent_experience", 
                "last_new_job", "experience", "major_discipline"
            ]
            for col in categorical_cols:
                input_df[col] = encoders[col].transform(input_df[col])
                
            # 1. Pastikan urutan 12 kolom sama dengan scaler (DITARUH DI LUAR FOR LOOP)
            input_df = input_df[scaler.feature_names_in_]
            
            # 2. Proses Scaling (menghasilkan array 12 kolom)
            input_scaled = scaler.transform(input_df)
            
            # 3. Ubah kembali jadi DataFrame agar mudah disaring
            input_scaled_df = pd.DataFrame(input_scaled, columns=scaler.feature_names_in_)
            
            # 4. SELEKSI FITUR: Hanya ambil 9 kolom sesuai selected_features.json
            selected_features = [
                "relevent_experience", "enrolled_university", "education_level", 
                "company_size", "company_type", "last_new_job", "city", 
                "city_development_index", "training_hours"
            ]
            final_input = input_scaled_df[selected_features]
            
            # 5. Prediksi dengan Model SVM (masukkan final_input yang berisi 9 fitur)
            prediction = model.predict(final_input)
            
            # Tampilkan Hasil
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