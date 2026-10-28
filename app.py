import streamlit as st
import pandas as pd
import pickle
import os
import html
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI PAGE ---
st.set_page_config(
    page_title="Job Change Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS INJECTION: ULTIMATE AURORA GLASSMORPHISM & PILL TABS ---
def apply_custom_css():
    st.markdown("""
    <style>
        /* Import Font Plus Jakarta Sans dari Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        /* 1. BACKGROUND AURORA (Gradient Halus) */
        .stApp {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        header {background-color: transparent !important;}
        footer {visibility: hidden !important;}

        /* 2. MAIN HEADER STYLING */
        .main-header {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 3.5rem;
            font-weight: 800 !important; /* ExtraBold */
            background: -webkit-linear-gradient(45deg, #1e293b, #4f46e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: -15px;
            letter-spacing: -1px;
            text-shadow: 2px 4px 10px rgba(0,0,0,0.05);
        }
        .sub-header {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 1.15rem;
            color: #334155 !important;
            font-weight: 500 !important; /* Medium */
            margin-bottom: 2rem;
        }

        /* 3. CAPSULE TABS NAVIGASI (Gaya Pill Terpisah) */
        div[data-baseweb="tab-list"] {
            border-bottom: none !important;
            gap: 18px !important;
            margin-bottom: 20px;
            justify-content: center !important;
        }
        div[data-baseweb="tab-border"] {
            display: none !important;
        }
        div[data-baseweb="tab"] p {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            color: inherit !important;
        }
        div[data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.5) !important;
            border-radius: 50px !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
            color: #64748b !important;
            font-weight: 600 !important;
            padding: 14px 44px !important;
            height: auto !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
            transition: all 0.3s ease;
        }
        div[data-baseweb="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.8) !important;
            transform: translateY(-2px);
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3) !important;
        }
        div[data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* ========================================================
           4. METRIC CARDS (GLASSMORPHISM MURNI DENGAN CORNER)
           ======================================================== */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.45) !important;
            border-radius: 15px !important;
            padding: 15px 25px !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
        }
        div[data-testid="stMetric"] > div { 
            background: transparent !important; 
        }
        
        [data-testid="stMetricLabel"] { 
            font-weight: 700; 
            color: #475569 !important; 
            font-size: 1rem; 
        }
        [data-testid="stMetricValue"] { 
            color: #0f172a !important; 
            font-weight: 900; 
            font-size: 2.2rem !important; 
        }

        /* 5. FORM CONTAINER (Kaca Buram) */
        [data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.4) !important;
            border-radius: 25px !important;
            border: 1px solid rgba(255, 255, 255, 0.8) !important;
            box-shadow: 0 15px 35px rgba(31, 38, 135, 0.05) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            padding: 2.5rem !important;
            margin-top: 1.5rem;
        }

        /* 6. INPUT FIELDS */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            color: #0f172a !important; 
        }
        div[data-baseweb="select"] span, div[data-baseweb="input"] input {
            color: #0f172a !important;
            font-weight: 600 !important;
        }
        .stSelectbox label p, .stNumberInput label p, .stSlider label p {
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
        }

        /* 7. PROGRESS BAR HORIZONTAL */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #818cf8, #c084fc) !important;
            border-radius: 50px;
        }
        
        /* 8. BUTTON STYLING */
        div.stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            border-radius: 50px !important;
            border: none !important;
            font-weight: 800 !important;
            padding: 0.5rem 2rem !important;
            box-shadow: 0 10px 20px rgba(118, 75, 162, 0.3) !important;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(118, 75, 162, 0.5) !important;
        }

        /* ========================================================
           9. RESULT CARD (tint gradient di kiri, ikon di kotak putih)
           ======================================================== */
        .result-card {
            position: relative;
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 20px 24px;
            margin: 0.5rem 0 1.25rem 0;
            border-radius: 20px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04);
            overflow: hidden;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        .result-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, var(--rc-tint) 0%, rgba(255, 255, 255, 0) 45%);
            pointer-events: none;
        }
        .result-card--success { --rc-tint: rgba(34, 197, 94, 0.18); }
        .result-card--error   { --rc-tint: rgba(239, 68, 68, 0.15); }

        .result-card__icon {
            position: relative;
            flex: 0 0 auto;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #ffffff;
            border-radius: 14px;
            border: 1px solid rgba(15, 23, 42, 0.05);
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        }
        .result-card__icon svg { width: 24px; height: 24px; }
        .result-card--success .result-card__icon svg { stroke: #22c55e; }
        .result-card--error .result-card__icon svg { stroke: #ef4444; }

        .result-card__body { position: relative; min-width: 0; }
        .result-card__title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.3;
        }
        .result-card__desc {
            margin-top: 4px;
            font-size: 0.95rem;
            font-weight: 500;
            color: #64748b;
            line-height: 1.55;
            max-width: 62ch;
        }

        /* ========================================================
           10. ACCORDION (expander) - baris putih, ikon kiri, plus/minus kanan
           ======================================================== */
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
            overflow: hidden;
            transition: background-color 0.2s ease;
        }
        [data-testid="stExpander"]:has(details[open]) {
            background: #f4f6fa !important;
        }
        [data-testid="stExpander"] details {
            border: none !important;
            background: transparent !important;
        }
        [data-testid="stExpander"] summary {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 13px 16px !important;
            background: transparent !important;
            color: #0f172a !important;
        }
        [data-testid="stExpander"] summary:hover {
            background: transparent !important;
            color: #0f172a !important;
        }
        [data-testid="stExpander"] summary svg {
            display: none !important;
        }
        [data-testid="stExpander"] summary p {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            color: #0f172a !important;
        }
        [data-testid="stExpander"] summary::before {
            content: "";
            flex: 0 0 auto;
            width: 18px;
            height: 18px;
            background-repeat: no-repeat;
            background-size: contain;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='4' width='14' height='17' rx='2'/%3E%3Cpath d='M9 4h6v3H9z'/%3E%3Cpath d='M9 12h6M9 16h4'/%3E%3C/svg%3E");
        }
        [data-testid="stExpander"] summary::after {
            content: "+";
            margin-left: auto;
            font-size: 1.4rem;
            font-weight: 400;
            line-height: 1;
            color: #94a3b8;
        }
        [data-testid="stExpander"] details[open] > summary::after {
            content: "\\2212";
        }
        [data-testid="stExpanderDetails"] {
            padding: 0 16px 16px 16px !important;
        }

        /* ========================================================
           11. DATA TABLE (gaya Team Members)
           ======================================================== */
        .tm-wrap {
            width: 100%;
            overflow-x: auto;
            background: #ffffff;
            border: 1px solid #e8ebf0;
            border-radius: 14px;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        .tm-table {
            width: 100%;
            border-collapse: collapse;
        }
        .tm-table th {
            padding: 11px 16px;
            background: #f6f7f9;
            color: #64748b;
            font-size: 0.74rem;
            font-weight: 600;
            text-align: left;
            white-space: nowrap;
            border-bottom: 1px solid #eceef2;
        }
        .tm-table td {
            padding: 15px 16px;
            color: #0f172a;
            font-size: 0.82rem;
            font-weight: 500;
            white-space: nowrap;
            border-bottom: 1px solid #f0f2f5;
        }
        .tm-table tbody tr:last-child td { border-bottom: none; }
        .tm-table tbody tr:hover td { background: #f8fafc; }

        /* ========================================================
           12. PROGRESS CARD (dark glass, bar gradient mengikuti aurora)
           ======================================================== */
        .pg-card {
            padding: 30px 32px 26px 32px;
            margin-top: 0.75rem;
            border-radius: 30px;
            background: linear-gradient(180deg, #3b3b3e 0%, #1e1e20 100%);
            box-shadow: 0 24px 48px rgba(31, 38, 135, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        .pg-row + .pg-row { margin-top: 26px; }
        .pg-head {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 10px;
        }
        .pg-name {
            color: #e2e8f0;
            font-size: 1.05rem;
            font-weight: 600;
        }
        .pg-value {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 500;
            line-height: 1;
            letter-spacing: -1px;
        }
        .pg-value span { color: #8b8d94; }
        .pg-track {
            height: 38px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
        }
        .pg-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
            box-shadow: 0 0 22px rgba(168, 237, 234, 0.55), 0 0 44px rgba(254, 214, 227, 0.28);
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

# --- VISUALISASI PERBANDINGAN MODEL ---
COMPARISON_DIR = os.path.join('assets', 'comparison')
COMPARISON_IMAGES = [
    ("comparison_metrics.png", "Perbandingan 5 Metrik Evaluasi",
     "Grouped bar chart akurasi, presisi, recall, F1-score, dan ROC-AUC tiap model."),
    ("comparison_confusion.png", "Confusion Matrix Berdampingan",
     "Perbandingan sebaran True/False Positive & Negative antar model."),
    ("comparison_roc.png", "Kurva ROC-AUC",
     "Kemampuan diskriminasi tiap model pada seluruh threshold."),
    ("comparison_radar.png", "Radar Performa Multidimensi",
     "Semakin lebar area terisi, semakin seimbang performa model."),
    ("comparison_efficiency.png", "Trade-off Akurasi vs Waktu Training",
     "Keseimbangan antara kualitas prediksi dan biaya komputasi."),
]

# Angka diambil dari output `python -m src.modeling.comparison` (test set 3.822 baris)
COMPARISON_SCORES = {
    "Decision Tree": {"accuracy": 69.99, "precision": 70.07, "recall": 69.99, "f1": 70.03, "auc": 60.72},
    "SVM": {"accuracy": 77.32, "precision": 75.17, "recall": 77.32, "f1": 75.63, "auc": 71.58},
    "Naïve Bayes": {"accuracy": 74.02, "precision": 73.67, "recall": 74.02, "f1": 73.84, "auc": 71.20},
}


@st.cache_resource
def build_string_encoders():
    string_encoders = {}
    for col, vals in OPTIONS.items():
        le = LabelEncoder()
        le.fit(sorted(vals))
        string_encoders[col] = le
    return string_encoders

# --- RESULT CARD: IKON & RENDER ---
RESULT_ICONS = {
    "success": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/></svg>'
    ),
    "error": (
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5"/><path d="M12 16h.01"/></svg>'
    ),
}

def render_result_card(kind, title, description):
    # HTML sengaja dibuat satu baris tanpa indentasi supaya tidak dibaca sebagai code block oleh markdown
    return (
        f'<div class="result-card result-card--{kind}">'
        f'<div class="result-card__icon">{RESULT_ICONS[kind]}</div>'
        f'<div class="result-card__body">'
        f'<div class="result-card__title">{title}</div>'
        f'<div class="result-card__desc">{description}</div>'
        f'</div></div>'
    )

# --- RINGKASAN HASIL (rule-based dari input kandidat) ---
def _experience_to_years(value):
    if value == "<1":
        return 0
    if value == ">20":
        return 21
    try:
        return int(value)
    except ValueError:
        return None

def profile_signals(d):
    """Mengembalikan (faktor_pendukung_bertahan, faktor_risiko_pindah) dari profil input."""
    stable, risk = [], []

    cdi = d["city_development_index"]
    if cdi >= 0.8:
        stable.append("indeks perkembangan kota tinggi")
    elif cdi < 0.6:
        risk.append("indeks perkembangan kota rendah")

    if d["relevent_experience"] == "Has relevent experience":
        stable.append("memiliki pengalaman relevan")
    else:
        risk.append("belum memiliki pengalaman relevan")

    if d["enrolled_university"] == "Full time course":
        risk.append("sedang menempuh kuliah penuh waktu")

    years = _experience_to_years(d["experience"])
    if years is not None:
        if years >= 10:
            stable.append("pengalaman kerja panjang")
        elif years <= 2:
            risk.append("pengalaman kerja masih singkat")

    return stable, risk

def build_result_copy(will_leave, leave_prob, d):
    stable, risk = profile_signals(d)
    prob_text = f"{leave_prob * 100:.1f}%" if leave_prob is not None else None

    if will_leave:
        title = "High Risk of Leaving"
        desc = "Model memprediksi kandidat ini berpotensi pindah kerja"
        desc += f", dengan probabilitas keluar {prob_text}." if prob_text else "."
        if risk:
            desc += f" Faktor profil yang sejalan: {', '.join(risk[:2])}."
    else:
        title = "Likely to Retain"
        desc = "Model memprediksi kandidat ini cenderung bertahan"
        desc += f", dengan probabilitas keluar hanya {prob_text}." if prob_text else "."
        if stable:
            desc += f" Faktor profil yang sejalan: {', '.join(stable[:2])}."

    return title, desc

# --- TABEL PROFIL (gaya Team Members) ---
def render_profile_table(data):
    # HTML satu baris tanpa indentasi supaya tidak dibaca sebagai code block oleh markdown
    head = "".join(f"<th>{html.escape(str(k))}</th>" for k in data.keys())
    cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in data.values())
    return (
        '<div class="tm-wrap"><table class="tm-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody><tr>{cells}</tr></tbody>"
        "</table></div>"
    )

# --- PROGRESS CARD (tab Model Progress) ---
def render_progress_card(items):
    # HTML satu baris tanpa indentasi supaya tidak dibaca sebagai code block oleh markdown
    rows = "".join(
        '<div class="pg-row">'
        f'<div class="pg-head"><div class="pg-name">{html.escape(name)}</div>'
        f'<div class="pg-value">{pct}<span>%</span></div></div>'
        f'<div class="pg-track"><div class="pg-fill" style="width:{pct}%"></div></div>'
        '</div>'
        for name, pct in items
    )
    return f'<div class="pg-card">{rows}</div>'

def main():
    apply_custom_css()

    st.markdown('<div class="main-header">Predict Candidate Movement</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered SVM analysis to predict employee retention.</div>', unsafe_allow_html=True)

    try:
        model, scaler, encoders = load_artifacts()
        string_encoders = build_string_encoders()
    except FileNotFoundError as e:
        st.error("File artefak ML tidak ditemukan. Pastikan assets/model_svm.pkl, dll tersedia.")
        return

    # --- IMPLEMENTASI CAPSULE TABS ---
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Model Comparison", "Model Progress"])

    with tab1:
        # --- IMPLEMENTASI METRIC CARDS ---
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("SVM Accuracy", "82.5%", "Optimal")
        col_m2.metric("Total Trainees", "19,158", "Dataset")
        col_m3.metric("Selected Features", "9/12", "Cleaned")

        with st.form("prediction_form"):
            col1, col_spacer, col2 = st.columns([1, 0.1, 1])
            
            with col1:
                st.markdown('<div class="section-title">Demographics & Company</div>', unsafe_allow_html=True)
                city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.8, help="Skor indeks perkembangan infrastruktur dan ekonomi dari kota domisili kandidat. Nilai mendekati 1.0 menunjukkan kota sangat maju.")
                training_hours = st.number_input("Training Hours", min_value=1, max_value=500, value=50, help="Total jam pelatihan sertifikasi atau data science yang telah diselesaikan oleh kandidat.")
                city = st.selectbox("City", OPTIONS["city"], help="Kode area kota asal/domisili kandidat saat ini.")
                company_size = st.selectbox("Company Size", OPTIONS["company_size"], help="Estimasi jumlah total karyawan di perusahaan tempat kandidat bekerja saat ini.")
                company_type = st.selectbox("Company Type", OPTIONS["company_type"], help="Jenis badan usaha dari perusahaan kandidat (Pvt Ltd = Private Limited Company).")
                gender = st.selectbox("Gender", OPTIONS["gender"], help="Identitas gender kandidat.")
                
            with col2:
                st.markdown('<div class="section-title">Experience & Education</div>', unsafe_allow_html=True)
                enrolled_university = st.selectbox("Enrolled University", OPTIONS["enrolled_university"], help="Status pendidikan kandidat saat ini. Pilih 'no_enrollment' jika tidak sedang mengambil kuliah/kursus penuh waktu.")
                education_level = st.selectbox("Education Level", OPTIONS["education_level"], help="Jenjang pendidikan formal tertinggi yang telah diselesaikan.")
                relevent_experience = st.selectbox("Relevant Experience", OPTIONS["relevent_experience"], help="Apakah kandidat memiliki pengalaman kerja profesional yang relevan di bidang Data Science/Analytics.")
                last_new_job = st.selectbox("Last New Job", OPTIONS["last_new_job"], help="Selisih waktu (dalam tahun) sejak kandidat pindah dari pekerjaan sebelumnya ke pekerjaan yang sekarang.")
                experience = st.selectbox("Years of Experience", OPTIONS["experience"], help="Total pengalaman kerja profesional kandidat dalam hitungan tahun.")
                major_discipline = st.selectbox("Major Discipline", OPTIONS["major_discipline"], help="Bidang studi utama saat kuliah (STEM = Science, Technology, Engineering, Mathematics).")

            st.write("") 
            submitted = st.form_submit_button("Analyze Candidate", use_container_width=True)

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
                        try: encoded_val = encoders[col].transform([val])[0]
                        except: pass
                    if encoded_val is None: encoded_val = string_encoders[col].transform([val])[0]
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
                will_leave = prediction[0] == 1

                leave_prob = None
                try:
                    leave_prob = float(model.predict_proba(final_input)[0][1])
                except AttributeError:
                    pass

                st.divider()

                title, desc = build_result_copy(will_leave, leave_prob, input_dict)
                st.markdown(
                    render_result_card("error" if will_leave else "success", title, desc),
                    unsafe_allow_html=True
                )

                if leave_prob is not None:
                    st.markdown(f"**Leave Probability: {leave_prob*100:.1f}%**")
                    st.progress(leave_prob)
                
                with st.expander("Tinjau Rincian Profil Kandidat (Data Formatted)"):
                    st.markdown(render_profile_table(input_dict), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan pemrosesan: {e}")

    with tab2:
        st.subheader("Perbandingan Model: Decision Tree vs SVM vs Naïve Bayes")
        st.markdown(
            "Evaluasi pada **3.822 data test** (20% hold-out) dengan 9 fitur terpilih. "
            "Semua model dilatih pada data training yang sama agar perbandingan adil."
        )

        cols = st.columns(3)
        for col, (name, s) in zip(cols, COMPARISON_SCORES.items()):
            is_best = s["accuracy"] == max(v["accuracy"] for v in COMPARISON_SCORES.values())
            col.metric(f"{name} · Accuracy", f"{s['accuracy']:.2f}%", "Terbaik" if is_best else None)

        if not os.path.isdir(COMPARISON_DIR):
            st.warning(
                "Grafik perbandingan belum tersedia. Jalankan:\n\n"
                "`python -m src.modeling.comparison`"
            )
        else:
            for filename, title, caption in COMPARISON_IMAGES:
                path = os.path.join(COMPARISON_DIR, filename)
                if not os.path.exists(path):
                    continue
                st.divider()
                st.markdown(f"#### {title}")
                st.caption(caption)
                st.image(path, use_container_width=True)

    with tab3:
        st.subheader("Model Feature Importance (Simulation)")
        st.markdown("Berikut adalah simulasi visualisasi kontribusi setiap fitur.")
        
        feature_importance = [
            ("City Development Index", 84),
            ("Training Hours", 62),
            ("Company Size", 45),
            ("Education Level", 31),
        ]
        st.markdown(render_progress_card(feature_importance), unsafe_allow_html=True)

if __name__ == "__main__":
    main()