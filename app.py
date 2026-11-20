import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# -------------------- ฟอนต์และการตั้งค่าระบบ --------------------
BASE_DIR = Path(__file__).resolve().parent
FONT_REGULAR = BASE_DIR / "fonts" / "Prompt-Regular.ttf"
FONT_BOLD = BASE_DIR / "fonts" / "Prompt-Bold.ttf"
IMG_DIR = BASE_DIR / "img"
DATA_PATH = BASE_DIR / "Heart3.csv"

# ลงทะเบียนฟอนต์ภาษาไทยสำหรับ Matplotlib ให้แสดงผลสระ วรรณยุกต์ และข้อความไทยได้ 100% ไม่เป็นกล่องสี่เหลี่ยม (Tofu bug)
if FONT_REGULAR.exists():
    fm.fontManager.addfont(str(FONT_REGULAR))
    if FONT_BOLD.exists():
        fm.fontManager.addfont(str(FONT_BOLD))
    plt.rcParams["font.family"] = "Prompt"
    plt.rcParams["font.sans-serif"] = ["Prompt", "Leelawadee UI", "Tahoma", "Sarabun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# -------------------- ตั้งค่าหน้าเว็บ Streamlit --------------------
st.set_page_config(
    page_title="HeartCare AI | ระบบวิเคราะห์และจำแนกความเสี่ยงโรคหัวใจ",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS เพื่อความสวยงาม ทันสมัย สไตล์ระบบการแพทย์ดิจิทัลระดับมืออาชีพ
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Prompt', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1420px;
}

/* Hero Header */
.hero-container {
    background: linear-gradient(135deg, #0f2b48 0%, #1e4976 50%, #0d9488 100%);
    padding: 30px 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px -5px rgba(15, 43, 72, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.15);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #e2e8f0;
    margin-top: 8px;
    margin-bottom: 0px;
    font-weight: 300;
}
.hero-status-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #6ee7b7;
    margin-top: 12px;
    border: 1px solid rgba(110, 231, 183, 0.4);
}

/* Metric Cards */
.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
}
.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.metric-box .metric-label {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 500;
}
.metric-box .metric-value {
    color: #0f172a;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-box .metric-sub {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 4px;
}

/* Highlight Cards */
.card-glass {
    background: #ffffff;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
    margin-bottom: 20px;
}

.section-header {
    color: #0f2b48;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 18px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Result Banners */
.result-banner-healthy {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 24px;
    color: #064e3b;
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.15);
}
.result-banner-risk {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 24px;
    color: #7f1d1d;
    box-shadow: 0 6px 18px rgba(239, 68, 68, 0.15);
}

/* Tip / Notice Box */
.notice-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 12px 16px;
    border-radius: 8px;
    color: #14532d;
    font-size: 0.92rem;
    margin: 10px 0;
}
.notice-warning {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 8px;
    color: #78350f;
    font-size: 0.92rem;
    margin: 10px 0;
}
.notice-info {
    background: #f0f9ff;
    border-left: 4px solid #0284c7;
    padding: 12px 16px;
    border-radius: 8px;
    color: #0369a1;
    font-size: 0.92rem;
    margin: 10px 0;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    object-fit: cover;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -------------------- การแมปตัวแปรทางการแพทย์ --------------------
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]
TARGET = "HeartDisease"

# คำอธิบายรหัสหมวดหมู่ภาษาไทยตามมาตรฐาน Kaggle Heart Failure Prediction
SEX_MAP = {
    "หญิง (Female - 0)": 0,
    "ชาย (Male - 1)": 1
}
SEX_INV = {0: "หญิง", 1: "ชาย"}

CHEST_MAP = {
    "ASY: ไม่แสดงอาการเจ็บหน้าอก (Asymptomatic - 1)": 1,
    "ATA: เจ็บหน้าอกผิดปกติแบบไม่ตรงจุด (Atypical Angina - 2)": 2,
    "NAP: แน่นหน้าอกที่ไม่ใช่จากหลอดเลือดหัวใจ (Non-Anginal Pain - 3)": 3,
    "TA: เจ็บแน่นหน้าอกตามแบบฉบับโรคหัวใจ (Typical Angina - 4)": 4
}
CHEST_INV = {1: "ASY (ไม่แสดงอาการ)", 2: "ATA (เจ็บไม่ตรงจุด)", 3: "NAP (แน่นหน้าอกทั่วไป)", 4: "TA (เจ็บแน่นแบบฉบับ)"}

ECG_MAP = {
    "LVH: ภาวะกล้ามเนื้อหัวใจห้องล่างซ้ายหนาตัว (Left Ventricular Hypertrophy - 1)": 1,
    "Normal: คลื่นไฟฟ้าหัวใจปกติ (Normal - 2)": 2,
    "ST: คลื่นไฟฟ้าหัวใจช่วง ST-T ผิดปกติ (ST-T wave abnormality - 3)": 3
}
ECG_INV = {1: "LVH (ห้องล่างซ้ายหนา)", 2: "Normal (ปกติ)", 3: "ST (ST-T ผิดปกติ)"}

YESNO_MAP = {
    "ไม่มี / ไม่ใช่ (0)": 0,
    "มี / ใช่ (1)": 1
}
YESNO_INV = {0: "ไม่ใช่", 1: "ใช่"}

SLOPE_MAP = {
    "Down: ความชันลดลง (Downsloping - 1)": 1,
    "Flat: ความชันราบเรียบ (Flat - 2)": 2,
    "Up: ความชันพุ่งขึ้น (Upsloping - 3)": 3
}
SLOPE_INV = {1: "Down (ลาดลง)", 2: "Flat (ราบเรียบ)", 3: "Up (ลาดขึ้น)"}

# -------------------- ฟังก์ชันโหลดและตรวจสอบข้อมูล --------------------
@st.cache_data
def load_default_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ชุดข้อมูลที่ {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def validate_data(data):
    missing_cols = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing_cols:
        raise ValueError(f"คอลัมน์ไม่ครบถ้วน ขาดคอลัมน์: {', '.join(missing_cols)}")
    clean = data[FEATURES + [TARGET]].copy()
    for c in clean.columns:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    if clean.isna().any().any():
        raise ValueError("พบค่าว่าง (Missing Values) หรือข้อมูลที่ไม่ใช่ตัวเลข กรุณาตรวจสอบไฟล์")
    unique_targets = set(clean[TARGET].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("คอลัมน์เป้าหมาย HeartDisease ต้องมีค่าเป็น 0 หรือ 1 เท่านั้น")
    if len(unique_targets) < 2:
        raise ValueError("ชุดข้อมูลต้องมีทั้งสองกลุ่ม (0 และ 1) เพื่อใช้ในการเรียนรู้ของโมเดล")
    return clean

# -------------------- ฝึกสอนโมเดล KNN พร้อม Caching --------------------
@st.cache_resource
def train_knn_model(data_hash, X_values, y_values, k, metric_name, weight_type, test_size, random_state):
    X = pd.DataFrame(X_values, columns=FEATURES)
    y = pd.Series(y_values, name=TARGET).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights=weight_type,
        metric=metric_name
    )
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    y_proba = knn.predict_proba(X_test_scaled)
    
    # คำนวณ K-Value Accuracy Curve (1 ถึง 25) เพื่อหาค่า K ที่มีประสิทธิภาพสูงสุด
    k_range = list(range(1, min(26, len(X_train))))
    k_accuracies = []
    for test_k in k_range:
        m = KNeighborsClassifier(n_neighbors=test_k, weights=weight_type, metric=metric_name)
        m.fit(X_train_scaled, y_train)
        k_accuracies.append(accuracy_score(y_test, m.predict(X_test_scaled)))
        
    return {
        "scaler": scaler,
        "knn": knn,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "k_range": k_range,
        "k_accuracies": k_accuracies
    }

# -------------------- ฟังก์ชันแสดงผลการ์ด Metric --------------------
def render_metric_card(label, value, subtext="", icon="📈"):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-header">
            <span class="metric-label">{label}</span>
            <span style="font-size: 1.2rem;">{icon}</span>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------- โหลดชุดข้อมูล --------------------
try:
    default_df = validate_data(load_default_data())
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดไฟล์ข้อมูลเริ่มต้น Heart3.csv ได้: {e}")
    st.stop()

# -------------------- ส่วนแถบด้านข้าง (Sidebar) --------------------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:2rem;">🫀</span>
        <div>
            <h2 style="margin:0; font-size:1.35rem; color:#0f2b48; font-weight:700;">HeartCare AI</h2>
            <span style="font-size:0.8rem; color:#64748b;">Medical KNN Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📌 เมนูระบบ")
    menu = st.radio(
        "เลือกหน้าการทำงาน",
        [
            "📊 ภาพรวมและสถิติข้อมูล",
            "🧠 ประเมินและปรับจูนโมเดล KNN",
            "🩺 วินิจฉัยและคัดกรองรายบุคคล",
            "📁 คัดกรองแบบกลุ่ม (Batch CSV)",
            "📚 พจนานุกรมข้อมูลทางการแพทย์"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ⚙️ การตั้งค่าโมเดล KNN")
    k_input = st.slider("จำนวนเพื่อนบ้านใกล้ที่สุด (K)", min_value=1, max_value=25, value=5, step=1,
                        help="จำนวนเพื่อนบ้านใกล้ที่สุดที่นำมาลงคะแนนเสียง (แนะนำเลขคี่เพื่อป้องกันคะแนนเสมอกัน)")
    
    test_size_pct = st.slider("สัดส่วนข้อมูลทดสอบ (Test Size %)", min_value=10, max_value=40, value=20, step=5,
                              help="เปอร์เซ็นต์ของชุดข้อมูลที่นำมาใช้ทดสอบโมเดล (ส่วนที่เหลือใช้ฝึกโมเดล)")
    
    metric_choice = st.selectbox(
        "ฟังก์ชันวัดระยะทาง (Distance Metric)",
        ["euclidean", "manhattan", "minkowski"],
        format_func=lambda x: "Euclidean (L2 - ระยะทางตรงมาตรฐาน)" if x == "euclidean" else ("Manhattan (L1 - ระยะตั้งฉากตามบล็อก)" if x == "manhattan" else "Minkowski (L-p metric)")
    )
    
    weights_choice = st.selectbox(
        "การถ่วงน้ำหนักเพื่อนบ้าน (Weighting)",
        ["distance", "uniform"],
        format_func=lambda x: "Distance (เพื่อนบ้านที่ใกล้กว่ามีอิทธิพลมากกว่า)" if x == "distance" else "Uniform (ทุกเพื่อนบ้านมีน้ำหนักเท่ากัน)"
    )
    
    seed_val = st.number_input("Random Seed (สุ่มคงที่)", min_value=0, max_value=9999, value=42, step=1)
    
    st.divider()
    
    st.markdown("### 📂 แหล่งชุดข้อมูล")
    user_file = st.file_uploader("อัปโหลดชุดข้อมูล Heart CSV อื่น (ถ้าต้องการ)", type=["csv"])
    if user_file is not None:
        try:
            active_df = validate_data(pd.read_csv(user_file))
            st.success("✅ ใช้ชุดข้อมูลที่อัปโหลดสำเร็จ")
        except Exception as err:
            st.error(f"ไฟล์ไม่ถูกต้อง: {err}")
            active_df = default_df
    else:
        active_df = default_df
        st.caption("กำลังใช้ชุดข้อมูลทางการแพทย์: `Heart3.csv` (918 รายการ)")

# ฝึกโมเดล
try:
    df_hash = str(pd.util.hash_pandas_object(active_df, index=True).sum())
    results = train_knn_model(
        df_hash,
        active_df[FEATURES].values.tolist(),
        active_df[TARGET].values.tolist(),
        k=int(k_input),
        metric_name=metric_choice,
        weight_type=weights_choice,
        test_size=test_size_pct / 100.0,
        random_state=int(seed_val)
    )
except Exception as e:
    st.error(f"❌ ไม่สามารถประมวลผลโมเดลได้: {e}")
    st.stop()

# -------------------- Hero Header สำหรับทุกหน้า --------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🫀 HeartCare AI</span>
        <span style="font-size:1.1rem; font-weight:400; opacity:0.85;">| ระบบคัดกรองและประเมินความเสี่ยงโรคหัวใจ</span>
    </div>
    <p class="hero-subtitle">
        ผู้ช่วยวิเคราะห์ทางการแพทย์อัจฉริยะด้วยขั้นตอนวิธี K-Nearest Neighbors (KNN) พร้อมการปรับสเกลข้อมูลแบบ StandardScaler
    </p>
    <div class="hero-status-pill">
        ● ฐานข้อมูลพร้อมใช้งาน (918 รายการ) | โมเดล KNN พร้อมทำงานเต็มประสิทธิภาพ
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# หน้าที่ 1: 📊 ภาพรวมและสถิติข้อมูล (Clinical Dashboard & EDA)
# ==============================================================================
if menu == "📊 ภาพรวมและสถิติข้อมูล":
    st.markdown('<div class="section-header">📈 ตัวชี้วัดสำคัญของชุดข้อมูลสุขภาพหัวใจ</div>', unsafe_allow_html=True)
    
    total_records = len(active_df)
    disease_count = int(active_df[TARGET].sum())
    healthy_count = total_records - disease_count
    disease_pct = (disease_count / total_records) * 100
    avg_age = active_df["Age"].mean()
    avg_bp = active_df["RestingBP"].mean()
    avg_chol = active_df["Cholesterol"].mean()
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        render_metric_card("จำนวนกลุ่มตัวอย่างทั้งหมด", f"{total_records:,} คน", "ข้อมูลในระบบ", "👥")
    with kpi2:
        render_metric_card("พบความเสี่ยงโรคหัวใจ", f"{disease_pct:.1f}%", f"{disease_count:,} คน (Class 1)", "⚠️")
    with kpi3:
        render_metric_card("กลุ่มสุขภาพปกติ", f"{100-disease_pct:.1f}%", f"{healthy_count:,} คน (Class 0)", "💚")
    with kpi4:
        render_metric_card("อายุเฉลี่ยผู้ป่วย", f"{avg_age:.1f} ปี", "ช่วง 28 - 77 ปี", "🎂")
    with kpi5:
        render_metric_card("ความดันเฉลี่ยขณะพัก", f"{avg_bp:.0f} mmHg", "ค่ามาตรฐาน < 120", "🩺")

    st.write("")
    
    # นำเข้ารูปภาพตัวอย่างแสดงความเปรียบเทียบทางการแพทย์
    st.markdown('<div class="section-header">🫀 โครงสร้างและสัณฐานวิทยาของหัวใจ</div>', unsafe_allow_html=True)
    img_col1, img_col2, img_col3 = st.columns(3)
    
    with img_col1:
        st.markdown("""
        <div style="background:white; padding:15px; border-radius:14px; border:1px solid #e2e8f0; text-align:center;">
            <span style="font-weight:600; color:#10b981; font-size:1.05rem;">หัวใจปกติ (Normal Heart)</span>
        </div>
        """, unsafe_allow_html=True)
        img1_p = IMG_DIR / "heart1.jpg"
        if img1_p.exists():
            st.image(str(img1_p), caption="โครงสร้างกล้ามเนื้อและหลอดเลือดหัวใจสมบูรณ์", use_container_width=True)
        else:
            st.info("รูปภาพ heart1.jpg")
            
    with img_col2:
        st.markdown("""
        <div style="background:white; padding:15px; border-radius:14px; border:1px solid #e2e8f0; text-align:center;">
            <span style="font-weight:600; color:#ef4444; font-size:1.05rem;">หัวใจมีพยาธิสภาพ (Heart Disease)</span>
        </div>
        """, unsafe_allow_html=True)
        img2_p = IMG_DIR / "heart2.jpg"
        if img2_p.exists():
            st.image(str(img2_p), caption="ภาวะหลอดเลือดโคโรนารีตีบและขาดเลือด", use_container_width=True)
        else:
            st.info("รูปภาพ heart2.jpg")
            
    with img_col3:
        st.markdown("""
        <div style="background:white; padding:15px; border-radius:14px; border:1px solid #e2e8f0; text-align:center;">
            <span style="font-weight:600; color:#0284c7; font-size:1.05rem;">หัวใจฟิตสมบูรณ์ (Healthy State)</span>
        </div>
        """, unsafe_allow_html=True)
        img3_p = IMG_DIR / "heart3.jpg"
        if img3_p.exists():
            st.image(str(img3_p), caption="สุขภาพหัวใจแข็งแรง ดัชนีชีพจรปกติ", use_container_width=True)
        else:
            st.info("รูปภาพ heart3.jpg")

    st.write("")
    
    # กราฟแสดงสัดส่วนและการแจกแจงแบบ Interactive ผ่าน Plotly (แก้ปัญหาตัวหนังสือไทยเป็นกล่องสี่เหลี่ยม 100%)
    st.markdown('<div class="section-header">📊 การวิเคราะห์การแจกแจงข้อมูลเชิงลึก (Interactive Data Visualization)</div>', unsafe_allow_html=True)
    c_chart1, c_chart2 = st.columns([1, 1.2])
    
    with c_chart1:
        # Donut Chart ด้วย Plotly สวย คมชัด อักษรไทยถูกต้อง 100%
        pie_df = pd.DataFrame({
            "สถานะ": ["ปกติ ไม่พบโรค (Class 0)", "มีความเสี่ยงโรคหัวใจ (Class 1)"],
            "จำนวน": [healthy_count, disease_count]
        })
        fig_donut = px.pie(
            pie_df,
            names="สถานะ",
            values="จำนวน",
            hole=0.55,
            color="สถานะ",
            color_discrete_map={
                "ปกติ ไม่พบโรค (Class 0)": "#10b981",
                "มีความเสี่ยงโรคหัวใจ (Class 1)": "#ef4444"
            },
            title="<b>สัดส่วนผู้ป่วย: ไม่พบโรค vs พบโรคหัวใจ</b>"
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#ffffff", width=2.5))
        )
        fig_donut.update_layout(
            font=dict(family="Prompt, sans-serif", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=50, b=30, l=10, r=10),
            height=380
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c_chart2:
        # Scatter Plot แสดงความสัมพันธ์ระหว่าง อายุ และ อัตราการเต้นหัวใจสูงสุด
        scatter_df = active_df.copy()
        scatter_df["ผลวินิจฉัย"] = scatter_df[TARGET].map({0: "ปกติ ไม่พบโรค (0)", 1: "พบความเสี่ยงโรคหัวใจ (1)"})
        
        fig_scatter = px.scatter(
            scatter_df,
            x="Age",
            y="MaxHR",
            color="ผลวินิจฉัย",
            color_discrete_map={
                "ปกติ ไม่พบโรค (0)": "#10b981",
                "พบความเสี่ยงโรคหัวใจ (1)": "#ef4444"
            },
            labels={"Age": "อายุ (ปี)", "MaxHR": "อัตราการเต้นหัวใจสูงสุด (bpm)"},
            title="<b>ความสัมพันธ์ระหว่าง อายุ และ อัตราการเต้นหัวใจสูงสุด (MaxHR)</b>",
            opacity=0.75,
            hover_data=["RestingBP", "Cholesterol"]
        )
        fig_scatter.update_layout(
            font=dict(family="Prompt, sans-serif", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(t=50, b=40, l=10, r=10),
            height=380
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # กราฟประเภทอาการเจ็บหน้าอก และ ความสัมพันธ์ของฟีเจอร์
    c_chart3, c_chart4 = st.columns([1, 1])
    with c_chart3:
        cp_df = active_df.copy()
        cp_df["ประเภทเจ็บหน้าอก"] = cp_df["ChestPainType"].map(CHEST_INV)
        cp_df["ผลการตรวจ"] = cp_df[TARGET].map({0: "ปกติ (0)", 1: "พบโรคหัวใจ (1)"})
        cp_summary = cp_df.groupby(["ประเภทเจ็บหน้าอก", "ผลการตรวจ"]).size().reset_index(name="จำนวน")
        
        fig_bar = px.bar(
            cp_summary,
            x="ประเภทเจ็บหน้าอก",
            y="จำนวน",
            color="ผลการตรวจ",
            barmode="group",
            color_discrete_map={"ปกติ (0)": "#10b981", "พบโรคหัวใจ (1)": "#ef4444"},
            title="<b>การกระจายตัวของโรคหัวใจจำแนกตามประเภทอาการเจ็บหน้าอก</b>",
            labels={"จำนวน": "จำนวนผู้ป่วย (คน)"}
        )
        fig_bar.update_layout(
            font=dict(family="Prompt, sans-serif", size=12),
            margin=dict(t=50, b=20, l=10, r=10),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_chart4:
        # สหสัมพันธ์ (Correlation) กับตัวแปรเป้าหมาย
        corr = active_df.corr()[TARGET].drop(TARGET).sort_values()
        corr_df = pd.DataFrame({"ตัวแปร": corr.index, "ค่าสัมประสิทธิ์สหสัมพันธ์": corr.values})
        corr_df["ทิศทาง"] = np.where(corr_df["ค่าสัมประสิทธิ์สหสัมพันธ์"] >= 0, "บวก (เพิ่มความเสี่ยง)", "ลบ (ลดความเสี่ยง)")
        
        fig_corr = px.bar(
            corr_df,
            x="ค่าสัมประสิทธิ์สหสัมพันธ์",
            y="ตัวแปร",
            orientation="h",
            color="ทิศทาง",
            color_discrete_map={"บวก (เพิ่มความเสี่ยง)": "#f97316", "ลบ (ลดความเสี่ยง)": "#0284c7"},
            title="<b>ความสัมพันธ์ของแต่ละตัวแปรกับความเสี่ยงโรคหัวใจ (Correlation)</b>"
        )
        fig_corr.update_layout(
            font=dict(family="Prompt, sans-serif", size=12),
            margin=dict(t=50, b=20, l=10, r=10),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ตารางสถิติและตัวอย่างข้อมูล
    st.markdown('<div class="section-header">📋 สถิติเชิงพรรณนาและข้อมูลตัวอย่าง</div>', unsafe_allow_html=True)
    tab_stat1, tab_stat2 = st.tabs(["📊 สถิติเชิงตัวเลข (Descriptive Statistics)", "📑 ดูตัวอย่างชุดข้อมูล (Data Table)"])
    with tab_stat1:
        st.dataframe(active_df[FEATURES].describe().T.round(2), use_container_width=True)
    with tab_stat2:
        st.dataframe(active_df.head(20), use_container_width=True)
        st.download_button(
            "⬇️ ดาวน์โหลดชุดข้อมูลนี้เป็น CSV",
            active_df.to_csv(index=False).encode("utf-8-sig"),
            "heart_data_export.csv",
            "text/csv"
        )


# ==============================================================================
# หน้าที่ 2: 🧠 ประเมินและปรับจูนโมเดล KNN (Model Lab & Evaluation)
# ==============================================================================
elif menu == "🧠 ประเมินและปรับจูนโมเดล KNN":
    st.markdown('<div class="section-header">🧠 การประเมินประสิทธิภาพโมเดล K-Nearest Neighbors</div>', unsafe_allow_html=True)
    
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    y_proba = results["y_proba"]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    st.markdown(f"""
    <div class="notice-info">
        📌 <b>พารามิเตอร์ปัจจุบัน:</b> K = <b>{k_input}</b> | ระยะทาง = <b>{metric_choice.title()}</b> | 
        การถ่วงน้ำหนัก = <b>{weights_choice.title()}</b> | สัดส่วนชุดฝึก {100-test_size_pct}% ({results['n_train']} ตัวอย่าง) | 
        ชุดทดสอบ {test_size_pct}% ({results['n_test']} ตัวอย่าง)
    </div>
    """, unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        render_metric_card("Accuracy", f"{acc*100:.2f}%", "ความถูกต้องรวม", "🎯")
    with m_col2:
        render_metric_card("Sensitivity (Recall)", f"{rec*100:.2f}%", "ความไวในการตรวจพบโรค", "🩺")
    with m_col3:
        render_metric_card("Precision", f"{prec*100:.2f}%", "ความแม่นยำการทำนาย", "🔍")
    with m_col4:
        render_metric_card("F1-Score", f"{f1*100:.2f}%", "ค่าเฉลี่ยฮาร์โมนิก", "⚖️")
    with m_col5:
        render_metric_card("ROC-AUC", f"{roc_auc:.3f}", "สมรรถนะการแยกคลาส", "🌟")
        
    st.write("")
    
    # ส่วนแสดง Confusion Matrix และ ROC Curve แบบ Interactive ไม่มีตัวหนังสือบัค
    eval_c1, eval_c2 = st.columns(2)
    
    with eval_c1:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        cm_text = [
            [f"จริง: ปกติ (TN)<br><b>{tn}</b> คน", f"จริง: ปกติ แต่ทำนายผิด (FP)<br><b>{fp}</b> คน"],
            [f"จริง: มีโรค แต่ทำนายหลุด (FN)<br><b>{fn}</b> คน", f"จริง: มีโรค ทำนายพบ (TP)<br><b>{tp}</b> คน"]
        ]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["ทำนาย: ไม่พบโรค (0)", "ทำนาย: พบโรคหัวใจ (1)"],
            y=["จริง: ไม่พบโรค (0)", "จริง: พบโรคหัวใจ (1)"],
            text=cm_text,
            texttemplate="%{text}",
            textfont={"size": 13, "family": "Prompt"},
            colorscale="Blues",
            showscale=False
        ))
        fig_cm.update_layout(
            title="<b>Confusion Matrix (ตารางเมทริกซ์ความสับสน)</b>",
            xaxis_title="<b>ผลการทำนายโดยโมเดล (Predicted)</b>",
            yaxis_title="<b>ผลตรวจจริงจากแพทย์ (Actual)</b>",
            font=dict(family="Prompt, sans-serif", size=13),
            margin=dict(t=50, b=40, l=40, r=20),
            height=380
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("💡 ในทางการแพทย์ เป้าหมายสำคัญคือการลด False Negative (FN) หรือการตรวจหลุดให้เหลือน้อยที่สุด")
        
    with eval_c2:
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'KNN (AUC = {roc_auc:.3f})',
            line=dict(color='#0284c7', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess (AUC = 0.500)',
            line=dict(color='#94a3b8', dash='dash')
        ))
        fig_roc.update_layout(
            title="<b>Receiver Operating Characteristic (ROC Curve)</b>",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            font=dict(family="Prompt, sans-serif", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=50, b=40, l=40, r=20),
            height=380
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # เส้นกราฟจำลองการหาค่า K ที่ดีที่สุด (Hyperparameter Optimization Curve)
    st.markdown('<div class="section-header">🎯 การวิเคราะห์เพื่อหาค่า K ที่มีประสิทธิภาพสูงสุด (K-Optimization Curve)</div>', unsafe_allow_html=True)
    k_df = pd.DataFrame({
        "K Value": results["k_range"],
        "Test Accuracy": [acc_val * 100 for acc_val in results["k_accuracies"]]
    })
    
    fig_k = px.line(
        k_df,
        x="K Value",
        y="Test Accuracy",
        markers=True,
        title="<b>ความแม่นยำ (Accuracy %) บนชุดทดสอบ ณ ค่า K ตั้งแต่ 1 ถึง 25</b>",
        labels={"Test Accuracy": "ความถูกต้อง (%)", "K Value": "จำนวนเพื่อนบ้าน (K)"}
    )
    # เพิ่มจุดเน้นค่า K ที่เลือกอยู่
    current_acc = acc * 100
    fig_k.add_trace(go.Scatter(
        x=[k_input],
        y=[current_acc],
        mode='markers',
        marker=dict(size=14, color='#ef4444', symbol='star'),
        name=f'ค่า K ปัจจุบัน ({k_input})'
    ))
    fig_k.update_layout(
        font=dict(family="Prompt, sans-serif", size=13),
        margin=dict(t=50, b=30, l=30, r=20),
        height=320
    )
    st.plotly_chart(fig_k, use_container_width=True)

    # รายงานผลจำแนกและตารางเปรียบเทียบผลทำนาย
    st.markdown('<div class="section-header">📑 รายงานการจำแนกประเภท (Classification Report)</div>', unsafe_allow_html=True)
    report_dict = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["ไม่พบโรค (0)", "พบความเสี่ยงโรคหัวใจ (1)"],
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown('<div class="section-header">🔍 รายการผลการทำนายในชุดทดสอบ (Test Predictions Breakdown)</div>', unsafe_allow_html=True)
    res_inspect = results["X_test"].copy()
    res_inspect["ค่าจริง (Actual)"] = y_test.values
    res_inspect["ค่าทำนาย (Predicted)"] = y_pred
    res_inspect["ความน่าจะเป็นพบโรค"] = [f"{p*100:.1f}%" for p in y_proba[:, 1]]
    res_inspect["ผลการจำแนก"] = np.where(y_test.values == y_pred, "✅ ถูกต้อง", "❌ คลาดเคลื่อน")
    
    st.dataframe(res_inspect.head(30), use_container_width=True)
    st.download_button(
        "⬇️ ดาวน์โหลดผลการทดสอบทั้งหมดเป็น CSV",
        res_inspect.to_csv(index=False).encode("utf-8-sig"),
        "knn_test_evaluation_results.csv",
        "text/csv"
    )


# ==============================================================================
# หน้าที่ 3: 🩺 วินิจฉัยและคัดกรองรายบุคคล (Individual Patient Diagnosis)
# ==============================================================================
elif menu == "🩺 วินิจฉัยและคัดกรองรายบุคคล":
    st.markdown('<div class="section-header">🩺 ระบบคัดกรองความเสี่ยงโรคหัวใจรายบุคคล (Clinical Patient Screening)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="notice-info">
        กรอกข้อมูลผลการตรวจสุขภาพของผู้ป่วยในแบบฟอร์มด้านล่าง หรือกดเลือก <b>"ตัวอย่างโปรไฟล์ผู้ป่วยสำเร็จรูป"</b> เพื่อทดสอบระบบได้ทันที
    </div>
    """, unsafe_allow_html=True)

    # สร้างปุ่ม Quick Load Preset Profile เพื่อความสะดวกและโชว์ได้ทันที
    st.markdown("##### ⚡ โหลดข้อมูลตัวอย่างผู้ป่วยจำลองเพื่อทดสอบทันที:")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    # State จัดการข้อมูลในแบบฟอร์ม
    if "preset_data" not in st.session_state:
        st.session_state.preset_data = {
            "age": 52, "sex": 1, "chest": 1, "bp": 130, "chol": 220,
            "fbs": 0, "ecg": 2, "maxhr": 145, "angina": 0, "oldpeak": 0.5, "slope": 2
        }

    with p_col1:
        if st.button("🟢 ตัวอย่าง: ผู้ป่วยสุขภาพดี (ความเสี่ยงต่ำ)", use_container_width=True):
            st.session_state.preset_data = {
                "age": 35, "sex": 0, "chest": 2, "bp": 115, "chol": 175,
                "fbs": 0, "ecg": 2, "maxhr": 168, "angina": 0, "oldpeak": 0.0, "slope": 3
            }
            st.rerun()

    with p_col2:
        if st.button("🟡 ตัวอย่าง: ผู้ป่วยความเสี่ยงปานกลาง", use_container_width=True):
            st.session_state.preset_data = {
                "age": 50, "sex": 1, "chest": 3, "bp": 135, "chol": 235,
                "fbs": 0, "ecg": 2, "maxhr": 142, "angina": 0, "oldpeak": 1.2, "slope": 2
            }
            st.rerun()

    with p_col3:
        if st.button("🔴 ตัวอย่าง: ผู้ป่วยความเสี่ยงสูง (มีภาวะโรคหัวใจ)", use_container_width=True):
            st.session_state.preset_data = {
                "age": 62, "sex": 1, "chest": 1, "bp": 155, "chol": 290,
                "fbs": 1, "ecg": 3, "maxhr": 115, "angina": 1, "oldpeak": 2.5, "slope": 1
            }
            st.rerun()

    cur = st.session_state.preset_data

    # แบบฟอร์มกรอกข้อมูลผู้ป่วย
    with st.form("patient_screening_form"):
        st.markdown("#### 1. ข้อมูลทั่วไปและสัญญาณชีพ")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            in_age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=cur["age"])
            in_sex = st.selectbox(
                "เพศกำเนิด",
                list(SEX_MAP.keys()),
                index=1 if cur["sex"] == 1 else 0
            )
        with r1_c2:
            in_bp = st.number_input("ความดันโลหิตขณะพัก (Resting BP - mmHg)", min_value=70, max_value=250, value=cur["bp"],
                                    help="ค่าปกติควรน้อยกว่า 120 mmHg (หากเกิน 140 เข้าเกณฑ์ความดันโลหิตสูง)")
            in_maxhr = st.number_input("อัตราการเต้นหัวใจสูงสุด (Max HR - bpm)", min_value=50, max_value=240, value=cur["maxhr"],
                                       help="อัตราการเต้นหัวใจสูงสุดขณะออกแรง")
        with r1_c3:
            # หา index ของ chest pain
            cp_idx = [v for v in CHEST_MAP.values()].index(cur["chest"])
            in_chest = st.selectbox(
                "ประเภทอาการเจ็บหน้าอก (Chest Pain Type)",
                list(CHEST_MAP.keys()),
                index=cp_idx
            )
            
        st.markdown("#### 2. ผลการตรวจทางห้องปฏิบัติการและคลื่นไฟฟ้าหัวใจ (ECG)")
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1:
            in_chol = st.number_input("ระดับคอเลสเตอรอลในเลือด (Cholesterol - mg/dL)", min_value=0, max_value=700, value=cur["chol"],
                                      help="ค่าปกติควรน้อยกว่า 200 mg/dL")
            in_fbs = st.selectbox(
                "น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dL",
                list(YESNO_MAP.keys()),
                index=1 if cur["fbs"] == 1 else 0
            )
        with r2_c2:
            ecg_idx = [v for v in ECG_MAP.values()].index(cur["ecg"])
            in_ecg = st.selectbox(
                "ผลตรวจคลื่นไฟฟ้าหัวใจขณะพัก (Resting ECG)",
                list(ECG_MAP.keys()),
                index=ecg_idx
            )
            in_angina = st.selectbox(
                "มีอาการเจ็บแน่นหน้าอกขณะออกกำลังกาย (Exercise Angina)",
                list(YESNO_MAP.keys()),
                index=1 if cur["angina"] == 1 else 0
            )
        with r2_c3:
            in_oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=10.0, value=float(cur["oldpeak"]), step=0.1, format="%.1f",
                                         help="ระดับการกดลงของคลื่น ST บ่งบอกถึงภาวะกล้ามเนื้อหัวใจขาดเลือด")
            slope_idx = [v for v in SLOPE_MAP.values()].index(cur["slope"])
            in_slope = st.selectbox(
                "ความชันของคลื่น ST (ST_Slope)",
                list(SLOPE_MAP.keys()),
                index=slope_idx
            )
            
        btn_submit = st.form_submit_button("🔍 ประเมินและวิเคราะห์ความเสี่ยง (AI Diagnostic)", use_container_width=True, type="primary")

    if btn_submit:
        # เตรียมเวกเตอร์ฟีเจอร์
        patient_row = pd.DataFrame([{
            "Age": in_age,
            "Sex": SEX_MAP[in_sex],
            "ChestPainType": CHEST_MAP[in_chest],
            "RestingBP": in_bp,
            "Cholesterol": in_chol,
            "FastingBS": YESNO_MAP[in_fbs],
            "RestingECG": ECG_MAP[in_ecg],
            "MaxHR": in_maxhr,
            "ExerciseAngina": YESNO_MAP[in_angina],
            "Oldpeak": in_oldpeak,
            "ST_Slope": SLOPE_MAP[in_slope]
        }], columns=FEATURES)

        scaler = results["scaler"]
        knn = results["knn"]
        
        # ปรับสเกลข้อมูลผู้ป่วยใหม่ด้วย Scaler เดียวกับที่ฝึกโมเดล
        patient_scaled = scaler.transform(patient_row)
        pred_label = int(knn.predict(patient_scaled)[0])
        pred_prob = knn.predict_proba(patient_scaled)[0]
        
        # คำนวณความน่าจะเป็นของแต่ละคลาส
        prob_healthy = pred_prob[0] * 100
        prob_disease = pred_prob[1] * 100
        
        st.write("")
        st.markdown('<div class="section-header">📋 รายงานผลการประเมินจากระบบปัญญาประดิษฐ์</div>', unsafe_allow_html=True)
        
        # แบนเนอร์แสดงผลการทำนายแบบมืออาชีพ
        if pred_label == 1:
            st.markdown(f"""
            <div class="result-banner-risk">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h2 style="margin:0; color:#991b1b; font-size:1.75rem; font-weight:700;">
                            ⚠️ ตรวจพบความเสี่ยงสูงต่อโรคหัวใจ (High Risk - Class 1)
                        </h2>
                        <p style="margin:6px 0 0 0; font-size:1.05rem; color:#b91c1c;">
                            แบบจำลอง KNN ประเมินว่าสัญญาณชีพและผลการตรวจของผู้ป่วยมีลักษณะสอดคล้องกับกลุ่มที่มีภาวะโรคหัวใจ
                        </p>
                    </div>
                    <div style="background:#fee2e2; border:2px solid #ef4444; border-radius:12px; padding:10px 20px; text-align:center;">
                        <span style="font-size:0.85rem; color:#991b1b; font-weight:600;">ความน่าจะเป็น</span>
                        <div style="font-size:1.8rem; font-weight:700; color:#ef4444;">{prob_disease:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner-healthy">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h2 style="margin:0; color:#065f46; font-size:1.75rem; font-weight:700;">
                            ✅ ผลการตรวจปกติ ไม่พบความเสี่ยงโรคหัวใจ (Low Risk - Class 0)
                        </h2>
                        <p style="margin:6px 0 0 0; font-size:1.05rem; color:#047857;">
                            แบบจำลอง KNN ประเมินว่าดัชนีสุขภาพของผู้ป่วยอยู่ในเกณฑ์ปลอดภัย สอดคล้องกับกลุ่มบุคคลที่มีสุขภาพหัวใจปกติ
                        </p>
                    </div>
                    <div style="background:#d1fae5; border:2px solid #10b981; border-radius:12px; padding:10px 20px; text-align:center;">
                        <span style="font-size:0.85rem; color:#065f46; font-weight:600;">ระดับความปลอดภัย</span>
                        <div style="font-size:1.8rem; font-weight:700; color:#10b981;">{prob_healthy:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        
        # แสดงรูปภาพอวตารผู้ป่วยและหัวใจประกอบผลการทำนาย
        res_col_img, res_col_gauge = st.columns([1, 1.2])
        with res_col_img:
            sub_c1, sub_c2 = st.columns(2)
            if pred_label == 1:
                with sub_c1:
                    u_dis = IMG_DIR / "user_disease.jpg"
                    if u_dis.exists():
                        st.image(str(u_dis), caption="ผู้ป่วยกลุ่มมีความเสี่ยง", use_container_width=True)
                with sub_c2:
                    h_dis = IMG_DIR / "heart2.jpg"
                    if h_dis.exists():
                        st.image(str(h_dis), caption="ลักษณะหัวใจมีพยาธิสภาพ", use_container_width=True)
            else:
                with sub_c1:
                    u_norm = IMG_DIR / "user_normal.jpg"
                    if u_norm.exists():
                        st.image(str(u_norm), caption="ผู้ป่วยกลุ่มปกติ แข็งแรง", use_container_width=True)
                with sub_c2:
                    h_norm = IMG_DIR / "heart1.jpg"
                    if h_norm.exists():
                        st.image(str(h_norm), caption="ลักษณะหัวใจสมบูรณ์ปกติ", use_container_width=True)

        with res_col_gauge:
            # Guage Chart ความเสี่ยงโรคหัวใจด้วย Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_disease,
                number={'suffix': "%", 'font': {'family': 'Prompt', 'size': 36}},
                title={'text': "<b>เกจวัดความเสี่ยงโรคหัวใจ (Risk Probability)</b>", 'font': {'family': 'Prompt', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#ef4444" if pred_label == 1 else "#10b981"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 35], 'color': '#d1fae5'},
                        {'range': [35, 65], 'color': '#fef3c7'},
                        {'range': [65, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(t=40, b=10, l=30, r=30),
                font=dict(family="Prompt, sans-serif")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # -------------------- XAI: วิเคราะห์ K เพื่อนบ้านใกล้เคียงที่สุด (Explainable AI) --------------------
        st.markdown('<div class="section-header">🔍 คำอธิบายเชิงเหตุผล: วิเคราะห์เพื่อนบ้านที่ใกล้เคียงที่สุด (K-Nearest Neighbors Explanation)</div>', unsafe_allow_html=True)
        
        # คำนวณหาระยะทางและดัชนีของ K เพื่อนบ้านที่ใกล้ที่สุด
        distances, indices = knn.kneighbors(patient_scaled, n_neighbors=int(k_input))
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]
        
        # ดึงข้อมูลจริงของเพื่อนบ้านจากชุดฝึก
        X_train_orig = results["X_train"]
        y_train_orig = results["y_train"]
        
        neighbors_df = X_train_orig.iloc[neighbor_indices].copy()
        neighbors_df["ระยะห่าง (Distance)"] = neighbor_distances.round(3)
        neighbors_df["สถานะจริงของผู้ป่วย (Actual)"] = [
            "⚠️ พบโรคหัวใจ (1)" if y_train_orig.iloc[idx] == 1 else "💚 ปกติ ไม่พบโรค (0)" 
            for idx in neighbor_indices
        ]
        
        nb_disease_count = sum(y_train_orig.iloc[idx] == 1 for idx in neighbor_indices)
        nb_healthy_count = len(neighbor_indices) - nb_disease_count
        
        st.markdown(f"""
        <div class="notice-info">
            💡 จากฐานข้อมูลทั้งหมด ระบบค้นพบผู้ป่วย <b>{k_input} ราย</b> ที่มีค่าดัชนีทางคลินิกใกล้เคียงกับคุณมากที่สุด:<br>
            • ผู้ที่มีภาวะโรคหัวใจ: <b>{nb_disease_count} ราย</b><br>
            • ผู้ที่มีสุขภาพปกติ: <b>{nb_healthy_count} ราย</b><br>
            <i>(โมเดล KNN อาศัยการโหวตจากกลุ่มเพื่อนบ้านที่ใกล้ที่สุดเหล่านี้ในการสรุปผล)</i>
        </div>
        """, unsafe_allow_html=True)
        
        # แปลงรหัสเพื่อแสดงตารางเพื่อนบ้านให้อ่านเข้าใจง่าย
        display_nb = neighbors_df.copy()
        display_nb["Sex"] = display_nb["Sex"].map(SEX_INV)
        display_nb["ChestPainType"] = display_nb["ChestPainType"].map(CHEST_INV)
        display_nb["RestingECG"] = display_nb["RestingECG"].map(ECG_INV)
        display_nb["ST_Slope"] = display_nb["ST_Slope"].map(SLOPE_INV)
        display_nb["ExerciseAngina"] = display_nb["ExerciseAngina"].map(YESNO_INV)
        
        cols_order = ["สถานะจริงของผู้ป่วย (Actual)", "ระยะห่าง (Distance)", "Age", "Sex", "RestingBP", "Cholesterol", "ChestPainType", "MaxHR", "Oldpeak", "ST_Slope"]
        st.dataframe(display_nb[cols_order], use_container_width=True)

        # -------------------- คำแนะนำทางการแพทย์เฉพาะบุคคล --------------------
        st.markdown('<div class="section-header">💡 ข้อแนะนำด้านสุขภาพและการปฏิบัติตัวเฉพาะบุคคล</div>', unsafe_allow_html=True)
        
        alerts = []
        if in_bp >= 140:
            alerts.append("🔴 <b>ความดันโลหิตสูง (Hypertension):</b> ค่า RestingBP อยู่ที่ " + str(in_bp) + " mmHg สูงกว่าเกณฑ์มาตรฐาน (120 mmHg) ควรลดอาหารรสเค็ม เกลือโซเดียม และปรึกษาแพทย์เพื่อควบคุมความดัน")
        elif in_bp >= 130:
            alerts.append("🟡 <b>ความดันโลหิตเริ่มสูง (Pre-hypertension):</b> ความดัน " + str(in_bp) + " mmHg ควรปรับพฤติกรรม ออกกำลังกายสม่ำเสมอ")
            
        if in_chol >= 240:
            alerts.append("🔴 <b>ไขมันในเลือดสูงมาก (Hypercholesterolemia):</b> ระดับคอเลสเตอรอล " + str(in_chol) + " mg/dL เพิ่มความเสี่ยงหลอดเลือดหัวใจตีบอย่างมีนัยสำคัญ ควรตรวจระดับไขมัน LDL/HDL เพิ่มเติม")
        elif in_chol >= 200:
            alerts.append("🟡 <b>คอเลสเตอรอลเริ่มสูงเกินเกณฑ์:</b> ระดับคอเลสเตอรอล " + str(in_chol) + " mg/dL ควรควบคุมอาหารที่มีไขมันอิ่มตัวสูงและของทอด")
            
        if YESNO_MAP[in_fbs] == 1:
            alerts.append("🟡 <b>น้ำตาลในเลือดขณะอดอาหารสูง (>120 mg/dL):</b> เข้าข่ายเสี่ยงเบาหวาน ซึ่งเป็นปัจจัยเร่งให้หลอดเลือดหัวใจเสื่อมสภาพเร็วขึ้น")
            
        if YESNO_MAP[in_angina] == 1:
            alerts.append("🔴 <b>มีอาการเจ็บหน้าอกขณะออกแรง (Exercise Angina):</b> เป็นสัญญาณเตือนสำคัญของภาวะกล้ามเนื้อหัวใจขาดเลือดชั่วคราว ควรหลีกเลี่ยงการออกแรงหนักเฉียบพลันและพบแพทย์อายุรกรรมหัวใจ")
            
        if in_oldpeak >= 2.0:
            alerts.append("🔴 <b>คลื่น ST-Depression สูง (" + str(in_oldpeak) + "):</b> บ่งชี้ถึงภาวะ Ischemia ของกล้ามเนื้อหัวใจที่รุนแรง")
            
        if not alerts:
            alerts.append("🟢 <b>ดัชนีสุขภาพโดยรวมอยู่ในเกณฑ์ดี:</b> ค่าความดัน คอเลสเตอรอล และคลื่นไฟฟ้าหัวใจไม่พบความผิดปกติที่เด่นชัด ควรรักษามาตรฐานการรับประทานอาหารและการออกกำลังกายสม่ำเสมอ")
            
        for alt in alerts:
            st.markdown(f'<div class="notice-box" style="margin-bottom:8px;">{alt}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="notice-warning">
            ⚠️ <b>ข้อจำกัดทางการแพทย์:</b> ระบบปัญญาประดิษฐ์นี้พัฒนาขึ้นเพื่อการวิเคราะห์และคัดกรองข้อมูลเบื้องต้นเพื่อการศึกษาเท่านั้น 
            ไม่สามารถนำมาใช้แทนการตรวจวินิจฉัย การตรวจเอกซเรย์คอมพิวเตอร์หลอดเลือดหัวใจ (CTA) หรือการฉีดสีสวนหลอดเลือดหัวใจ (CAG) โดยแพทย์ผู้เชี่ยวชาญได้
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# หน้าที่ 4: 📁 คัดกรองแบบกลุ่ม (Batch CSV Screening)
# ==============================================================================
elif menu == "📁 คัดกรองแบบกลุ่ม (Batch CSV)":
    st.markdown('<div class="section-header">📁 การคัดกรองผู้ป่วยความเสี่ยงโรคหัวใจแบบชุดข้อมูล (Batch Screening)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="notice-info">
        ฟังก์ชันนี้เหมาะสำหรับแพทย์หรือโรงพยาบาลที่ต้องการนำเข้าข้อมูลผลตรวจของผู้ป่วยจำนวนมากในครั้งเดียว (ไฟล์ CSV) เพื่อให้โมเดล KNN จำแนกความเสี่ยงพร้อมคำนวณเปอร์เซ็นต์แบบอัตโนมัติ
    </div>
    """, unsafe_allow_html=True)

    batch_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลผู้ป่วย (ต้องมีคอลัมน์ตามโครงสร้างที่กำหนด)", type=["csv"], key="batch_uploader")
    
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing_cols = [c for c in FEATURES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ ไฟล์ CSV ขาดคอลัมน์ที่จำเป็น: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ ตรวจพบข้อมูลผู้ป่วย {len(batch_df):,} รายการ พร้อมทำการวิเคราะห์")
                
                # ทำนายผล
                scaler = results["scaler"]
                knn = results["knn"]
                
                X_batch = batch_df[FEATURES].copy()
                X_batch_scaled = scaler.transform(X_batch)
                batch_preds = knn.predict(X_batch_scaled)
                batch_probs = knn.predict_proba(X_batch_scaled)
                
                output_batch = batch_df.copy()
                output_batch["ทำนายความเสี่ยง (Prediction)"] = [
                    "⚠️ มีความเสี่ยง (1)" if p == 1 else "💚 ปกติ (0)" for p in batch_preds
                ]
                output_batch["ความน่าจะเป็นพบโรค (%)"] = (batch_probs[:, 1] * 100).round(1)
                
                high_risk_count = sum(batch_preds == 1)
                low_risk_count = len(batch_preds) - high_risk_count
                
                st.write("")
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    render_metric_card("จำนวนผู้ป่วยที่ตรวจ", f"{len(batch_df):,} คน", "ในไฟล์ที่อัปโหลด", "👥")
                with b_col2:
                    render_metric_card("จัดเป็นกลุ่มเสี่ยงโรคหัวใจ", f"{high_risk_count:,} คน", f"{(high_risk_count/len(batch_df))*100:.1f}%", "⚠️")
                with b_col3:
                    render_metric_card("จัดเป็นกลุ่มปกติ", f"{low_risk_count:,} คน", f"{(low_risk_count/len(batch_df))*100:.1f}%", "💚")

                st.write("")
                st.markdown("##### 📑 ผลการจำแนกรายบุคคลทั้งหมด:")
                st.dataframe(output_batch, use_container_width=True)
                
                st.download_button(
                    "⬇️ ดาวน์โหลดผลการคัดกรองทั้งหมดเป็นไฟล์ CSV",
                    output_batch.to_csv(index=False).encode("utf-8-sig"),
                    "batch_heart_disease_predictions.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}")
    else:
        st.markdown("##### 💡 ตัวอย่างรูปแบบไฟล์ CSV ที่รองรับ:")
        st.caption("สามารถดาวน์โหลดเทมเพลตตัวอย่างด้านล่างเพื่อนำไปกรอกข้อมูลผู้ป่วย:")
        sample_template = active_df[FEATURES].head(5)
        st.dataframe(sample_template, use_container_width=True)
        st.download_button(
            "⬇️ ดาวน์โหลดตัวอย่างไฟล์เทมเพลต CSV",
            sample_template.to_csv(index=False).encode("utf-8-sig"),
            "sample_patient_template.csv",
            "text/csv"
        )


# ==============================================================================
# หน้าที่ 5: 📚 พจนานุกรมข้อมูลทางการแพทย์ (Clinical Reference & Dictionary)
# ==============================================================================
elif menu == "📚 พจนานุกรมข้อมูลทางการแพทย์":
    st.markdown('<div class="section-header">📚 พจนานุกรมข้อมูลและคำอธิบายดัชนีชี้วัดทางการแพทย์ (Medical Data Dictionary)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="notice-info">
        ตารางสรุปความหมายทางการแพทย์ เกณฑ์ค่ามาตรฐาน และรหัสตัวแปรที่ใช้ในการฝึกโมเดล K-Nearest Neighbors เพื่อให้เข้าใจที่มาและการนำไปใช้ประโยชน์
    </div>
    """, unsafe_allow_html=True)

    data_dict = [
        {"ชื่อตัวแปร": "Age", "ความหมายทางการแพทย์": "อายุของผู้ป่วย", "หน่วย / ค่า": "ปี (28 - 77)", "เกณฑ์มาตรฐาน": "อายุที่เพิ่มขึ้นเพิ่มความเสี่ยงโรคหลอดเลือดหัวใจ"},
        {"ชื่อตัวแปร": "Sex", "ความหมายทางการแพทย์": "เพศกำเนิด", "หน่วย / ค่า": "0 = หญิง, 1 = ชาย", "เกณฑ์มาตรฐาน": "เพศชายมีอัตราการเกิดโรคหลอดเลือดหัวใจสูงกว่าเพศหญิงก่อนวัยหมดประจำเดือน"},
        {"ชื่อตัวแปร": "ChestPainType", "ความหมายทางการแพทย์": "ประเภทอาการเจ็บแน่นหน้าอก", "หน่วย / ค่า": "1: ASY (ไม่มีอาการ), 2: ATA (เจ็บผิดปกติ), 3: NAP (แน่นทั่วไป), 4: TA (เจ็บแบบฉบับ)", "เกณฑ์มาตรฐาน": "ASY มักพบในกลุ่มขาดเลือดเงียบ ส่วน TA เป็นสัญญาณคลาสสิกของกล้ามเนื้อหัวใจขาดเลือด"},
        {"ชื่อตัวแปร": "RestingBP", "ความหมายทางการแพทย์": "ความดันโลหิตขณะพัก", "หน่วย / ค่า": "mmHg", "เกณฑ์มาตรฐาน": "ปกติ < 120 mmHg, ความดันโลหิตสูงเมื่อ ≥ 140 mmHg"},
        {"ชื่อตัวแปร": "Cholesterol", "ความหมายทางการแพทย์": "ระดับคอเลสเตอรอลในเลือด", "หน่วย / ค่า": "mg/dL", "เกณฑ์มาตรฐาน": "ปกติ < 200 mg/dL, เสี่ยงสูงเมื่อ ≥ 240 mg/dL"},
        {"ชื่อตัวแปร": "FastingBS", "ความหมายทางการแพทย์": "น้ำตาลในเลือดหลังงดอาหาร", "หน่วย / ค่า": "0 = ปกติ (≤120), 1 = สูง (>120)", "เกณฑ์มาตรฐาน": "ระดับน้ำตาลสูงบ่งชี้ถึงโรคเบาหวานซึ่งเป็นปัจจัยเสี่ยงหลักของโรคหัวใจ"},
        {"ชื่อตัวแปร": "RestingECG", "ความหมายทางการแพทย์": "ผลตรวจคลื่นไฟฟ้าหัวใจขณะพัก", "หน่วย / ค่า": "1 = LVH, 2 = Normal, 3 = ST", "เกณฑ์มาตรฐาน": "LVH = หัวใจห้องล่างซ้ายโต, ST = มีความผิดปกติของคลื่น ST-T"},
        {"ชื่อตัวแปร": "MaxHR", "ความหมายทางการแพทย์": "อัตราการเต้นหัวใจสูงสุดขณะออกแรง", "หน่วย / ค่า": "ครั้งต่อนาที (bpm)", "เกณฑ์มาตรฐาน": "ค่ามาตรฐานคำนวณคร่าวๆ จาก 220 - อายุ"},
        {"ชื่อตัวแปร": "ExerciseAngina", "ความหมายทางการแพทย์": "อาการเจ็บหน้าอกขณะออกกำลังกาย", "หน่วย / ค่า": "0 = ไม่มี, 1 = มี", "เกณฑ์มาตรฐาน": "การเกิดเจ็บหน้าอกขณะออกแรงบ่งชี้ว่าหลอดเลือดหัวใจส่งเลือดไปเลี้ยงไม่ทัน"},
        {"ชื่อตัวแปร": "Oldpeak", "ความหมายทางการแพทย์": "การลดต่ำลงของช่วง ST เทียบกับขณะพัก", "หน่วย / ค่า": "มิลลิเมตร (mm)", "เกณฑ์มาตรฐาน": "ค่าที่มากกว่า 1.5 - 2.0 mm มีความสัมพันธ์สูงกับภาวะกล้ามเนื้อหัวใจขาดเลือด"},
        {"ชื่อตัวแปร": "ST_Slope", "ความหมายทางการแพทย์": "ความชันของช่วง ST ขณะออกแรงสูงสุด", "หน่วย / ค่า": "1 = Down (ลาดลง), 2 = Flat (ราบ), 3 = Up (ลาดขึ้น)", "เกณฑ์มาตรฐาน": "Up ปกติดี, Flat และ Down มีความเสี่ยงสูงต่อโรคหลอดเลือดหัวใจตีบ"},
        {"ชื่อตัวแปร": "HeartDisease (Target)", "ความหมายทางการแพทย์": "ผลการวินิจฉัยภาวะโรคหัวใจ", "หน่วย / ค่า": "0 = ปกติ, 1 = ตรวจพบโรคหัวใจ", "เกณฑ์มาตรฐาน": "ตัวแปรเป้าหมายที่ใช้ในการจำแนกประเภท (Binary Classification)"}
    ]

    st.dataframe(pd.DataFrame(data_dict), use_container_width=True)

    st.markdown('<div class="section-header">⚙️ หลักการทำงานของ K-Nearest Neighbors (KNN) ทางการแพทย์</div>', unsafe_allow_html=True)
    st.markdown("""
    **ขั้นตอนวิธี K-Nearest Neighbors (KNN)** ทำงานโดย:
    1. **StandardScaler Normalization:** ทำการปรับสเกลข้อมูลทุกฟีเจอร์ให้อยู่ในรูปแบบ Z-score ($Mean = 0, Std = 1$) เพื่อไม่ให้ตัวแปรที่มีค่าตัวเลขมาก (เช่น คอเลสเตอรอล 200-500) มีอิทธิพลกลบตัวแปรที่มีสเกลเล็ก (เช่น Oldpeak 0-3 หรือ Sex 0-1)
    2. **Distance Calculation:** คำนวณระยะห่างระหว่างผู้ป่วยรายใหม่กับผู้ป่วยทุกคนในประวัติด้วยสมการระยะทาง เช่น Euclidean Distance:
       $$d(x, y) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$$
    3. **K-Nearest Neighbors Voting:** คัดเลือกเพื่อนบ้านที่ใกล้ที่สุดจำนวน $K$ คน และคำนวณการทำนายโดยให้น้ำหนักตามส่วนกลับของระยะทาง (Distance Weighting)
    """)

# -------------------- ส่วนท้ายหน้าเว็บ (Footer) --------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0;">
    💙 <b>HeartCare AI Diagnostic Assistant</b> • พัฒนาด้วย Streamlit, Scikit-Learn และ Plotly<br>
    จัดทำขึ้นเพื่อการศึกษาและการวิเคราะห์ข้อมูลทางการแพทย์ (Medical Machine Learning Project)
</div>
""", unsafe_allow_html=True)
