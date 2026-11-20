import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -------------------- ตั้งค่าหน้าเว็บ --------------------
st.set_page_config(
    page_title="HeartCare AI | KNN",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
.stApp { background: linear-gradient(135deg, #f5f8ff 0%, #eef4ff 55%, #f8fbff 100%); }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px; }
.hero {
  background: linear-gradient(115deg, #102a56 0%, #176b87 55%, #24a4a5 100%);
  padding: 28px 32px; border-radius: 22px; color: white; margin-bottom: 22px;
  box-shadow: 0 12px 30px rgba(16,42,86,.15);
}
.hero h1 { color: white; font-size: 2rem; margin: 0 0 6px 0; font-weight: 700; }
.hero p { color: #e3f4ff; margin: 0; font-size: 1rem; }
.metric-card {
  background: white; border: 1px solid #e4ebf5; border-radius: 16px;
  padding: 18px 20px; box-shadow: 0 5px 18px rgba(30,60,100,.05);
}
.metric-label { color: #65758b; font-size: .88rem; }
.metric-value { color: #173b69; font-size: 1.8rem; font-weight: 700; }
.section-title { color: #173b69; font-weight: 700; font-size: 1.25rem; margin: 12px 0; }
.info-box { background: #eaf5ff; border-left: 4px solid #2487c9; padding: 12px 15px; border-radius: 8px; color: #24486b; }
.success-box { background: #eaf8f1; border-left: 4px solid #20a36b; padding: 14px 16px; border-radius: 9px; }
.warning-box { background: #fff5e7; border-left: 4px solid #e99b28; padding: 12px 15px; border-radius: 8px; }
div[data-testid="stForm"] { background: white; padding: 20px; border-radius: 16px; border: 1px solid #e4ebf5; }
[data-testid="stSidebar"] { background: #f0f5fc; }
.stButton>button, .stFormSubmitButton>button { border-radius: 10px; font-weight: 600; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "Heart3.csv"
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]
TARGET = "HeartDisease"

# ค่ารหัสหมวดหมู่ตามรูปแบบข้อมูลที่ใช้ในชุดข้อมูล Heart Failure Prediction
SEX_MAP = {"หญิง (0)": 0, "ชาย (1)": 1}
CHEST_MAP = {
    "ASY - ไม่มีอาการเจ็บหน้าอกแบบ anginal (1)": 1,
    "ATA - เจ็บหน้าอกแบบ atypical angina (2)": 2,
    "NAP - เจ็บหน้าอกที่ไม่ใช่ angina (3)": 3,
    "TA - เจ็บหน้าอกแบบ typical angina (4)": 4
}
ECG_MAP = {
    "LVH - ภาวะหัวใจห้องล่างซ้ายโต (1)": 1,
    "Normal - ปกติ (2)": 2,
    "ST - ความผิดปกติของ ST-T (3)": 3
}
YESNO_MAP = {"ไม่ (0)": 0, "ใช่ (1)": 1}
SLOPE_MAP = {"Down - ลาดลง (1)": 1, "Flat - ราบ (2)": 2, "Up - ลาดขึ้น (3)": 3}

@st.cache_data
def load_default_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("ไม่พบไฟล์ Heart3.csv ในโฟลเดอร์เดียวกับ app.py")
    return pd.read_csv(DATA_PATH)

def validate_data(data):
    missing_cols = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing_cols:
        raise ValueError("ไม่พบคอลัมน์ที่จำเป็น: " + ", ".join(missing_cols))
    clean = data[FEATURES + [TARGET]].copy()
    for c in clean.columns:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    if clean.isna().any().any():
        raise ValueError("พบค่าว่างหรือค่าที่ไม่ใช่ตัวเลขในคอลัมน์ที่จำเป็น กรุณาตรวจสอบไฟล์")
    if not set(clean[TARGET].unique()).issubset({0, 1}):
        raise ValueError("คอลัมน์ HeartDisease ต้องมีค่าเป็น 0 หรือ 1 เท่านั้น")
    if clean[TARGET].nunique() < 2:
        raise ValueError("ข้อมูลต้องมีผลลัพธ์ทั้งสองกลุ่ม (0 และ 1) เพื่อฝึกโมเดล")
    return clean

@st.cache_resource
def train_knn(data_hash, X_values, y_values, k, test_size, random_state):
    X = pd.DataFrame(X_values, columns=FEATURES)
    y = pd.Series(y_values, name=TARGET).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if k > len(X_train):
        raise ValueError(f"ค่า K ต้องไม่เกินจำนวนข้อมูลฝึก ({len(X_train)})")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance", metric="minkowski", p=2))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return model, X_test, y_test, pred, proba, len(X_train), len(X_test)

def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )

# -------------------- โหลดข้อมูล --------------------
try:
    default_df = validate_data(load_default_data())
except Exception as e:
    st.error(f"เกิดปัญหาในการโหลดข้อมูล: {e}")
    st.stop()

st.markdown("""
<div class="hero">
  <h1>💙 HeartCare AI</h1>
  <p>ระบบจำแนกความเสี่ยงโรคหัวใจด้วย Machine Learning เทคนิค K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💙 HeartCare AI")
    st.caption("ระบบวิเคราะห์ข้อมูลเพื่อการศึกษา")
    page = st.radio("เมนูหลัก", ["ภาพรวมข้อมูล", "ฝึกและประเมินโมเดล", "ทำนายข้อมูลใหม่"], index=0)
    st.divider()
    st.markdown("### ⚙️ ตั้งค่าโมเดล")
    k_value = st.slider("จำนวนเพื่อนบ้าน (K)", min_value=1, max_value=25, value=5, step=1)
    test_percent = st.slider("สัดส่วนข้อมูลทดสอบ (%)", 10, 40, 20, 5)
    seed = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
    st.caption("โมเดลใช้ StandardScaler และระยะทาง Euclidean โดยให้น้ำหนักเพื่อนบ้านตามระยะทาง")

# ให้ผู้ใช้เลือกไฟล์ข้อมูลเสริมได้ โดยไม่เปลี่ยนไฟล์ต้นฉบับ
with st.sidebar:
    uploaded = st.file_uploader("อัปโหลด CSV ชุดข้อมูล (ถ้าต้องการ)", type=["csv"])
    if uploaded is not None:
        try:
            active_df = validate_data(pd.read_csv(uploaded))
            st.success("โหลดชุดข้อมูลที่อัปโหลดแล้ว")
        except Exception as e:
            st.error(f"ไฟล์ไม่ถูกต้อง: {e}")
            active_df = default_df
    else:
        active_df = default_df

try:
    active_df = validate_data(active_df)
    if len(active_df) < 10:
        raise ValueError("ควรมีข้อมูลอย่างน้อย 10 แถวเพื่อแบ่งชุดฝึกและทดสอบ")
    if active_df[TARGET].value_counts().min() < 2:
        raise ValueError("แต่ละกลุ่มผลลัพธ์ต้องมีอย่างน้อย 2 ตัวอย่าง")
    model, X_test, y_test, y_pred, y_proba, n_train, n_test = train_knn(
        str(pd.util.hash_pandas_object(active_df, index=True).sum()),
        active_df[FEATURES].values.tolist(),
        active_df[TARGET].values.tolist(),
        int(k_value), test_percent / 100, int(seed)
    )
except Exception as e:
    st.error(f"ไม่สามารถฝึกโมเดลได้: {e}")
    st.stop()

# -------------------- หน้าแสดงผล --------------------
if page == "ภาพรวมข้อมูล":
    st.markdown('<div class="section-title">📊 ภาพรวมชุดข้อมูล</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("จำนวนข้อมูลทั้งหมด", f"{len(active_df):,}")
    with c2: metric_card("ตัวแปรนำเข้า", str(len(FEATURES)))
    with c3: metric_card("กลุ่มเป้าหมาย", "2 กลุ่ม")
    with c4: metric_card("ข้อมูลที่ใช้ฝึก", f"{n_train:,}")
    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### สัดส่วนผลลัพธ์ HeartDisease")
        counts = active_df[TARGET].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(counts.values, labels=["0: ไม่พบโรค", "1: พบโรค"], autopct="%1.1f%%",
               startangle=90, wedgeprops={"width": 0.42, "edgecolor": "white"})
        ax.set_title("HeartDisease Distribution")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### สถิติข้อมูลตัวเลข")
        st.dataframe(active_df[FEATURES].describe().T.round(2), use_container_width=True)
    st.markdown("#### ตัวอย่างข้อมูล")
    st.dataframe(active_df.head(10), use_container_width=True)
    st.markdown('<div class="info-box">หมายเหตุ: ไฟล์ CSV ไม่มีคำอธิบายรหัสหมวดหมู่กำกับไว้ หน้าแบบฟอร์มจึงใช้การจับคู่รหัสตามรูปแบบมาตรฐานของชุดข้อมูล Heart Failure Prediction โปรดตรวจสอบกับแหล่งที่มาของข้อมูลก่อนใช้งานจริง</div>', unsafe_allow_html=True)

elif page == "ฝึกและประเมินโมเดล":
    st.markdown('<div class="section-title">🧠 ผลการฝึกและประเมินโมเดล KNN</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">กำลังใช้ K = <b>{k_value}</b> | แบ่งข้อมูลฝึก {100-test_percent}% และข้อมูลทดสอบ {test_percent}% | StandardScaler + KNN (Euclidean)</div>', unsafe_allow_html=True)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Accuracy", f"{acc*100:.2f}%")
    with m2: metric_card("Precision", f"{prec*100:.2f}%")
    with m3: metric_card("Recall", f"{rec*100:.2f}%")
    with m4: metric_card("F1-score", f"{f1*100:.2f}%")
    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["ทำนาย 0", "ทำนาย 1"])
        ax.set_yticks([0, 1], ["จริง 0", "จริง 1"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "#173b69", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=.046)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        st.markdown("#### รายงานผลการจำแนก")
        report = classification_report(
            y_test, y_pred, labels=[0, 1],
            target_names=["ไม่พบโรค (0)", "พบโรค (1)"],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.markdown("#### เปรียบเทียบผลจริงกับผลทำนาย")
    result_df = X_test.copy()
    result_df["ค่าจริง"] = y_test.values
    result_df["ค่าทำนาย"] = y_pred
    result_df["ผลการจำแนก"] = np.where(y_test.values == y_pred, "ถูกต้อง", "ไม่ถูกต้อง")
    st.dataframe(result_df.head(30), use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลดผลการทำนาย (CSV)", result_df.to_csv(index=False).encode("utf-8-sig"),
                       "knn_test_predictions.csv", "text/csv")

else:
    st.markdown('<div class="section-title">🩺 ทำนายข้อมูลใหม่</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">กรอกข้อมูลในแบบฟอร์มด้านล่าง ระบบจะนำข้อมูลไปผ่าน StandardScaler และโมเดล KNN ที่ฝึกจากชุดข้อมูลที่เลือกไว้</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown("#### ข้อมูลทั่วไป")
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=50)
            sex_label = st.selectbox("เพศ", list(SEX_MAP.keys()))
            chest_label = st.selectbox("ประเภทอาการเจ็บหน้าอก", list(CHEST_MAP.keys()))
            resting_bp = st.number_input("ความดันโลหิตขณะพัก (mmHg)", min_value=0, max_value=300, value=120)
        with b:
            cholesterol = st.number_input("คอเลสเตอรอล (mg/dL)", min_value=0, max_value=1000, value=200)
            fasting_label = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dL", list(YESNO_MAP.keys()))
            ecg_label = st.selectbox("ผลคลื่นไฟฟ้าหัวใจขณะพัก", list(ECG_MAP.keys()))
            max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (ครั้ง/นาที)", min_value=30, max_value=250, value=150)
        with c:
            angina_label = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย", list(YESNO_MAP.keys()))
            oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=15.0, value=0.0, step=0.1, format="%.1f")
            slope_label = st.selectbox("ST_Slope", list(SLOPE_MAP.keys()))
            st.caption("กรุณากรอกค่าตามผลตรวจและหน่วยที่ระบุ")
        submitted = st.form_submit_button("🔍 วิเคราะห์และทำนาย", use_container_width=True, type="primary")

    if submitted:
        new_row = pd.DataFrame([{
            "Age": age, "Sex": SEX_MAP[sex_label],
            "ChestPainType": CHEST_MAP[chest_label],
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": YESNO_MAP[fasting_label],
            "RestingECG": ECG_MAP[ecg_label], "MaxHR": max_hr,
            "ExerciseAngina": YESNO_MAP[angina_label],
            "Oldpeak": oldpeak, "ST_Slope": SLOPE_MAP[slope_label]
        }], columns=FEATURES)
        prediction = int(model.predict(new_row)[0])
        probabilities = model.predict_proba(new_row)[0]
        classes = list(model.named_steps["knn"].classes_)
        probability_map = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        risk_probability = probability_map.get(1, 0.0)
        st.divider()
        st.markdown("### ผลการจำแนกจากโมเดล")
        if prediction == 1:
            st.markdown('<div class="warning-box"><b>ผลการจำแนก: กลุ่มพบโรคหัวใจ (Class 1)</b><br>โมเดลจัดข้อมูลนี้ให้อยู่ในกลุ่มตัวอย่างที่มี HeartDisease = 1</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box"><b>ผลการจำแนก: กลุ่มไม่พบโรคหัวใจ (Class 0)</b><br>โมเดลจัดข้อมูลนี้ให้อยู่ในกลุ่มตัวอย่างที่มี HeartDisease = 0</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            metric_card("ความน่าจะเป็นของกลุ่ม 0", f"{probability_map.get(0, 0)*100:.2f}%")
        with p2:
            metric_card("ความน่าจะเป็นของกลุ่ม 1", f"{risk_probability*100:.2f}%")
        st.caption("ค่าความน่าจะเป็นคำนวณจากน้ำหนักระยะทางของเพื่อนบ้านใน KNN ไม่ใช่ความน่าจะเป็นทางการแพทย์")
        st.markdown("#### ข้อมูลที่ส่งเข้าโมเดล")
        st.dataframe(new_row, use_container_width=True)
        st.markdown('<div class="warning-box">ข้อจำกัดสำคัญ: ผลลัพธ์นี้เป็นการจำแนกจากชุดข้อมูลเพื่อการศึกษา ไม่ใช่การวินิจฉัยโรคหรือคำแนะนำทางการแพทย์ ห้ามใช้แทนการตรวจโดยแพทย์ หากมีอาการผิดปกติควรพบแพทย์</div>', unsafe_allow_html=True)

st.divider()
st.caption("HeartCare AI • K-Nearest Neighbors • พัฒนาด้วย Streamlit | ใช้เพื่อการศึกษาเท่านั้น")
