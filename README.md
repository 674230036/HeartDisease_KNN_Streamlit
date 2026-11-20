# 🫀 HeartCare AI — ระบบวิเคราะห์และจำแนกความเสี่ยงโรคหัวใจด้วย Machine Learning (KNN)

ระบบเว็บแอปพลิเคชันคัดกรองและประเมินความเสี่ยงโรคหัวใจระดับคลินิก พัฒนาด้วย **Streamlit**, **Scikit-Learn** และ **Plotly** โดยใช้เทคนิค **K-Nearest Neighbors (KNN)** พร้อมการปรับมาตรฐานข้อมูลด้วย **StandardScaler**

---

## 🌟 จุดเด่นและความสามารถของระบบ (Key Features)

1. **แก้ไขปัญหาฟอนต์ภาษาไทยสมบูรณ์แบบ 100% (No Tofu / Square Bug)**:
   - ฝังฟอนต์ `Prompt` (Google Fonts) เข้าในโฟลเดอร์ `fonts/` และลงทะเบียนฟอนต์เข้ากับ Matplotlib
   - ใช้ **Plotly Interactive Charts** เป็นกลไกหลักในการวาดกราฟ แสดงผลสระ วรรณยุกต์ และข้อความภาษาไทยได้ถูกต้อง ชัดเจน คมชัดทุกอุปกรณ์
   - เพิ่ม `fonts-thai-tlwg` ใน `packages.txt` รองรับการ Deploy บน Linux / Streamlit Community Cloud

2. **📊 ภาพรวมและสถิติข้อมูล (Clinical Dashboard & EDA)**:
   - KPI Cards สรุปตัวชี้วัดสำคัญ (ผู้ป่วยทั้งหมด 918 ราย, อัตราพบโรค, อายุเฉลี่ย, ความดันเฉลี่ย, คอเลสเตอรอลเฉลี่ย)
   - แกลเลอรีรูปภาพเปรียบเทียบสัณฐานวิทยาของหัวใจปกติและหัวใจที่มีพยาธิสภาพ
   - กราฟโดนัท (Interactive Donut Chart) แสดงสัดส่วนผู้ป่วยปกติ vs ผู้มีความเสี่ยงโรคหัวใจ
   - แผนภาพการกระจาย (Scatter Plot) วิเคราะห์ความสัมพันธ์ระหว่างอายุและอัตราการเต้นหัวใจสูงสุด (MaxHR)
   - กราฟแท่งจำแนกตามประเภทอาการเจ็บหน้าอก (Chest Pain Type) และค่าสัมประสิทธิ์สหสัมพันธ์ (Correlation)
   - ดูข้อมูลตัวอย่างและดาวน์โหลดชุดข้อมูลเป็น CSV

3. **🧠 ประเมินและปรับจูนโมเดล KNN (Model Lab & Optimization)**:
   - ปรับแต่งไฮเปอร์พารามิเตอร์ได้อิสระ: ค่า $K$, ระยะทาง (Euclidean, Manhattan, Minkowski), การถ่วงน้ำหนัก (Distance, Uniform), สัดส่วนชุดทดสอบ
   - ตัวชี้วัดระดับคลินิกครบครัน: **Accuracy**, **Sensitivity (Recall)**, **Precision**, **F1-Score**, **ROC-AUC**
   - **Interactive Confusion Matrix** พร้อมการแปลผลทางการแพทย์ (TN, FP, FN, TP)
   - **ROC Curve** พร้อมค่า AUC
   - **K-Optimization Curve** กราฟเปรียบเทียบหาค่า $K$ ที่แม่นยำที่สุด (ตั้งแต่ $K=1$ ถึง 25)
   - ตาราง Classification Report และรายการทำนายผลในชุดทดสอบ พร้อมปุ่มดาวน์โหลด CSV

4. **🩺 วินิจฉัยและคัดกรองรายบุคคล (Individual Patient Screening)**:
   - **⚡ ปุ่มพรีเซ็ตข้อมูลจำลอง 1-คลิก**: สุขภาพดี, เสี่ยงปานกลาง, เสี่ยงสูง (สะดวกในการทดสอบและนำเสนอ)
   - แบบฟอร์มกรอกข้อมูลพร้อมระบุหน่วยและเกณฑ์ปกติทางการแพทย์
   - แบนเนอร์สรุปผลพร้อมเกจวัดระดับความเสี่ยง (Risk Gauge)
   - **🔍 XAI (Explainable AI)**: แสดงรายชื่อและค่าตรวจของผู้ป่วย $K$ คนในฐานข้อมูลที่มีลักษณะใกล้เคียงที่สุด เพื่ออธิบายเหตุผลของการทำนาย
   - คำแนะนำด้านสุขภาพเฉพาะบุคคล (Personalized Medical Recommendations) อิงตามเกณฑ์ AHA/ACC

5. **📁 คัดกรองแบบกลุ่ม (Batch CSV Screening)**:
   - อัปโหลดไฟล์ CSV รายชื่อผู้ป่วยหลายรายเพื่อทำนายผลพร้อมกันแบบอัตโนมัติ
   - แสดงสถิติสรุปกลุ่มเสี่ยง vs ปกติ และดาวน์โหลดผลลัพธ์เป็นไฟล์ CSV

6. **📚 พจนานุกรมข้อมูลทางการแพทย์ (Medical Data Dictionary)**:
   - ตารางอธิบายความหมาย เกณฑ์ปกติ และความสำคัญทางคลินิกของทั้ง 11 ตัวแปร
   - คำอธิบายหลักการทำงานและสมการทางคณิตศาสตร์ของขั้นตอนวิธี KNN

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
HeartDisease_KNN_Streamlit/
├── app.py                 # โค้ดหลักของเว็บแอปพลิเคชัน Streamlit
├── Heart3.csv             # ฐานข้อมูลตัวอย่างผู้ป่วยโรคหัวใจ (918 รายการ)
├── requirements.txt       # รายการไลบรารี Python ที่ต้องติดตั้ง
├── packages.txt           # แพ็กเกจระบบสำหรับ Streamlit Cloud (fonts-thai-tlwg)
├── README.md              # เอกสารและคู่มือการใช้งานระบบ
├── fonts/                 # ฟอนต์ภาษาไทยสำหรับการแสดงผล
│   ├── Prompt-Regular.ttf
│   └── Prompt-Bold.ttf
└── img/                   # รูปภาพประกอบทางการแพทย์และอวตารผู้ป่วย
    ├── heart1.jpg
    ├── heart2.jpg
    ├── heart3.jpg
    ├── user_normal.jpg
    └── user_disease.jpg
```

---

## 🚀 วิธีการติดตั้งและรันบนเครื่องคอมพิวเตอร์ (Local Setup)

1. **ติดตั้งไลบรารีที่จำเป็น**:
   ```bash
   pip install -r requirements.txt
   ```

2. **สั่งรันแอปพลิเคชัน Streamlit**:
   ```bash
   streamlit run app.py
   ```

3. เปิดเว็บเบราว์เซอร์ไปที่ `http://localhost:8501`

---

## ☁️ วิธีการนำขึ้น Streamlit Community Cloud (Deployment)

1. Commit และ Push ไฟล์ทั้งหมดขึ้น GitHub Repository:
   ```bash
   git add .
   git commit -m "Enhance UI, fix Thai font tofu bug, and add full clinical features"
   git push origin main
   ```
2. เข้าสู่ระบบที่ [https://share.streamlit.io](https://share.streamlit.io)
3. เลือก **New app** และระบุ Repository, Branch (`main`) และ Main file path (`app.py`)
4. กด **Deploy!**

---

## ⚠️ ข้อควรระวังและจริยธรรมทางการแพทย์ (Medical Disclaimer)

ระบบปัญญาประดิษฐ์นี้พัฒนาขึ้นเพื่อการศึกษาและการวิเคราะห์ข้อมูลทางการแพทย์เบื้องต้น **ไม่สามารถใช้ทดแทนการตรวจวินิจฉัย การรักษา หรือดุลยพินิจของแพทย์ผู้เชี่ยวชาญด้านโรคหัวใจได้** หากมีอาการเจ็บแน่นหน้าอก เหนื่อยง่าย หรือสงสัยว่ามีความผิดปกติ ควรพบแพทย์ทันที
