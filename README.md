# HeartCare AI — KNN with Streamlit

เว็บแอปจำแนกข้อมูลโรคหัวใจจาก `Heart3.csv` ด้วย K-Nearest Neighbors (KNN)

## ไฟล์ในโปรเจกต์
- `app.py` — โค้ดเว็บแอป Streamlit
- `Heart3.csv` — ชุดข้อมูลต้นฉบับที่ใช้ฝึกและทดสอบ
- `requirements.txt` — ไลบรารีที่ต้องติดตั้ง
- `README.md` — คู่มือใช้งาน

## วิธีรันบนเครื่อง
1. ติดตั้ง Python 3.10 หรือใหม่กว่า
2. เปิด Terminal ในโฟลเดอร์โปรเจกต์
3. ติดตั้งไลบรารี:
   ```bash
   pip install -r requirements.txt
   ```
4. เริ่มแอป:
   ```bash
   streamlit run app.py
   ```

## วิธีนำขึ้น Streamlit Community Cloud
1. สร้าง GitHub repository ใหม่
2. อัปโหลด `app.py`, `Heart3.csv`, `requirements.txt` และ `README.md`
3. เข้า https://share.streamlit.io/ แล้วเชื่อมบัญชี GitHub
4. เลือก repository, branch และไฟล์หลัก `app.py`
5. กด Deploy

## ความสามารถ
- แสดงภาพรวมข้อมูลและสัดส่วนคลาส
- เลือกค่า K, สัดส่วนชุดทดสอบ และ random state
- ใช้ StandardScaler และ KNN (Euclidean distance, distance weighting)
- ประเมินด้วย Accuracy, Precision, Recall, F1-score, Confusion Matrix และ Classification Report
- อัปโหลด CSV ชุดข้อมูลอื่นได้ หากมีคอลัมน์ตามที่ระบบกำหนด
- กรอกข้อมูลใหม่ผ่านแบบฟอร์มและทำนายผลได้
- ดาวน์โหลดผลทำนายจากชุดทดสอบเป็น CSV

## โครงสร้างข้อมูลที่ต้องมี
คอลัมน์คุณลักษณะ: `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`

คอลัมน์เป้าหมาย: `HeartDisease` (0 หรือ 1)

หมายเหตุ: ไฟล์ CSV ไม่มี data dictionary ระบุรหัสหมวดหมู่โดยตรง แบบฟอร์มใช้รหัสตามรูปแบบที่พบโดยทั่วไปในชุดข้อมูล Heart Failure Prediction จึงควรตรวจสอบ mapping กับแหล่งที่มาของชุดข้อมูลก่อนนำไปใช้จริง

## ข้อควรระวัง
โครงการนี้เป็นตัวอย่างเพื่อการศึกษา ไม่ใช่อุปกรณ์หรือซอฟต์แวร์วินิจฉัยโรค ผลทำนายไม่ควรใช้ตัดสินใจรักษา
