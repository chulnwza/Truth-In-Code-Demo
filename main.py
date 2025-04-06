from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
from features import extract_features_from_code  # ฟังก์ชันที่เราสร้างไว้ใน features.py

app = FastAPI()

# CORS สำหรับให้ frontend ใช้งานได้ (ถ้ามี React หรือ HTML ภายนอก)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # เปลี่ยนเป็นเฉพาะ domain ถ้าจะปลอดภัยมากขึ้น
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล
model = joblib.load("app/model/truth_model_ex1.pkl")

# Templates สำหรับหน้าเว็บ
templates = Jinja2Templates(directory="templates")


# หน้าเว็บหลัก
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(code: str = Form(...)):
    try:
        features_df = extract_features_from_code(code)
        prediction = model.predict(features_df)[0]
        label = "ChatGPT 🤖" if prediction == 1 else "Human 👨‍💻"
        print("🔁 label:", label)  # <== เพิ่มบรรทัดนี้
        return JSONResponse({"prediction": label})
    except Exception as e:
        print("❌ error:", str(e))  # <== เพิ่มเช็ค error
        return JSONResponse({"error": str(e)}, status_code=500)
