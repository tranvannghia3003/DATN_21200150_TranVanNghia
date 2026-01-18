import io
import os
import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from process_image import ImageProcessor
from logger import info, error

# ===================== APP =====================
DEFAULT_SERVER_URL = os.getenv("OPENAPI_SERVER_URL", "http://localhost:5000")
app = FastAPI(
    title="Face Recognition API",
    servers=[{"url": DEFAULT_SERVER_URL}],
)

# ===================== CORS =====================
origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5000,http://127.0.0.1:5000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== PATHS =====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT_DIR, "web")
BUILD_DIR = os.path.join(ROOT_DIR, "build")

os.makedirs(BUILD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

# ===================== MODEL =====================
yolo_path = "best.pt"
if not os.path.exists(yolo_path):
    raise RuntimeError(" Không tìm thấy YOLO model best.pt")

image_processor = ImageProcessor(yolo_path)

# ===================== ATTENDANCE STORE =====================
attendance_logs = []   # RAM-only (đủ cho đồ án)

# ===================== CAMERA (OPTIONAL) =====================
def generate_frames():
    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam.isOpened():
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cam.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ===================== HEALTH =====================
@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== UPLOAD =====================
@app.post("/upload")
def upload_image(image: UploadFile = File(...)):
    try:
        img_bytes = image.file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        result = image_processor.process_image(pil_img)
        return result

    except Exception as e:
        error(str(e))
        raise HTTPException(status_code=500, detail="Upload failed")

# ===================== ATTENDANCE =====================
@app.post("/attendance")
def save_attendance(log: dict = Body(...)):
    name = log.get("name")
    
    # Lọc người lạ
    if not name or name in ["Unknown", "Nguoi la"]:
        return {"status": "ignored"}

    # Lấy thời gian hiện tại làm mặc định
    now_str = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

    attendance_logs.append({
        # Nếu frontend gửi time thì dùng, không thì dùng now_str
        "time": log.get("time") if log.get("time") else now_str,
        "name": name,
        "conf": float(log.get("conf", 0)),
        "raw": log.get("raw", {})
    })

    return {"status": "saved"}

@app.get("/attendance/list")
def list_attendance():
    return attendance_logs

# ===================== RETRIEVE =====================
@app.post("/retrieve")
def retrieve_image(
    image: UploadFile = File(...),
    customerName: str = Form(...)
):
    try:
        img_bytes = image.file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        processor = ImageProcessor(yolo_path, customerName)
        return processor.retrieve_image(pil_img)

    except Exception as e:
        error(str(e))
        raise HTTPException(status_code=500, detail="Retrieve failed")

# ===================== RUN =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=False
    )
