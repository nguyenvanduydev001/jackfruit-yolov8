from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import base64

app = FastAPI()

# Cấu hình CORS để Streamlit có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO("backend/models/jackfruit_yolov8s.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)
    result = results[0]

    # Xuất ảnh có bounding box
    img_res = result.plot()  # numpy array (BGR)
    img_pil = Image.fromarray(img_res[..., ::-1])  # đổi sang RGB để hiển thị

    # Encode ảnh sang base64 để gửi cho frontend
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Lấy nhãn và độ tin cậy
    predictions = []
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = result.names[cls_id]
        predictions.append({
            "label": label,
            "confidence": round(conf, 3)
        })

    return {"predictions": predictions, "image": img_str}
@app.get("/")
def root():
    return {"message": "Jackfruit YOLOv8 API is running!"}

