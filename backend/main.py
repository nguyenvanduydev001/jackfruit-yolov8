from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, base64, os

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khai báo các model có sẵn
MODEL_PATHS = {
    "YOLOv8n": os.path.join("backend", "models", "jackfruit_yolov8n.pt"),
    "YOLOv8s": os.path.join("backend", "models", "jackfruit_yolov8s.pt")
}

# Cache model để không load lại nhiều lần
loaded_models = {}

def get_model(model_name: str):
    if model_name not in loaded_models:
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' không tồn tại.")
        loaded_models[model_name] = YOLO(MODEL_PATHS[model_name])
    return loaded_models[model_name]


@app.get("/")
def root():
    return {"message": "Jackfruit YOLOv8 API is running!"}


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("YOLOv8s")
):
    model = get_model(model_name)
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)
    result = results[0]

    # Xuất ảnh có bounding box
    img_res = result.plot()
    img_pil = Image.fromarray(img_res[..., ::-1])

    # Encode ảnh sang base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Tạo danh sách dự đoán
    predictions = []
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = result.names[cls_id]
        predictions.append({
            "label": label,
            "confidence": round(conf, 3)
        })

    return {"model_used": model_name, "predictions": predictions, "image": img_str}
