from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, base64, os, requests

app = FastAPI()

# Cho phép truy cập từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Các model khả dụng
MODEL_PATHS = {
    "YOLOv8n": os.path.join("backend", "models", "jackfruit_yolov8n.pt"),
    "YOLOv8s": os.path.join("backend", "models", "jackfruit_yolov8s.pt"),
}

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
    model_name: str = Form("YOLOv8s"),
    file: UploadFile = File(None),
    image_url: str = Form(None)
):
    # Xử lý ảnh từ file hoặc URL
    if file:
        image_bytes = await file.read()
    elif image_url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(image_url, headers=headers, timeout=10, verify=False)
            if response.status_code != 200:
                return {"error": f"Không thể tải ảnh từ URL. Mã lỗi: {response.status_code}"}
            image_bytes = response.content
        except Exception as e:
            return {"error": f"Lỗi khi tải ảnh: {str(e)}"}
    else:
        return {"error": "Chưa cung cấp ảnh hoặc URL."}

    model = get_model(model_name)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)
    result = results[0]

    # Xuất ảnh có bounding box
    img_res = result.plot()
    img_pil = Image.fromarray(img_res[..., ::-1])

    # Encode ảnh sang base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Danh sách dự đoán
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
