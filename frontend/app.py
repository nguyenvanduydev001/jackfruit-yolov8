import streamlit as st
import numpy as np
import requests
import base64
import time
from PIL import Image
import io
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Phân loại độ chín trái mít", page_icon="🍈", layout="wide")
st.title("🍈 Nhận dạng & phân loại độ chín trái mít (YOLOv8)")
st.write("Ứng dụng YOLOv8 – hỗ trợ nhiều model và nguồn ảnh (file, URL, hoặc webcam).")

# --- Session State ---
for key, default in {
    "uploaded_file": None,
    "image_url": "",
    "last_source": None,
    "analysis_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def reset_analysis():
    st.session_state.analysis_result = None

# --- Chọn model ---
model_choice = st.selectbox("🔍 Chọn model YOLOv8:", ["YOLOv8n", "YOLOv8s"], index=1)

# --- Tabs: upload / URL / webcam ---
st.markdown("### 🖼️ Chọn nguồn ảnh:")
upload_tab, url_tab, cam_tab = st.tabs(["📁 Tải ảnh lên", "🌐 Dán URL ảnh", "📷 Webcam real-time"])

# ===================================
# 📁 Upload
# ===================================
with upload_tab:
    uploaded_file = st.file_uploader("Chọn ảnh từ máy tính", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.image_url = ""
            st.session_state.last_source = "upload"
            reset_analysis()

# ===================================
# 🌐 URL
# ===================================
with url_tab:
    url_input = st.text_input("Nhập URL ảnh trực tuyến", value=st.session_state.image_url)
    if url_input and url_input != st.session_state.image_url:
        st.session_state.image_url = url_input
        st.session_state.uploaded_file = None
        st.session_state.last_source = "url"
        reset_analysis()

# ===================================
# 📷 WEBCAM
# ===================================
with cam_tab:
    st.info("📸 Nhận dạng real-time từ webcam. Hiển thị FPS & độ trễ infer (ms).")

    # ✅ Thêm công tắc chế độ giảm độ trễ
    low_latency = st.checkbox("⚡ Bật Low Latency Mode (tăng tốc, giảm chất lượng)", value=False)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.fps = 0
            self.prev_time = 0
            self.latency_ms = 0

        def recv(self, frame):
            start_time = time.time()

            img = frame.to_ndarray(format="bgr24")

            # Nếu bật chế độ giảm độ trễ → resize ảnh nhỏ lại
            if low_latency:
                img = cv2.resize(img, (480, 360))

            # Mã hóa gửi đến backend
            _, img_encoded = cv2.imencode(".jpg", img)
            files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            data = {"model_name": model_choice}

            try:
                response = requests.post(API_URL, files=files, data=data, timeout=5)
                if response.status_code == 200:
                    resp_json = response.json()
                    img_bytes = base64.b64decode(resp_json["image"])
                    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                else:
                    img_np = img
            except Exception:
                img_np = img  # fallback khi mất kết nối

            # Tính FPS và độ trễ
            now = time.time()
            self.latency_ms = (now - start_time) * 1000
            self.fps = 1 / (now - self.prev_time) if self.prev_time else 0
            self.prev_time = now

            # Hiển thị FPS, Latency, Model, Mode
            color = (0, 255, 0) if not low_latency else (255, 200, 0)
            cv2.putText(img_np, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(img_np, f"Latency: {self.latency_ms:.0f} ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(img_np, f"Model: {model_choice}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_np, f"Mode: {'Low Latency' if low_latency else 'Normal'}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            return av.VideoFrame.from_ndarray(img_np, format="bgr24")

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="jackfruit-detect",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

# ===================================
# 🖼️ Hiển thị preview ảnh tĩnh
# ===================================
if st.session_state.last_source == "upload" and st.session_state.uploaded_file:
    image = Image.open(st.session_state.uploaded_file)
    st.image(image, caption="Ảnh tải lên", use_container_width=True)
elif st.session_state.last_source == "url" and st.session_state.image_url:
    st.image(st.session_state.image_url, caption="Ảnh từ URL", use_container_width=True)

# ===================================
# 🚀 Phân tích ảnh (upload / URL)
# ===================================
if st.button("🚀 Phân tích ảnh"):
    if not st.session_state.uploaded_file and not st.session_state.image_url:
        st.warning("Vui lòng tải ảnh hoặc nhập URL hợp lệ.")
    else:
        with st.spinner("Đang xử lý..."):
            files = {"file": st.session_state.uploaded_file.getvalue()} if st.session_state.uploaded_file else None
            data = {"model_name": model_choice, "image_url": st.session_state.image_url}
            try:
                response = requests.post(API_URL, files=files, data=data, timeout=30)
            except Exception as e:
                st.error(f"Lỗi khi gửi yêu cầu: {e}")
                st.stop()

        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                st.error(data["error"])
            else:
                st.session_state.analysis_result = data
        else:
            st.error(f"Lỗi kết nối API ({response.status_code})")

# ===================================
# 📊 Hiển thị kết quả
# ===================================
if st.session_state.analysis_result:
    data = st.session_state.analysis_result
    img_str = data["image"]
    pred_img = Image.open(io.BytesIO(base64.b64decode(img_str)))

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.last_source == "upload" and st.session_state.uploaded_file:
            st.image(Image.open(st.session_state.uploaded_file), caption="Ảnh gốc", use_container_width=True)
        elif st.session_state.last_source == "url" and st.session_state.image_url:
            st.image(st.session_state.image_url, caption="Ảnh gốc", use_container_width=True)
    with col2:
        st.image(pred_img, caption=f"Kết quả ({data['model_used']})", use_container_width=True)

    st.subheader("📊 Kết quả phân loại:")
    for p in data["predictions"]:
        st.write(f"🔸 **{p['label']}** – Độ tin cậy: {p['confidence']*100:.1f}%")
