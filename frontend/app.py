import streamlit as st
import requests
import base64
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Phân loại độ chín trái mít", page_icon="🍈", layout="wide")

st.title("🍈 Nhận dạng & phân loại độ chín trái mít")
st.write("Ứng dụng YOLOv8 – hỗ trợ lựa chọn giữa nhiều model và nguồn ảnh (file hoặc URL).")

# --- Session State ---
for key, default in {
    "uploaded_file": None,
    "image_url": "",
    "last_source": None,  # "upload" hoặc "url"
    "analysis_result": None,  # Lưu kết quả phân tích
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_analysis():
    """Reset kết quả mỗi khi đổi nguồn ảnh"""
    st.session_state.analysis_result = None


# --- Chọn model ---
model_choice = st.selectbox(
    "🔍 Chọn model YOLOv8:",
    ["YOLOv8n", "YOLOv8s"],
    index=1
)

# --- Tabs: upload hoặc URL ---
st.markdown("### 🖼️ Chọn nguồn ảnh:")
upload_tab, url_tab = st.tabs(["📁 Tải ảnh lên", "🌐 Dán URL ảnh"])

with upload_tab:
    uploaded_file = st.file_uploader("Chọn ảnh từ máy tính", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.image_url = ""
            st.session_state.last_source = "upload"
            reset_analysis()

with url_tab:
    url_input = st.text_input("Nhập URL ảnh trực tuyến", value=st.session_state.image_url)
    if url_input and url_input != st.session_state.image_url:
        st.session_state.image_url = url_input
        st.session_state.uploaded_file = None
        st.session_state.last_source = "url"
        reset_analysis()

# --- Hiển thị ảnh preview ---
if st.session_state.last_source == "upload" and st.session_state.uploaded_file:
    try:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="Ảnh tải lên", use_container_width=True)
    except Exception:
        st.warning("Không thể đọc ảnh tải lên.")
elif st.session_state.last_source == "url" and st.session_state.image_url:
    try:
        st.image(st.session_state.image_url, caption="Ảnh từ URL", use_container_width=True)
    except Exception:
        st.warning("⚠️ URL không hợp lệ hoặc không thể tải ảnh xem trước.")

# --- Gửi yêu cầu ---
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

# --- Hiển thị kết quả (nếu có) ---
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
