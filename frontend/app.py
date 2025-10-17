import streamlit as st
import requests
import base64
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Nhận dạng & Phân loại độ chín trái mít", page_icon="🍈", layout="wide")

st.title("🍈 Hệ thống nhận dạng & phân loại độ chín trái mít")
st.write("Ứng dụng YOLOv8 để phục vụ nông nghiệp thông minh.")

uploaded_file = st.file_uploader("Tải lên ảnh trái mít", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    if st.button("Phân tích ảnh"):
        with st.spinner("Đang xử lý..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()
            img_str = data["image"]
            pred_img = Image.open(io.BytesIO(base64.b64decode(img_str)))
            st.image(pred_img, caption="Kết quả nhận dạng", use_container_width=True)

            st.subheader("Kết quả phân loại:")
            for p in data["predictions"]:
                st.write(f"🔸 **{p['label']}** – Độ tin cậy: {p['confidence']*100:.1f}%")
        else:
            st.error("Lỗi khi gọi API.")
