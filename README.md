# 🥭 Nhận dạng và Phân loại Độ chín Trái Mít bằng YOLOv8  
### Ứng dụng trong Nông nghiệp Thông minh

---

## 📘 1. Giới thiệu

Dự án này nghiên cứu và ứng dụng **YOLOv8** để nhận dạng và phân loại **độ chín của trái mít**, phục vụ cho hệ thống **nông nghiệp thông minh**.  

Hệ thống được xây dựng dưới dạng ứng dụng web, bao gồm:
- **Backend (FastAPI)**: Chịu trách nhiệm nhận ảnh, chạy mô hình YOLOv8 và trả về kết quả nhận dạng (bounding box, nhãn, độ tin cậy).
- **Frontend (Streamlit)**: Cung cấp giao diện web thân thiện, cho phép người dùng tải ảnh lên và xem kết quả nhận dạng trực quan.

---

## 🧩 2. Cấu trúc Thư mục Dự án

```

jackfruit-yolov8/
│
├── backend/
│   ├── main.py                \# FastAPI backend
│   ├── models/
│   │   └── jackfruit\_yolov8s.pt \# Tệp mô hình YOLOv8 đã huấn luyện
│   └── **init**.py
│
├── frontend/
│   ├── app.py                 \# Streamlit giao diện
│   └── **init**.py
│
├── venv/                      \# Môi trường ảo Python (tự tạo)
│
├── requirements.txt           \# Danh sách thư viện cần cài đặt
└── README.md                  \# File hướng dẫn này

````

---

## ⚙️ 3. Yêu cầu Môi trường

- Python **>= 3.9**
- Đã cài **pip**
- GPU (tùy chọn) để tăng tốc xử lý cho YOLOv8.

---

## 🚀 4. Hướng dẫn Cài đặt và Khởi tạo

> ⚠️ **LƯU Ý QUAN TRỌNG:** Tất cả các lệnh dưới đây cần được thực hiện trong **Thư mục Gốc của Dự án** (`jackfruit-yolov8/`).

### 🧱 Bước 1: Tạo môi trường ảo (Virtual Environment)

```bash
python -m venv venv
````

### 🧱 Bước 2: Kích hoạt môi trường ảo

  * **Windows:**

  ` bash   venv\Scripts\activate    `

  * **macOS / Linux:**

  ` bash   source venv/bin/activate    `

> Khi kích hoạt thành công, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh.

### 🧱 Bước 3: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

-----

## ⚡ 5. Khởi chạy Hệ thống

Để hệ thống hoạt động, cần khởi động **Backend (FastAPI)** và **Frontend (Streamlit)** trên hai cửa sổ terminal khác nhau.

### 🔹 Bước 1: Khởi động Backend (FastAPI)

> **(Đảm bảo đang ở thư mục gốc và môi trường ảo đang hoạt động)**

Chạy lệnh để khởi động API:

```bash
uvicorn backend.main:app --reload
```

  * **Kiểm tra thành công:**
      * Terminal hiển thị: `Uvicorn running on http://127.0.0.1:8000`
      * Tru cập `http://127.0.0.1:8000/` trên trình duyệt sẽ thấy phản hồi JSON: `{"message": "Jackfruit YOLOv8 API is running!"}`

-----

### 🔹 Bước 2: Khởi động Frontend (Streamlit)

1.  Mở **một cửa sổ Terminal MỚI**
2.  **Kích hoạt lại** môi trường ảo (theo Bước 2 ở mục 4).
3.  Chạy ứng dụng giao diện:

<!-- end list -->

```bash
streamlit run frontend/app.py
```

  * **Truy cập ứng dụng:**
      * Terminal sẽ hiển thị: `Local URL: http://localhost:8501`
      * Mở trình duyệt và truy cập địa chỉ này để bắt đầu sử dụng.

-----

### 🔹 Bước 3: Cách sử dụng Ứng dụng Web

1.  Tru cập giao diện web tại `http://localhost:8501`.
2.  Sử dụng mục **Tải lên ảnh trái mít** để chọn một tệp hình ảnh (`.jpg`, `.jpeg`, `.png`).
3.  Nhấn nút **Phân tích ảnh**.
4.  Kết quả sẽ hiển thị:
      * Ảnh đã được nhận dạng (với bounding box).
      * Phân loại độ chín (`Xanh` / `Chín` / `Đang chín`) và độ tin cậy tương ứng.

-----

## 🧠 6. Thông tin Mô hình

  * **Mô hình:** `jackfruit_yolov8s.pt`
  * **Framework:** Ultralytics YOLOv8
  * **Mô hình đầu ra (JSON của API):**

<!-- end list -->

```json
{
  "predictions": [
    {"label": "Chín", "confidence": 0.94},
    {"label": "Xanh", "confidence": 0.88}
  ],
  "image": "<base64-encoded của ảnh đã vẽ bounding box>"
}
```

-----

## 🧾 7. Ghi chú và Khắc phục sự cố

  * **Không tìm thấy mô hình (.pt):** Đảm bảo file mô hình phải nằm đúng đường dẫn: `backend/models/jackfruit_yolov8s.pt`.
  * **Cảnh báo `use_column_width`:** Nếu gặp cảnh báo này trong Streamlit, hãy mở file `frontend/app.py` và thay thế `use_column_width=True` bằng `use_container_width=True`.

-----

## 👨‍🎓 8. Thông tin Đồ án

  * **Sinh viên thực hiện:** Nguyễn Văn Duy
  * **Đề tài:** Nghiên cứu và ứng dụng YOLOv8 trong nhận dạng và phân loại độ chín trái mít phục vụ nông nghiệp thông minh
  * **Giảng viên hướng dẫn:** (Điền tên giảng viên)

-----

## 🌱 9. Gợi ý Phát triển Mở rộng

  * **Nhận dạng thời gian thực (Real-time):** Mở rộng để nhận dạng trực tiếp qua camera hoặc video.
  * **Triển khai Cloud:** Đưa API lên các dịch vụ như **Render, HuggingFace, hoặc PythonAnywhere** để tạo demo trực tuyến.
  * **Tích hợp IoT:** Kết hợp với cảm biến để thu thập dữ liệu về môi trường, hỗ trợ ra quyết định trong nông nghiệp thông minh.

-----

```
```