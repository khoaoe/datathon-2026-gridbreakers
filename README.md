# Datathon 2026 — The Gridbreakers (VinTelligence)

> Phân tích dữ liệu thương mại điện tử thời trang Việt Nam & dự báo doanh thu/giá vốn hàng ngày  
> **Cuộc thi:** VinTelligence Datathon 2026 - The Gridbreakers · Vòng 1
> **Đội thi:** *Liên Minh Gridbreakers: Những Kẻ Tái Định Nghĩa Thực Tại Dữ Liệu và Kiến Tạo Tương Lai Cho Doanh Nghiệp*

---

## Tổng quan

Dự án này giải quyết ba phần thi của VinTelligence Datathon 2026:

| Phần thi | Trọng số | Mô tả |
|----------|----------|-------|
| Trắc nghiệm (MCQ) | 20 điểm | Kiến thức nền tảng về khoa học dữ liệu |
| Phân tích khám phá (EDA) | 60 điểm | 5 chủ đề phân tích đa bảng, mỗi chủ đề kết hợp 3–4 bảng dữ liệu |
| Dự báo doanh thu | 20 điểm | Dự báo doanh thu thuần & giá vốn hàng ngày cho 548 ngày (01/2023 – 07/2024) |

**Kết quả chính:**
- Leaderboard MAE tốt nhất: **791.764** (EX-51 Bridge w15)
- Tổng tác động kinh doanh từ 6 đề xuất EDA: **≈255 triệu VND/năm** (21,8% doanh thu 2022)

---

## Cấu trúc thư mục

```
datathon-2026-gridbreakers/
├── data/                          # Dữ liệu gốc (15 bảng CSV)
├── data_cleaning/                 # Script tiền xử lý dữ liệu
├── 03_Forecasting/                # Mã nguồn cho Phần 3: Mô hình Dự báo doanh thu
│   ├── config.py                  #   Cấu hình đường dẫn, siêu tham số
│   ├── feature_engineering.py     #   Trích xuất đặc trưng v3 (875 dòng)
│   ├── tracker.py                 #   Theo dõi thực nghiệm
│   ├── utils.py                   #   Hàm tiện ích
│   ├── ex_01_naive_baseline.py    #   EX-01: Naive seasonal baseline
│   ├── ex_03_lgbm.py              #   EX-03: LightGBM recursive
│   ├── ...                        #   52 biến thể thực nghiệm
│   └── ex_52_recalibrated.py      #   EX-52: Monthly recalibration
├── notebook/
│   ├── 00_Data_QA_and_Integrity.ipynb # Kiểm tra chất lượng dữ liệu
│   ├── 01_MCQ_Answers.ipynb           # PHẦN 1:Trả lời trắc nghiệm
│   ├── 02_EDA.ipynb                   # PHẦN 2: Trực quan hóa và phân tích dữ liệu
│   └── 02_Data_Storytelling.ipynb     # PHẦN 2: Kể câu chuyện kinh doanh bằng dữ liệu
├── output/
│   ├── models/                    # Mô hình đã huấn luyện (pkl)
│   ├── submissions/               # File nộp bài (CSV)
│   └── tracking/                  # Log thực nghiệm
├── report/
│   ├── figures/                   # Hình ảnh trong báo cáo
│   ├── report.tex                 # Mã nguồn LaTeX cho BÁO CÁO
│   └── report.pdf                 # BÁO CÁO
└── README.md
```

---

## Phương pháp

### Phân tích khám phá (EDA)

5 chủ đề phân tích, mỗi chủ đề kết hợp 3–4 bảng dữ liệu theo khung Descriptive → Diagnostic → Predictive → Prescriptive:

1. **Nghịch lý khuyến mãi** — cơ chế fixed discount xói mòn biên lợi nhuận gộp (−63,3% GM ở phân khúc Performance)
2. **Tín hiệu số làm proxy doanh thu** — conversion rate và sessions là chỉ báo đồng thời (Pearson r = 0,44), không phải dự báo
3. **Tồn kho: thời điểm, không phải số lượng** — 97,9% stockout chỉ kéo dài ≤2 ngày (transition-gap), trùng đỉnh Q2
4. **Hành vi khách hàng RFM × địa lý** — miền Trung có AOV cao nhất (+14,5%) nhưng thị phần thấp; 27,7% khách Never Purchased
5. **Giao hàng × đánh giá** — tốc độ giao hàng chênh lệch chỉ 0,013 sao; driver thật sự là chất lượng sản phẩm

### Hệ thống dự báo

| Giai đoạn | Mô tả | MAE |
|-----------|-------|-----|
| Baseline | Naive seasonal average | 1.247.026 |
| EX-03 | LightGBM recursive + FE v2 | 973.611 |
| EX-22 | Deep FE + holiday distance features | 796.018 |
| EX-24 | + Double date event decomposition | 795.838 |
| **EX-51 Bridge w15** | **Hybrid: 85% recursive + 15% stateless** | **791.764** |

**Kiến trúc lõi:**
- **LightGBM** với 100+ đặc trưng (lịch, Fourier, biến trễ, hồ sơ lịch sử, khuyến mãi)
- **Bridge blending** — kết hợp mô hình đệ quy (giữ mức doanh thu cơ sở) với mô hình phi trạng thái (ổn định cấu trúc mùa vụ) để kiểm soát tích lũy sai số đệ quy
- **SHAP explainability** — phân tích đóng góp đặc trưng qua SHAP values, LightGBM gain, beeswarm, và partial dependence

---

## Cài đặt & chạy

```bash
# Clone repository
git clone https://github.com/khoaoe/datathon-2026-gridbreakers.git
cd datathon-2026-gridbreakers

# Tạo môi trường conda
conda create -n datathon python=3.11 -y
conda activate datathon

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy thực nghiệm chính
python -m modeling.ex_03_lgbm

# Tạo biểu đồ explainability
cd report && python generate_all_explainability.py
```

---


## Thành viên

| Tên | Vai trò |
|-----|---------|
| **Nguyễn Ngọc Khoa** | Trưởng đội · Modeling & Feature Engineering |
| Nguyễn Thiên Ấn | EDA & Business Analysis |
| Lê Công Minh | Data Cleaning & Visualization |
| Lê Nguyên Khang | MCQ & Report Writing |

---

## Giấy phép

Dự án này được phát triển cho mục đích thi đấu VinTelligence Datathon 2026.
