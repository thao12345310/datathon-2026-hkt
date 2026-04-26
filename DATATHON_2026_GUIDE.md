# 🏁 DATATHON 2026 — THE GRIDBREAKERS
### Hướng dẫn Toàn diện & Kế hoạch Thực thi Vòng 1
*VinTelligence × VinUniversity Data Science & AI Club*

---

## 📋 MỤC LỤC

1. [Tổng quan cuộc thi](#1-tổng-quan-cuộc-thi)
2. [Cấu trúc dữ liệu](#2-cấu-trúc-dữ-liệu)
3. [Yêu cầu 3 phần thi](#3-yêu-cầu-3-phần-thi)
4. [Thang điểm chi tiết](#4-thang-điểm-chi-tiết)
5. [Kiến thức & Kỹ năng cần có](#5-kiến-thức--kỹ-năng-cần-có)
6. [Công cụ & Thư viện](#6-công-cụ--thư-viện)
7. [Kế hoạch thực thi chi tiết](#7-kế-hoạch-thực-thi-chi-tiết)
8. [Hướng dẫn nộp bài](#8-hướng-dẫn-nộp-bài)
9. [Checklist cuối cùng](#9-checklist-cuối-cùng)

---

## 1. TỔNG QUAN CUỘC THI

| Mục | Nội dung |
|-----|----------|
| **Tên cuộc thi** | Datathon 2026 — The Gridbreakers |
| **Tổ chức** | VinTelligence & VinUni DS&AI Club |
| **Chủ đề** | Phân tích dữ liệu doanh nghiệp thời trang thương mại điện tử Việt Nam |
| **Giai đoạn dữ liệu** | 04/07/2012 – 31/12/2022 (train), 01/01/2023 – 01/07/2024 (test) |
| **Kaggle Link** | https://www.kaggle.com/competitions/datathon-2026-round-1 |
| **Vòng chung kết** | 23/05/2026 tại Đại học VinUniversity, Hà Nội |
| **Tổng điểm** | 100 điểm |

### 🎯 Tư duy cốt lõi cần nắm
1. **Hiểu dữ liệu trước khi xây và chạy mô hình**
2. **Diễn giải kết quả một cách rõ ràng**
3. **Liên hệ insight với bối cảnh kinh doanh thực tế**

---

## 2. CẤU TRÚC DỮ LIỆU

### 2.1 Sơ đồ tổng quan (15 file CSV)

```
📁 Dữ liệu Datathon 2026
│
├── 🔵 MASTER (tham chiếu)
│   ├── products.csv        — Danh mục sản phẩm (id, name, category, segment, size, color, price, cogs)
│   ├── customers.csv       — Khách hàng (id, zip, city, signup_date, gender, age_group, acquisition_channel)
│   ├── promotions.csv      — Chiến dịch khuyến mãi (id, name, type, discount, dates, category, channel)
│   └── geography.csv       — Địa lý (zip, city, region, district)
│
├── 🟡 TRANSACTION (giao dịch)
│   ├── orders.csv          — Đơn hàng (id, date, customer_id, zip, status, payment, device, source)
│   ├── order_items.csv     — Chi tiết đơn (order_id, product_id, qty, unit_price, discount, promo_id)
│   ├── payments.csv        — Thanh toán 1:1 với đơn hàng (method, value, installments)
│   ├── shipments.csv       — Vận chuyển (ship_date, delivery_date, shipping_fee)
│   ├── returns.csv         — Trả hàng (return_id, date, reason, qty, refund_amount)
│   └── reviews.csv         — Đánh giá (review_id, date, rating 1-5, title)
│
├── 🟢 ANALYTICAL (phân tích)
│   ├── sales.csv           — Doanh thu train (Date, Revenue, COGS) [2012–2022]
│   └── sample_submission.csv — Mẫu nộp bài
│
└── 🔴 OPERATIONAL (vận hành)
    ├── inventory.csv       — Tồn kho cuối tháng (stock, units_received, units_sold, flags)
    └── web_traffic.csv     — Lưu lượng web (sessions, visitors, page_views, bounce_rate, source)
```

### 2.2 Quan hệ giữa các bảng

| Quan hệ | Cardinality |
|---------|-------------|
| orders ↔ payments | 1 : 1 |
| orders ↔ shipments | 1 : 0 hoặc 1 |
| orders ↔ returns | 1 : 0 hoặc nhiều |
| orders ↔ reviews | 1 : 0 hoặc nhiều (~20% đơn delivered) |
| order_items ↔ promotions | nhiều : 0 hoặc 1 |
| products ↔ inventory | 1 : nhiều (1 dòng/sản phẩm/tháng) |

### 2.3 Công thức quan trọng trong dữ liệu

```python
# Gross Margin (Tỷ suất lợi nhuận gộp)
gross_margin = (price - cogs) / price

# Discount Amount (Giảm giá)
# Nếu promo_type == 'percentage':
discount = quantity × unit_price × (discount_value / 100)
# Nếu promo_type == 'fixed':
discount = quantity × discount_value

# Revenue thuần (ước tính)
net_revenue = (unit_price × quantity) - discount_amount
```

---

## 3. YÊU CẦU 3 PHẦN THI

### 📝 Phần 1 — Câu hỏi Trắc nghiệm (10 câu, 20 điểm)

Mỗi câu cần tính toán trực tiếp từ dữ liệu. Dưới đây là hướng giải cho từng câu:

| Câu | Nội dung | Dữ liệu cần | Gợi ý kỹ thuật |
|-----|----------|-------------|----------------|
| Q1 | Trung vị inter-order gap của khách hàng mua >1 lần | `orders.csv` | `groupby customer_id`, `sort order_date`, `diff()`, `median()` |
| Q2 | Segment có gross margin TB cao nhất | `products.csv` | `groupby segment`, `mean((price-cogs)/price)` |
| Q3 | Lý do trả hàng nhiều nhất của Streetwear | `returns.csv` + `products.csv` | `merge on product_id`, `filter category='Streetwear'`, `value_counts()` |
| Q4 | Traffic source có bounce_rate TB thấp nhất | `web_traffic.csv` | `groupby traffic_source`, `mean(bounce_rate)`, `idxmin()` |
| Q5 | % dòng order_items có áp dụng promo | `order_items.csv` | `(promo_id.notna().sum() / len) * 100` |
| Q6 | Age group có trung bình đơn hàng/khách cao nhất | `customers.csv` + `orders.csv` | `merge`, `groupby age_group`, `count orders / count customers` |
| Q7 | Region tạo ra doanh thu cao nhất | `geography.csv` + `sales_train.csv` + `orders.csv` | `join` theo zip/region |
| Q8 | Phương thức thanh toán nhiều nhất trong đơn cancelled | `orders.csv` | `filter status='cancelled'`, `value_counts(payment_method)` |
| Q9 | Size có tỷ lệ trả hàng cao nhất | `returns.csv` + `order_items.csv` + `products.csv` | `join`, `returns / order_items per size` |
| Q10 | Kế hoạch trả góp có giá trị TB/đơn cao nhất | `payments.csv` | `groupby installments`, `mean(payment_value)` |

---

### 📊 Phần 2 — Trực quan hoá & Phân tích EDA (60 điểm)

#### 4 Cấp độ phân tích cần đạt

```
Descriptive  →  Diagnostic  →  Predictive  →  Prescriptive
(What?)          (Why?)         (What next?)    (What to do?)
```

#### Gợi ý các góc phân tích có chiều sâu

**📌 Chủ đề 1: Phân tích Doanh thu & Seasonality**
- Descriptive: Doanh thu theo tháng/quý/năm, trend tổng thể
- Diagnostic: Doanh thu tăng/giảm vì KM? vì traffic? vì product mix?
- Predictive: Mùa vụ có lặp lại không? Pattern nào đáng tin cậy?
- Prescriptive: Khi nào nên tung KM? Tháng nào cần đẩy tồn kho?

**📌 Chủ đề 2: Phân tích Khách hàng (Customer Segmentation)**
- Descriptive: Phân bổ theo age_group, gender, city, acquisition_channel
- Diagnostic: Nhóm nào mua nhiều nhất? Kênh nào mang khách chất lượng?
- Predictive: Khách hàng nào có nguy cơ churn (inter-order gap dài)?
- Prescriptive: Nên đầu tư vào kênh nào? Cần chính sách gì giữ chân khách?

**📌 Chủ đề 3: Hiệu quả Khuyến mãi**
- Descriptive: % đơn có KM, giá trị discount TB, phân bổ promo_type
- Diagnostic: KM có thực sự tăng doanh thu không (so sánh có/không KM)?
- Predictive: KM stackable có ảnh hưởng đến return rate không?
- Prescriptive: Loại KM nào ROI tốt nhất? Category nào nên ưu tiên KM?

**📌 Chủ đề 4: Sản phẩm & Tồn kho**
- Descriptive: Sản phẩm bán chạy, category/segment phân bổ
- Diagnostic: Stockout xảy ra ở đâu? Overstock tốn tiền bao nhiêu?
- Predictive: Fill_rate, sell_through_rate dự báo xu hướng tồn kho
- Prescriptive: Reorder policy cho từng category/segment

**📌 Chủ đề 5: Trải nghiệm Khách hàng (Returns & Reviews)**
- Descriptive: Tỷ lệ return theo size/category/reason, phân bổ rating
- Diagnostic: Rating thấp liên quan đến return? Defective tập trung ở đâu?
- Predictive: Sản phẩm nào rủi ro return cao?
- Prescriptive: QC cải thiện ở category nào? Size chart cần update không?

---

### 🤖 Phần 3 — Mô hình Dự báo Doanh thu (20 điểm)

#### Định nghĩa bài toán
- **Input**: `sales.csv` (04/07/2012 – 31/12/2022) + các file hỗ trợ
- **Output**: Dự báo `Revenue` và `COGS` cho từng ngày từ 01/01/2023 – 01/07/2024
- **Metric**: MAE ↓, RMSE ↓, R² ↑

#### Pipeline đề xuất

```python
# Bước 1: Feature Engineering từ dữ liệu thời gian
time_features = [
    'day_of_week', 'day_of_month', 'week_of_year',
    'month', 'quarter', 'year', 'is_weekend',
    'is_holiday_season'  # Tết, 9/9, 11/11, 12/12...
]

# Bước 2: Lag features & Rolling statistics
lag_features = ['lag_7', 'lag_14', 'lag_30', 'lag_365']
rolling_features = ['rolling_mean_7', 'rolling_std_7', 'rolling_mean_30']

# Bước 3: External signals từ các bảng khác
external_features = [
    'monthly_orders',       # từ orders.csv
    'promo_active_flag',    # từ promotions.csv
    'web_sessions',         # từ web_traffic.csv
    'inventory_fill_rate',  # từ inventory.csv
]

# Bước 4: Model
# Option A — Tree-based (khuyến nghị):
#   XGBoost / LightGBM / CatBoost
# Option B — Time Series:
#   Prophet + XGBoost hybrid
# Option C — Ensemble
```

#### Các lưu ý quan trọng
- ⚠️ **Không dùng Revenue/COGS từ test set làm feature** — vi phạm sẽ bị loại toàn bộ phần 3
- ⚠️ **Cross-validation phải theo chiều thời gian** (TimeSeriesSplit, không shuffle)
- ✅ Đặt `random_seed` cho tất cả model
- ✅ Dùng SHAP / feature importance để giải thích model

---

## 4. THANG ĐIỂM CHI TIẾT

### Phần 1 (20đ)
| Kết quả | Điểm |
|---------|------|
| Đúng | 2đ/câu |
| Sai / Không trả lời | 0đ |

### Phần 2 (60đ)

| Tiêu chí | Điểm tối đa | Mức cao nhất |
|----------|-------------|--------------|
| Chất lượng visualizations | 15đ | 13–15đ: Tất cả chart đạt chuẩn, chọn đúng loại |
| Chiều sâu phân tích | 25đ | 21–25đ: Đủ 4 cấp độ D→D→P→P |
| Insight kinh doanh | 15đ | 13–15đ: Đề xuất cụ thể, định lượng, áp dụng được |
| Sáng tạo & storytelling | 5đ | 4–5đ: Góc nhìn độc đáo, kết hợp nhiều nguồn |

> 💡 **Mẹo**: Chiều sâu phân tích chiếm 25/60 điểm — đây là tiêu chí quan trọng nhất!

### Phần 3 (20đ)

| Thành phần | Điểm | Mức cao nhất |
|------------|------|--------------|
| Hiệu suất model (Kaggle) | 12đ | 10–12đ: Top leaderboard |
| Báo cáo kỹ thuật | 8đ | 7–8đ: Pipeline rõ, CV đúng, SHAP đầy đủ |

---

## 5. KIẾN THỨC & KỸ NĂNG CẦN CÓ

### 5.1 Xử lý dữ liệu
```
✅ Đọc và merge nhiều file CSV (pandas merge, join)
✅ Xử lý missing values (fillna, dropna, imputation)
✅ Xử lý datetime (parse dates, extract features, timedelta)
✅ Groupby + aggregation (sum, mean, count, median)
✅ Window functions (rolling, expanding, shift)
✅ Phát hiện outliers (IQR, z-score)
✅ Tạo derived features từ raw data
```

### 5.2 Phân tích & Trực quan hóa
```
✅ Phân phối dữ liệu (histogram, boxplot, violin)
✅ Time series plot (trend, seasonality)
✅ Correlation heatmap
✅ Bar chart, grouped bar, stacked bar
✅ Scatter plot với regression line
✅ Geographic visualization (choropleth nếu có thể)
✅ Viết nhận xét phân tích đủ 4 cấp: D→D→P→P
```

### 5.3 Tư duy Mô hình hóa
```
✅ Hiểu sự khác biệt giữa train/validation/test split
✅ Time series cross-validation (không shuffle dữ liệu!)
✅ Feature engineering cho time series
✅ Evaluation metrics: MAE, RMSE, R²
✅ Overfitting vs underfitting
✅ Hyperparameter tuning (GridSearch / Optuna / manual)
✅ Model explainability: SHAP values, feature importance
```

### 5.4 Kỹ năng Mềm
```
✅ Data storytelling: kể chuyện bằng dữ liệu
✅ Business thinking: kết nối số liệu với quyết định kinh doanh
✅ Viết báo cáo kỹ thuật (LaTeX, NeurIPS template)
✅ Version control với Git/GitHub
```

---

## 6. CÔNG CỤ & THƯ VIỆN

### Setup môi trường
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install xgboost lightgbm catboost
pip install shap prophet
pip install statsmodels jupyter
```

### Bảng tham chiếu nhanh

| Tác vụ | Thư viện | Hàm chính |
|--------|----------|-----------|
| Đọc dữ liệu | `pandas` | `pd.read_csv()`, `pd.merge()` |
| Xử lý datetime | `pandas` | `pd.to_datetime()`, `.dt.month`, `.diff()` |
| Thống kê mô tả | `pandas` | `.describe()`, `.value_counts()`, `.groupby()` |
| Time series | `statsmodels` | `seasonal_decompose()` |
| Visualization cơ bản | `matplotlib`, `seaborn` | `plt.plot()`, `sns.heatmap()` |
| Visualization tương tác | `plotly` | `px.line()`, `px.bar()` |
| ML Model | `sklearn` | `Pipeline`, `TimeSeriesSplit` |
| Gradient Boosting | `xgboost` / `lightgbm` | `XGBRegressor`, `LGBMRegressor` |
| Prophet | `prophet` | `Prophet().fit().predict()` |
| Explainability | `shap` | `shap.TreeExplainer()`, `shap.summary_plot()` |

### Code mẫu — Load & Merge dữ liệu
```python
import pandas as pd
import numpy as np

# Load Master tables
products = pd.read_csv('products.csv')
customers = pd.read_csv('customers.csv')
geography = pd.read_csv('geography.csv')
promotions = pd.read_csv('promotions.csv')

# Load Transaction tables
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv('order_items.csv')
payments = pd.read_csv('payments.csv')
shipments = pd.read_csv('shipments.csv')
returns = pd.read_csv('returns.csv', parse_dates=['return_date'])
reviews = pd.read_csv('reviews.csv')

# Load Operational
inventory = pd.read_csv('inventory.csv', parse_dates=['snapshot_date'])
web_traffic = pd.read_csv('web_traffic.csv', parse_dates=['date'])

# Load Analytical
sales_train = pd.read_csv('sales.csv', parse_dates=['Date'])
sample_sub = pd.read_csv('sample_submission.csv', parse_dates=['Date'])

print("Data loaded successfully!")
print(f"Training period: {sales_train['Date'].min()} → {sales_train['Date'].max()}")
print(f"Test period: {sample_sub['Date'].min()} → {sample_sub['Date'].max()}")
```

### Code mẫu — Feature Engineering cho Forecasting
```python
def create_time_features(df, date_col='Date'):
    df = df.copy()
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    # Mùa mua sắm Việt Nam
    df['is_tet_season'] = df['month'].isin([1, 2]).astype(int)
    df['is_sale_season'] = df['month'].isin([11, 12]).astype(int)  # 11/11, 12/12
    return df

def create_lag_features(df, target_col='Revenue', lags=[7, 14, 30, 365]):
    df = df.sort_values('Date').copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col='Revenue', windows=[7, 14, 30]):
    for w in windows:
        df[f'rolling_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
        df[f'rolling_std_{w}'] = df[target_col].shift(1).rolling(w).std()
    return df
```

### Code mẫu — Time Series Cross Validation
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

tscv = TimeSeriesSplit(n_splits=5)

maes, rmses, r2s = [], [], []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = xgb.XGBRegressor(n_estimators=500, random_state=42)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    maes.append(mean_absolute_error(y_val, preds))
    rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    r2s.append(r2_score(y_val, preds))
    print(f"Fold {fold+1}: MAE={maes[-1]:.2f}, RMSE={rmses[-1]:.2f}, R²={r2s[-1]:.4f}")

print(f"\nAvg MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
print(f"Avg RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
print(f"Avg R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
```

---

## 7. KẾ HOẠCH THỰC THI CHI TIẾT

### 🗓️ Timeline tổng quan

```
Giai đoạn 1 (Ngày 1–2): Setup & EDA ban đầu
Giai đoạn 2 (Ngày 2–3): Phần 1 — Trả lời MCQ
Giai đoạn 3 (Ngày 3–5): Phần 2 — EDA sâu & Visualizations
Giai đoạn 4 (Ngày 4–6): Phần 3 — Modeling & tuning
Giai đoạn 5 (Ngày 6–7): Viết báo cáo & nộp bài
```

---

### 📅 Giai đoạn 1: Setup & EDA ban đầu (Ngày 1–2)

**Mục tiêu**: Hiểu toàn bộ dữ liệu trước khi làm bài

#### Checklist
- [ ] Setup môi trường Python (Jupyter/Colab/Kaggle Notebook)
- [ ] Download toàn bộ dữ liệu từ Kaggle
- [ ] Setup GitHub repo (cấu trúc thư mục rõ ràng)
- [ ] Load tất cả 15 file, kiểm tra `shape`, `dtypes`, `head()`
- [ ] Kiểm tra missing values từng bảng: `df.isnull().sum()`
- [ ] Kiểm tra duplicates: `df.duplicated().sum()`
- [ ] Xác nhận các FK constraint hợp lệ (join không bị mất dữ liệu)
- [ ] Thống kê phân bổ các cột quan trọng
- [ ] Vẽ time series Revenue tổng để nắm trend

```
📁 Cấu trúc GitHub đề xuất:
datathon-2026/
├── README.md
├── data/                  (không commit file CSV lớn)
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda_overview.ipynb
│   ├── 02_mcq_answers.ipynb
│   ├── 03_eda_deep.ipynb
│   └── 04_forecasting.ipynb
├── src/
│   ├── feature_engineering.py
│   └── utils.py
├── reports/
│   └── report.pdf
└── submission/
    └── submission.csv
```

---

### 📅 Giai đoạn 2: Phần 1 — MCQ (Ngày 2)

**Mục tiêu**: Tính toán chính xác 10 câu trắc nghiệm

Với mỗi câu:
1. Xác định file dữ liệu cần dùng
2. Viết code, tính toán kết quả
3. Đối chiếu với 4 đáp án
4. Ghi lại kết quả + đoạn code vào notebook riêng

> 💡 Dùng 1 notebook riêng `02_mcq_answers.ipynb` cho phần này, code rõ ràng từng câu.

---

### 📅 Giai đoạn 3: Phần 2 — EDA & Visualizations (Ngày 3–5)

**Mục tiêu**: Tạo 4–6 câu chuyện phân tích đạt cấp Prescriptive

#### Workflow cho mỗi câu chuyện phân tích

```
1. Đặt câu hỏi kinh doanh cụ thể
       ↓
2. Xác định bảng dữ liệu cần join
       ↓
3. Tính toán số liệu tổng hợp
       ↓
4. Chọn loại visualization phù hợp
       ↓
5. Vẽ chart với đầy đủ: title, axis labels, annotations
       ↓
6. Viết 4 cấp phân tích (D → D → P → P)
       ↓
7. Đề xuất hành động kinh doanh cụ thể
```

#### Yêu cầu mỗi visualization
- ✅ Title rõ ràng
- ✅ Nhãn trục X, Y đầy đủ
- ✅ Legend nếu có nhiều series
- ✅ Annotation số liệu quan trọng
- ✅ Source note (dữ liệu từ file nào)

---

### 📅 Giai đoạn 4: Phần 3 — Forecasting (Ngày 4–6)

**Mục tiêu**: Xây dựng model cho R² cao, MAE/RMSE thấp

#### Pipeline chi tiết

```
Bước 1: Chuẩn bị dữ liệu
├── Load sales_train.csv
├── EDA time series: trend, seasonality, stationarity
└── Kiểm tra outliers, xử lý nếu cần

Bước 2: Feature Engineering
├── Time features (year, month, quarter, week, day, weekend)
├── Lag features (lag_7, lag_14, lag_30, lag_365)
├── Rolling statistics (mean, std, min, max với window 7, 14, 30)
├── Seasonal features (Tết, 9/9, 11/11, 12/12)
└── External features (join từ web_traffic, inventory, promotions)

Bước 3: Model Selection & Training
├── Baseline: Linear Regression, SARIMA
├── Tree-based: XGBoost, LightGBM
├── Ensemble: XGB + Prophet hybrid
└── Cross-validation: TimeSeriesSplit(n_splits=5)

Bước 4: Evaluation
├── Tính MAE, RMSE, R² trên từng fold
└── So sánh models, chọn model tốt nhất

Bước 5: Final Prediction
├── Train lại trên toàn bộ train set
├── Dự báo cho sample_submission.csv dates
└── Tạo submission.csv theo đúng format

Bước 6: Explainability
├── SHAP values cho top features
├── Feature importance plot
└── Viết diễn giải bằng ngôn ngữ kinh doanh
```

#### Format file nộp
```csv
Date,Revenue,COGS
2023-01-01,26607.2,2585.15
2023-01-02,1007.89,163.0
...
```
> ⚠️ Giữ đúng thứ tự dòng như `sample_submission.csv`, không xáo trộn!

---

### 📅 Giai đoạn 5: Báo cáo & Nộp bài (Ngày 6–7)

**Mục tiêu**: Hoàn thiện báo cáo LaTeX, nộp đúng hạn

#### Cấu trúc báo cáo (max 4 trang, NeurIPS template)

```
1. Abstract (1 đoạn tóm tắt)

2. Phần 2: Data Exploration & Analysis
   - 3–4 visualizations đại diện
   - Phân tích 4 cấp cho mỗi viz
   - Business recommendations

3. Phần 3: Forecasting Methodology
   - Data preprocessing
   - Feature engineering
   - Model architecture & validation
   - Results (MAE, RMSE, R²)
   - Model explainability (SHAP)

4. Conclusion & Future Work

References (không tính số trang)
Appendix (không tính số trang)
```

---

## 8. HƯỚNG DẪN NỘP BÀI

### Checklist nộp bài đầy đủ

#### 1. Kaggle Submission
- [ ] Upload `submission.csv` lên Kaggle
- [ ] Kiểm tra số dòng khớp với `sample_submission.csv`
- [ ] Chụp màn hình leaderboard score

#### 2. GitHub Repository
- [ ] Repo public (hoặc cấp quyền cho BTC)
- [ ] `README.md` có: mô tả cấu trúc thư mục, hướng dẫn run
- [ ] Toàn bộ notebooks (clean, có output)
- [ ] `submission.csv` trong thư mục `/submission`
- [ ] `random_seed` được đặt trong tất cả nơi cần thiết

#### 3. Báo cáo PDF
- [ ] Dùng NeurIPS 2025 LaTeX template
- [ ] Tối đa 4 trang (không tính references và appendix)
- [ ] Có link GitHub trong báo cáo
- [ ] Phần 2: Visualizations + 4-level analysis
- [ ] Phần 3: Pipeline + SHAP + Results

#### 4. Form nộp bài chính thức
- [ ] Đáp án 10 câu trắc nghiệm
- [ ] Upload báo cáo PDF
- [ ] Link GitHub repository
- [ ] Link submission Kaggle
- [ ] Ảnh thẻ sinh viên tất cả thành viên
- [ ] ☑️ Xác nhận có ít nhất 1 thành viên tham gia trực tiếp VCK ngày **23/05/2026** tại VinUniversity, Hà Nội

> 🚨 **Quan trọng**: Không xác nhận tham gia VCK = **không đủ điều kiện vào vòng tiếp theo!**

---

## 9. CHECKLIST CUỐI CÙNG

### Trước khi nộp bài

#### Phần 1 — MCQ
- [ ] Đã tính toán và kiểm tra lại tất cả 10 câu
- [ ] Ghi rõ đáp án trong form

#### Phần 2 — EDA
- [ ] Ít nhất 4–5 visualizations chất lượng cao
- [ ] Mỗi viz có title, axis labels, annotations
- [ ] Phân tích đủ 4 cấp: Descriptive → Diagnostic → Predictive → Prescriptive
- [ ] Có ít nhất 1 phân tích kết hợp nhiều bảng dữ liệu
- [ ] Đề xuất hành động kinh doanh cụ thể, định lượng được

#### Phần 3 — Forecasting
- [ ] Không dùng Revenue/COGS test làm feature (**critical!**)
- [ ] Cross-validation theo thời gian (TimeSeriesSplit)
- [ ] Random seed được đặt
- [ ] SHAP values / feature importance có trong báo cáo
- [ ] `submission.csv` đúng format, đúng thứ tự dòng
- [ ] Code có thể chạy lại và cho kết quả tương tự

#### Báo cáo
- [ ] Không quá 4 trang (không tính references)
- [ ] Template NeurIPS 2025
- [ ] Link GitHub trong báo cáo

---

## 💡 TIP & TRICKS CUỐI

### Điều nên làm ✅
- Bắt đầu với EDA kỹ trước khi code model
- Commit code lên GitHub thường xuyên
- Đặt tên biến, function rõ ràng (code phải readable)
- Comment giải thích logic trong code
- Kiểm tra submission format nhiều lần trước khi nộp
- Dành thời gian cho phần báo cáo — ban giám khảo đọc báo cáo, không chạy code!

### Điều cần tránh ❌
- Data leakage: dùng future data trong feature
- Shuffle dữ liệu khi cross-validate time series
- Chart không có title/label
- Phân tích chỉ dừng ở Descriptive
- Nộp submission sai format / sai thứ tự dòng
- Để repo private và không cấp quyền cho BTC

### Phân bổ thời gian gợi ý
| Phần | Thời gian |
|------|-----------|
| Setup + EDA ban đầu | 15% |
| MCQ (Phần 1) | 10% |
| Deep EDA + Viz (Phần 2) | 40% |
| Forecasting (Phần 3) | 25% |
| Báo cáo + Nộp bài | 10% |

---

*Chúc các đội thi thành công! 🚀*
*"Breaking Business Boundaries — One Dataset at a Time"*
