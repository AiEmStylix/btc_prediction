# Bitcoin Price Direction Prediction

Dự đoán **xu hướng giá Bitcoin** (tăng/giảm) bằng mô hình BiLSTM, kết hợp dữ liệu kỹ thuật (OHLCV) với sentiment analysis từ Twitter sử dụng FinBERT.

> Đồ án nghiên cứu — kết quả cho thấy với ~362 ngày dữ liệu, tín hiệu dự đoán hướng giá từ cả OHLCV lẫn sentiment đều quá yếu để model học được hiệu quả.

## Tổng quan

| Thành phần | Mô tả |
|---|---|
| **Dữ liệu giá** | OHLCV Bitcoin (open, high, low, close, volume) — 362 ngày |
| **Dữ liệu tweet** | ~2.3 triệu tweet Bitcoin từ [Hugging Face](https://huggingface.co/datasets/AiEmStylix/btc_tweets) |
| **Sentiment model** | [FinBERT](https://huggingface.co/ProsusAI/finbert) (pretrained trên dữ liệu tài chính) |
| **Prediction model** | BiLSTM (Bidirectional LSTM) — phân loại binary (tăng/giảm) |
| **Sanity check** | Logistic Regression + Random Forest trên cùng dữ liệu |

## Dataset

Dữ liệu được host trên Hugging Face: [AiEmStylix/btc_tweets](https://huggingface.co/datasets/AiEmStylix/btc_tweets)

| File | Mô tả | Link |
|---|---|---|
| `dataset_cleaned.csv` | Tweet đã qua bước làm sạch (output của `clean_data.py`) | [Download](https://huggingface.co/datasets/AiEmStylix/btc_tweets/resolve/main/dataset_cleaned.csv) |
| `merged_dataset.csv` | Dữ liệu đã merge OHLCV + sentiment (dùng cho training) | [Download](https://huggingface.co/datasets/AiEmStylix/btc_tweets/resolve/main/merged_dataset.csv) |

> **Lưu ý:** Script `clean_data.py` sẽ tự tải `dataset_cleaned.csv` từ Hugging Face nếu file chưa có ở local.

## Pipeline

```
1. Thu thập dữ liệu
   ├── Giá BTC (OHLCV) từ API
   └── Tweet Bitcoin từ Kaggle/Hugging Face

2. Tiền xử lý (clean_data.py)
   ├── Loại bỏ retweet, bot, spam
   └── Chuẩn hóa text cho FinBERT

3. Sentiment Analysis (sentiment.ipynb — chạy trên Colab)
   ├── Chạy FinBERT trên toàn bộ tweet
   └── Tính sentiment_score = P(positive) - P(negative)

4. Gộp dữ liệu (merge)
   └── Ghép sentiment hàng ngày với OHLCV → merged_dataset.csv

5. Huấn luyện (training.ipynb — chạy trên Colab)
   ├── BiLSTM với sliding window (14 ngày)
   ├── So sánh: OHLCV + Sentiment vs. OHLCV thuần
   └── Đánh giá: Accuracy, AUC, F1, Confusion Matrix

6. Sanity check (sanity_check_classic_ml.py)
   └── Kiểm chứng bằng ML cổ điển (LR + RF)
```

## Cấu trúc project

```
btc_prediction/
├── main.py                       # Entry point (CLI)
├── clean_data.py                 # Làm sạch tweet data
├── sanity_check_classic_ml.py    # Sanity check với ML cổ điển
├── sentiment.ipynb               # Phân tích sentiment bằng FinBERT (Colab)
├── training.ipynb                # Huấn luyện BiLSTM (Colab)
├── pyproject.toml                # Cấu hình project (uv)
└── README.md
```

## Cài đặt

Project sử dụng [uv](https://docs.astral.sh/uv/) để quản lý dependencies.

```bash
# Clone repo
git clone https://github.com/AiEmStylix/btc_prediction.git
cd btc_prediction

# Cài dependencies
uv sync
```

## Cách chạy

### Làm sạch dataset tweet

```bash
# Tự tải dataset từ Hugging Face nếu chưa có file ở local
uv run python main.py clean

# Hoặc chỉ định file thủ công:
uv run python main.py clean raw_tweets.csv output_clean.csv
```

### Sentiment Analysis & Training

Hai bước này yêu cầu GPU nên chạy trên **Google Colab**:

1. Mở [`sentiment.ipynb`](sentiment.ipynb) trên Colab → chạy hết để tạo file sentiment
2. Merge sentiment với OHLCV (trong notebook)
3. Mở [`training.ipynb`](training.ipynb) trên Colab → huấn luyện BiLSTM

### Sanity check

```bash
uv run python main.py sanity
```

So sánh Logistic Regression + Random Forest với BiLSTM trên cùng dữ liệu để kiểm chứng kết quả.

## Kết quả

### BiLSTM (training.ipynb)

| Metric | OHLCV + Sentiment | OHLCV thuần |
|---|---|---|
| Accuracy | 0.659 | 0.561 |
| AUC | 0.532 | 0.493 |
| F1 | 0.417 | 0.550 |
| Naive baseline | 0.415 | 0.415 |

**Nhận xét:**
- Model OHLCV + Sentiment có accuracy cao hơn baseline nhưng AUC chỉ ~0.53 (gần random)
- Độ lệch chuẩn xác suất dự đoán gần 0 → **model collapse** — model xuất ra gần như cùng 1 giá trị cho mọi mẫu
- Sentiment có vẻ giúp một chút nhưng không đáng kể

### Sanity check — ML cổ điển

Logistic Regression và Random Forest cũng không thắng nổi naive baseline trên cùng dữ liệu → **vấn đề nằm ở data, không phải model**.

## Kết luận

1. Với chỉ ~362 ngày dữ liệu, tín hiệu dự đoán hướng giá BTC quá yếu/nhiễu
2. Cả BiLSTM lẫn ML cổ điển đều không thắng được naive baseline một cách thuyết phục
3. Sentiment từ Twitter có correlation nhẹ nhưng không đủ mạnh để cải thiện đáng kể

## Tech stack

- **Python 3.14** + uv
- **FinBERT** — sentiment analysis
- **TensorFlow/Keras** — BiLSTM
- **scikit-learn** — Logistic Regression, Random Forest, metrics
- **pandas, numpy** — xử lý dữ liệu
- **Google Colab** — GPU cho sentiment analysis và training

## License

Project phục vụ mục đích học tập và nghiên cứu.
