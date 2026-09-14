"""
Entry point chính cho project Bitcoin Prediction.
Chạy từng bước pipeline qua command line.

Usage:
    python main.py clean      - Làm sạch dataset tweet
    python main.py sanity     - Chạy sanity check với ML cổ điển
"""

import sys


def print_usage():
    print("Bitcoin Price Direction Prediction")
    print("=" * 40)
    print("Usage:")
    print("  python main.py clean    Làm sạch raw tweet data")
    print("  python main.py sanity   Sanity check bằng Logistic Regression + Random Forest")
    print()
    print("Lưu ý: Sentiment analysis và training BiLSTM chạy trên Colab (xem notebooks).")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    cmd = sys.argv[1].lower()

    if cmd == "clean":
        from clean_data import clean_twitter_dataset, download_dataset, HF_DATASET_URL
        import os
        input_csv = sys.argv[2] if len(sys.argv) > 2 else 'dataset_cleaned.csv'
        output_csv = sys.argv[3] if len(sys.argv) > 3 else 'dataset_cleaned_local.csv'

        # Tự tải từ HF nếu chưa có
        if not os.path.exists(input_csv):
            download_dataset(HF_DATASET_URL, input_csv)

        clean_twitter_dataset(input_csv, output_csv)

    elif cmd == "sanity":
        from sanity_check_classic_ml import main as run_sanity
        run_sanity()

    else:
        print(f"Lệnh không hợp lệ: {cmd}")
        print_usage()


if __name__ == "__main__":
    main()
