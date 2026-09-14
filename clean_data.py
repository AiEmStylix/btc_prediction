"""
Script làm sạch dataset tweet Bitcoin trước khi đưa vào phân tích sentiment.
Bao gồm: loại bỏ retweet, lọc bot/spam, chuẩn hóa text cho FinBERT.

Nếu file CSV chưa có sẵn ở local, script sẽ tự tải từ Hugging Face.
"""

import pandas as pd
import re
import numpy as np
import os
from urllib.request import urlretrieve

# Dataset trên Hugging Face
HF_DATASET_URL = "https://huggingface.co/datasets/AiEmStylix/btc_tweets/resolve/main/dataset_cleaned.csv"


def download_dataset(url, dest):
    """Tải dataset từ Hugging Face nếu chưa có ở local."""
    if os.path.exists(dest):
        print(f"File {dest} đã tồn tại, bỏ qua download.")
        return dest

    print(f"Đang tải dataset từ Hugging Face...")
    print(f"  URL: {url}")
    urlretrieve(url, dest)
    print(f"Đã tải xong: {dest}")
    return dest


def clean_twitter_dataset(input_file, output_file):
    print("Đang đọc file...")
    df = pd.read_csv(input_file, on_bad_lines='skip', engine='python', encoding='utf-8')
    original_len = len(df)
    print(f"Tổng số tweet ban đầu: {original_len:,}")

    # Xóa null + duplicate
    df = df.dropna(subset=['text', 'date'])
    df['user_followers'] = df['user_followers'].fillna(0)
    df = df.drop_duplicates(subset=['user_name', 'text'])  # tránh 1 user spam 1 câu nhiều lần

    # Bỏ retweet
    df['is_retweet'] = df['is_retweet'].astype(str).str.lower()
    df = df[~df['is_retweet'].isin(['true', '1', 't'])]

    # --- Lọc bot ---
    def is_bot(row):
        # Account verified thì bỏ qua
        if str(row.get('user_verified', 'false')).lower() in ['true', '1', 't']:
            return False

        followers = pd.to_numeric(row.get('user_followers', 0), errors='coerce') or 0
        friends = pd.to_numeric(row.get('user_friends', 0), errors='coerce') or 0

        if followers < 30:
            return True
        # follow nhiều nhưng followers ít => khả năng cao là bot
        if followers < 100 and friends > (followers * 10):
            return True
        return False

    print("Lọc bot/clone account...")
    df['is_bot'] = df.apply(is_bot, axis=1)
    df = df[df['is_bot'] == False].drop(columns=['is_bot'])

    # --- Lọc spam ---
    def is_spam(row):
        text = str(row['text'])

        # tag quá nhiều người
        if len(re.findall(r'@\w+', text)) > 3:
            return True

        # tweet chỉ toàn link, gần như ko có nội dung
        urls = re.findall(r'http[s]?://\S+', text)
        text_no_url = re.sub(r'http[s]?://\S+', '', text).strip()
        if len(urls) > 0 and len(text_no_url) < 15:
            return True

        # quá nhiều hashtag
        hashtags_data = str(row.get('hashtags', ''))
        if hashtags_data not in ['nan', 'None', '']:
            if len(hashtags_data.split(',')) > 5:
                return True
        elif len(re.findall(r'#\w+', text)) > 5:
            return True

        return False

    print("Lọc spam/shill...")
    df['is_spam'] = df.apply(is_spam, axis=1)
    df = df[df['is_spam'] == False].drop(columns=['is_spam'])

    # Chuẩn hóa text cho FinBERT (xóa URL, mention, khoảng trắng thừa)
    def clean_text(text):
        text = str(text)
        text = re.sub(r'http[s]?://\S+', '', text)  # bỏ URL
        text = re.sub(r'@\w+', '', text)              # bỏ @mention
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_text'] = df['text'].apply(clean_text)

    # tweet nào sau khi xóa URL/tag mà còn < 5 ký tự thì bỏ luôn
    df = df[df['clean_text'].str.len() >= 5]

    final_len = len(df)
    removed = original_len - final_len
    print(f"\nXong! Còn lại: {final_len:,} / {original_len:,}")
    print(f"Đã loại bỏ {removed:,} dòng nhiễu ({removed/original_len*100:.1f}%)")

    df.to_csv(output_file, index=False)
    print(f"Đã lưu: {output_file}")


if __name__ == "__main__":
    INPUT_CSV = 'dataset_cleaned.csv'
    OUTPUT_CSV = 'dataset_cleaned_local.csv'

    # Tự tải từ HF nếu chưa có file ở local
    if not os.path.exists(INPUT_CSV):
        download_dataset(HF_DATASET_URL, INPUT_CSV)

    clean_twitter_dataset(INPUT_CSV, OUTPUT_CSV)

