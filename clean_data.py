import pandas as pd
import re
import numpy as np

def clean_twitter_dataset(input_file, output_file):
    print("⏳ Currently reading file...")
    df = pd.read_csv(input_file, on_bad_lines='skip', engine='python', encoding='utf-8')
    original_len = len(df)
    print(f"Total tweet before: {original_len:,}")

    # Delete null, duplicate
    df = df.dropna(subset=['text', 'date'])
    df['user_followers'] = df['user_followers'].fillna(0)
    df = df.drop_duplicates(subset=['user_name', 'text']) # Tránh 1 user spam 1 câu nhiều lần

    # Remove retweet
    df['is_retweet'] = df['is_retweet'].astype(str).str.lower()
    df = df[~df['is_retweet'].isin(['true', '1', 't'])]

    # Remove bot, spam account
    def is_bot(row):
        if str(row.get('user_verified', 'false')).lower() in ['true', '1', 't']:
            return False
            
        followers = pd.to_numeric(row.get('user_followers', 0), errors='coerce') or 0
        friends = pd.to_numeric(row.get('user_friends', 0), errors='coerce') or 0

        if followers < 30:
            return True
        if followers < 100 and friends > (followers * 10):
            return True
        return False

    print("SCANNED FOR BOT/CLONE ACCOUNT")
    df['is_bot'] = df.apply(is_bot, axis=1)
    df = df[df['is_bot'] == False].drop(columns=['is_bot'])

    def is_spam(row):
        text = str(row['text'])
        
        if len(re.findall(r'@\w+', text)) > 3:
            return True
            
        # Spam link 
        urls = re.findall(r'http[s]?://\S+', text)
        text_without_url = re.sub(r'http[s]?://\S+', '', text).strip()
        if len(urls) > 0 and len(text_without_url) < 15:
            return True
            
        # Too Many Hashtag 
        hashtags_data = str(row.get('hashtags', ''))
        if hashtags_data not in ['nan', 'None', '']:
            if len(hashtags_data.split(',')) > 5:
                return True
        elif len(re.findall(r'#\w+', text)) > 5: # Fallback for hashtag count
            return True
            
        return False

    print("Scanning for spam/shill...")
    df['is_spam'] = df.apply(is_spam, axis=1)
    df = df[df['is_spam'] == False].drop(columns=['is_spam'])

    # NORMALIZE FOR FINBERT
    def clean_text_for_model(text):
        text = str(text)
        text = re.sub(r'http[s]?://\S+', '', text) 
        text = re.sub(r'@\w+', '', text)           
        text = re.sub(r'\s+', ' ', text).strip()   
        return text

    df['clean_text'] = df['text'].apply(clean_text_for_model)
    
    # Xóa những dòng mà sau khi xóa URL/Tag xong thì chẳng còn chữ nào (độ dài < 5 ký tự)
    df = df[df['clean_text'].str.len() >= 5]

    final_len = len(df)
    print("\nDone !")
    print(f" After clean: {final_len:,}")
    print(f"Removed: {original_len - final_len:,} noise ({((original_len - final_len)/original_len)*100:.2f}%)")
    
    df.to_csv(output_file, index=False)
    print(f"💾 Đã lưu dataset sạch tại: {output_file}")

if __name__ == "__main__":
    INPUT_CSV = 'bitcoin_tweets_latest.csv' 
    OUTPUT_CSV = 'dataset_cleaned.csv'
    
    clean_twitter_dataset(INPUT_CSV, OUTPUT_CSV)
