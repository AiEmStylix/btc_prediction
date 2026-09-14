"""
Sanity check: chạy Logistic Regression + Random Forest trên cùng dữ liệu
với BiLSTM để xem vấn đề nằm ở model hay ở data.

Nếu ML cổ điển cũng ko thắng baseline => tín hiệu trong data quá yếu.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

SEED = 42
np.random.seed(SEED)
WINDOW = 14

# Features
FEAT_FULL = ['open', 'high', 'low', 'close', 'volume',
             'sentiment_mean', 'tweet_count', 'pos_ratio', 'neg_ratio',
             'has_tweet_flag', 'days_since_tweet']
FEAT_BASELINE = ['open', 'high', 'low', 'close', 'volume']


def create_flat_sequences(X, y, window):
    """Flatten sliding window thành vector 1D cho model cổ điển
    (Logistic Regression, RF ko nhận input 3D như LSTM)."""
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window].flatten())
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)


def run_classical(data, feature_cols, label):
    print(f"\n{'='*60}")
    print(f"  [Sanity check] {label}")
    print(f"{'='*60}")

    split_idx = int(len(data) * 0.85)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(train[feature_cols])
    X_test_sc = scaler.transform(test[feature_cols])

    y_train = train['direction'].values
    y_test = test['direction'].values

    X_train, y_tr = create_flat_sequences(X_train_sc, y_train, WINDOW)
    X_test, y_te = create_flat_sequences(X_test_sc, y_test, WINDOW)

    # Baseline: luôn đoán lớp đa số
    majority = int(round(y_tr.mean()))
    naive_pred = np.full_like(y_te, majority)
    naive_acc = accuracy_score(y_te, naive_pred)
    best_naive = max((y_te == 0).mean(), (y_te == 1).mean())

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=SEED),
    }

    for name, clf in models.items():
        clf.fit(X_train, y_tr)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = float('nan')

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        print(f"  (naive baseline: {naive_acc:.3f} | best naive: {best_naive:.3f})")

        if acc > best_naive and auc > 0.55:
            print("=> Co tin hieu hoc duoc")
        else:
            print("=> Khong thang baseline / AUC qua thap")


def main():
    data = pd.read_csv("merged_dataset.csv")
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)

    # Tạo label: 1 nếu giá tăng so với ngày trước
    data['direction'] = (data['close'].diff() > 0).astype(int)
    data = data.iloc[1:].reset_index(drop=True)

    run_classical(data, FEAT_FULL, "OHLCV + Sentiment")
    run_classical(data, FEAT_BASELINE, "OHLCV thuan (baseline)")

    print(f"\n{'='*60}")
    print("Ý NGHĨA KẾT QUẢ:")
    print("- ML co dien CUNG ko thang baseline => van de o DATA, ko phai BiLSTM")
    print("- ML co dien THANG nhung BiLSTM ko => BiLSTM qua phuc tap cho data nay")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()