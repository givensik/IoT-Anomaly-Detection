# preprocess_flow_csv.py

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib


RAW_DATA_DIR = '../data/raw'
PROCESSED_DATA_DIR = '../data/processed'
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def is_float_column(series):
    try:
        series.astype(float)
        return True
    except:
        return False

def preprocess_file(path, output_name):
    print(f"📂 Processing {path}...")
    df = pd.read_csv(path)

    # 1. 라벨 컬럼 제거
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])

    # 2. 수치형 컬럼만 선택
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # 💡 디버깅용 - 수치형 데이터 수 확인
    print(f"👉 Numeric columns shape before cleaning: {numeric_df.shape}")

    # 3. 결측치 제거
    numeric_df = numeric_df.dropna()

    # 4. 무한값 제거
    numeric_df = numeric_df[~numeric_df.isin([float('inf'), float('-inf')]).any(axis=1)]

    # 💡 디버깅용 - inf 제거 후 shape
    print(f"✅ Shape after dropping NaN/Inf: {numeric_df.shape}")

    # 💡 혹시 inf가 제거 안 되는 경우 출력해보기
    if numeric_df.isin([float('inf'), float('-inf')]).any().any():
        print("🚨 Still contains infinite values!")

    # 💡 혹시 너무 큰 값 있는지 확인
    if (numeric_df.abs() > 1e100).any().any():
        print("🚨 Contains extremely large values!")

    # 5. 정규화
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled, columns=numeric_df.columns)
    joblib.dump(scaler, "../data/scaler.pkl")  # scaler 저장

    # 6. 저장
    output_path = os.path.join(PROCESSED_DATA_DIR, output_name)
    scaled_df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}, shape = {scaled_df.shape}")

# === 메인 ===
if __name__ == '__main__':
    # 학습 데이터 전처리리
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.startswith('BenignTraffic') and filename.endswith('pcap_Flow.csv'):
            preprocess_file(
                path=os.path.join(RAW_DATA_DIR, filename),
                output_name='benign_processed.csv'  # 하나로 합칠거면 이렇게
            )
