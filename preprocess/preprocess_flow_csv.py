# preprocess_flow_csv.py

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np # numpy import 추가

# --- 경로 설정 ---
RAW_DATA_DIR = '../data/raw'  # 원본 데이터 위치
PROCESSED_DATA_DIR = '../data/processed' # 전처리된 데이터 저장 위치
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- 함수 정의 ---
def preprocess_and_load(path: str) -> pd.DataFrame:
    """
    단일 파일을 읽고 기본적인 전처리를 수행한 후,
    데이터프레임을 반환합니다. (스케일링 및 저장은 제외)
    """
    print(f"📂 Loading and cleaning {os.path.basename(path)}...")
    df = pd.read_csv(path, low_memory=False)

    # 1. 라벨 컬럼 제거
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])

    # 2. 수치형 컬럼만 선택
    numeric_df = df.select_dtypes(include=[np.number]) # np.number로 더 안정적으로 선택

    # 3. 결측치(NaN) 및 무한값(inf)이 포함된 행 제거
    shape_before = numeric_df.shape
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.dropna(inplace=True)
    print(f"    - Shape changed from {shape_before} to {numeric_df.shape} after dropping NaN/Inf.")

    return numeric_df

# === 메인 실행 블록 ===
if __name__ == '__main__':
    
    # 1. 전처리된 데이터프레임을 담을 빈 리스트 생성
    benign_df_list = []

    # 2. BenignTraffic 파일들을 순회하며 전처리 후 리스트에 추가
    print("--- Starting to process Benign traffic files ---")
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.startswith('BenignTraffic') and filename.endswith('pcap_Flow.csv'):
            file_path = os.path.join(RAW_DATA_DIR, filename)
            processed_df = preprocess_and_load(path=file_path)
            benign_df_list.append(processed_df)

    # 3. 리스트에 수집된 모든 데이터프레임을 하나로 합치기
    if benign_df_list:
        print("\n--- Concatenating all processed files ---")
        # 공통된 컬럼을 기준으로 합치고, 없는 컬럼은 NaN으로 처리 후 제거
        final_benign_df = pd.concat(benign_df_list, ignore_index=True, join='inner')

        # 4. 전체 데이터에 대해 스케일러 적용
        print("--- Scaling the final combined dataframe ---")
        scaler = MinMaxScaler()
        # 데이터프레임의 모든 값을 float으로 변환하여 스케일러 오류 방지
        final_benign_df = final_benign_df.astype(float)
        
        scaled_values = scaler.fit_transform(final_benign_df)
        scaled_df = pd.DataFrame(scaled_values, columns=final_benign_df.columns)

        # 5. 최종 결과물 및 스케일러, 컬럼 정보 저장
        print("--- Saving final artifacts ---")
        
        # 스케일러 저장
        scaler_path = os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved to {scaler_path}")

        # 컬럼 정보 저장
        columns_path = os.path.join(PROCESSED_DATA_DIR, 'columns.pkl')
        joblib.dump(scaled_df.columns.tolist(), columns_path)
        print(f"✅ Columns list saved to {columns_path}")

        # 최종 전처리된 데이터 저장
        output_path = os.path.join(PROCESSED_DATA_DIR, 'benign_processed.csv')
        scaled_df.to_csv(output_path, index=False)
        print(f"✅ Final processed data saved to {output_path}, with shape {scaled_df.shape}")

    else:
        print("No 'BenignTraffic' files found to process.")