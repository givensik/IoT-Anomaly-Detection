# test.py 또는 detect.py

import torch
import pandas as pd
from sklearn.metrics import classification_report
from models.autoencoder import AutoEncoder
import joblib


columns = joblib.load("data/columns.pkl")  # 학습 시 저장해둔 컬럼 순서

# 모델 불러오기
input_dim = 79  # 전처리한 데이터 컬럼 수
model = AutoEncoder(input_dim)
model.load_state_dict(torch.load('models/autoencoder.pth'))
model.eval()

# 테스트 데이터 불러오기
test_df = pd.read_csv("data/processed/DDoS-PreProcessed.csv")  # 또는 공격 데이터
test_df = test_df[columns]  # 컬럼 순서 맞추기
test_data = torch.tensor(test_df.values, dtype=torch.float32)

print("테스트 데이터 shape:", test_data.shape)
print("모델 입력 차원:", model.encoder[0].in_features)  # 예시, 실제 코드에 맞게 수정

# 모델에 통과시켜 재구성
with torch.no_grad():
    recon = model(test_data)
    loss = ((recon - test_data) ** 2).mean(dim=1)  # MSE per sample

# 임계값 기준
threshold = loss.mean() + 3 * loss.std()
anomalies = loss > threshold

print(f"Detected {anomalies.sum().item()} anomalies out of {len(loss)} samples.")
