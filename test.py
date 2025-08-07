# test.py 또는 detect.py

import torch
import pandas as pd
from sklearn.metrics import classification_report
from models.autoencoder import AutoEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt


columns = joblib.load("data/columns.pkl")  # 학습 시 저장해둔 컬럼 순서

# 모델 불러오기
input_dim = 79  # 전처리한 데이터 컬럼 수
model = AutoEncoder(input_dim)
model.load_state_dict(torch.load('models/autoencoder.pth'))
model.eval()

# 테스트 데이터 불러오기
# DDoS : data/processed/DDoS-PreProcessed.csv
# ARP : data/processed/ARP-PreProcessed.csv
# benign : data/processed/benign_processed.csv
test_df = pd.read_csv("data/processed/benign_processed.csv")  # 또는 공격 데이터
test_df = test_df[columns]  # 컬럼 순서 맞추기
test_data = torch.tensor(test_df.values, dtype=torch.float32)

print("테스트 데이터 shape:", test_data.shape)
print("모델 입력 차원:", model.encoder[0].in_features)  # 예시, 실제 코드에 맞게 수정

# 모델에 통과시켜 재구성
with torch.no_grad():
    recon = model(test_data)
    loss = ((recon - test_data) ** 2).mean(dim=1)  # MSE per sample

# 임계값 기준
# threshold = loss.mean() + 3 * loss.std() 
# threshold = loss.mean() + 2 * loss.std() # 임계값 줄이기
threshold = np.percentile(loss.numpy(), 99.9)
anomalies = loss > threshold

print(f"Detected {anomalies.sum().item()} anomalies out of {len(loss)} samples.")
plt.hist(loss.numpy(), bins=1000, log=True)
plt.axvline(threshold, color='r', linestyle='--', label='threshold')
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE Loss")
plt.ylabel("Count")
plt.legend()
plt.show()

sorted_loss = np.sort(loss.numpy())
print("Top 10 highest losses:", sorted_loss[-10:])