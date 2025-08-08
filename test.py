# test.py 또는 detect.py

import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models.autoencoder import AutoEncoder
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
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

# JYP: 아래 내용부터는 새로 추가 (정상 데이터와 공격 데이터를 함께 로드해 탐지율을 테스트하기 위함)
# 더 밑쪽의 ====== 까지가 내가 추가한 내용

benign_df = pd.read_csv("data/processed/benign_processed.csv")  # 정상 데이터
attack_df = pd.read_csv("data/processed/DDoS-PreProcessed.csv")  # 공격 데이터

benign_df['is_anomaly'] = 0  # 정상 데이터 라벨
attack_df['is_anomaly'] = 1  # 공격 데이터 라벨

test_df = pd.concat([benign_df, attack_df], ignore_index=True)
y_true = test_df['is_anomaly']  
X_test_df = test_df.drop(columns=['is_anomaly'])

for col in columns:
    if col not in X_test_df.columns:
        X_test_df[col] = 0.0  # 없는 컬럼은 0으로 채움
X_test_df = X_test_df[columns]  # 컬럼 순서 맞추기

X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
print(f"테스트 준비 완료... Shape: {X_test.shape}")

# 모델에 통과시켜 재구성
with torch.no_grad():
    recon = model(X_test)
    loss = torch.mean((X_test - recon) ** 2, dim=1)  # MSE per sample



# 임계값 기준
# threshold = loss.mean() + 3 * loss.std() 
# threshold = loss.mean() + 2 * loss.std() # 임계값 줄이기

# JYP: 아래 내용은 최적의 임계값을 찾는 부분

precisions, recalls, thresholds = precision_recall_curve(y_true, loss.numpy())
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # F1 score 계산

best_threshold = thresholds[np.argmax(f1_scores)]  # F1 score가 최대인 임계값 선택
print(f"최적의 Threshold: {best_threshold:.4f}")

threshold = best_threshold  # 최적의 임계값 사용
anomalies = loss > threshold
# =========================================

print(f"Detected {anomalies.sum().item()} anomalies out of {len(loss)} samples.")

y_pred = (loss > threshold).int()  # 0: 정상, 1: 이상

report = classification_report(y_true, y_pred.cpu(), target_names=['Benign (0)', 'Attack (1)'])
print(report)

print("\n--- Visualizing Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred.cpu())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (0)', 'Attack (1)'], yticklabels=['Benign (0)', 'Attack (1)'])
plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.legend()
plt.show()

sorted_loss = np.sort(loss.numpy())
print("Top 10 highest losses:", sorted_loss[-10:])