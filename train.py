import torch
from torch.utils.data import DataLoader, TensorDataset
from models.autoencoder import AutoEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import os
import joblib


# 데이터 로딩
df = pd.read_csv('data/processed/benign_processed.csv')
data = torch.tensor(df.values, dtype=torch.float32)

# 학습/검증 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(train_data), batch_size=256, shuffle=True)

# 모델 구성
input_dim = data.shape[1]
model = AutoEncoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 30
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x = batch[0]
        output = model(x)
        loss = criterion(output, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.6f}")

# 검증
model.eval()
with torch.no_grad():
    val_output = model(val_data)
    val_loss = criterion(val_output, val_data)
    print(f"Validation Loss: {val_loss.item():.6f}")

# 저장
joblib.dump(df.columns.tolist(), "data/columns.pkl")
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/autoencoder.pth')
print("모델 저장 완료!")
