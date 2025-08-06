import torch
from models.autoencoder import AutoEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# 전처리된 데이터 불러오기
df = pd.read_csv('data/processed/benign_processed.csv')
data = torch.tensor(df.values, dtype=torch.float32)

# 학습 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 모델, loss, optimizer 설정
input_dim = data.shape[1]
model = AutoEncoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 20
for epoch in range(epochs):
    model.train()
    output = model(train_data)
    loss = criterion(output, train_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


torch.save(model.state_dict(), 'models/autoencoder.pth')
print("모델 저장 완료!")