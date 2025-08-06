import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
df = pd.read_csv('../data/raw/BenignTraffic.csv')

# 상위 5개 출력
print(df.head())
print(df.columns)
print(df.shape)


# 열 제거
drop_cols = ['src_ip', 'dst_ip', 'src_mac', 'dst_mac', 'stream']
df = df.drop(columns=drop_cols)


# 결측치 제거
df = df.dropna()

# 무한값 제거
df = df[~df.isin([float('inf'), float('-inf')]).any(axis=1)]

# 숫자형 데이터만 추출
df_numeric = df.select_dtypes(include=['number'])

# 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_numeric)

scaled_df = pd.DataFrame(scaled_data)  # numpy → DataFrame으로 변환
scaled_df.to_csv('../data/processed/benign_processed.csv', index=False)

# 확인용 출력
print(scaled_data[:5])