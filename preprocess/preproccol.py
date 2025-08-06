# 파일마다 열 구조 비교
import pandas as pd

files = [
    # '../data/raw/BenignTraffic.csv',
    '../data/raw/BenignTraffic1.pcap_Flow.csv',
    '../data/raw/BenignTraffic2.pcap_Flow.csv',
    '../data/raw/BenignTraffic3.pcap_Flow.csv',
    '../data/raw/DDoS-HTTP_Flood-.pcap_Flow.csv'
]
# 기준 열: 첫 번째 파일의 열
base_columns = None
df = pd.read_csv('../data/raw/BenignTraffic1.pcap_Flow.csv')
print(df.shape[1])  # 컬럼 수 출력

for file in files:
    df = pd.read_csv(file, nrows=1)  # 빠르게 열만 보기 위해 한 줄만 읽음
    columns = list(df.columns)
    print(f"\n{file} ({len(columns)} columns):")
    print(columns)

    if base_columns is None:
        base_columns = columns
    else:
        if columns != base_columns:
            print("⚠️ 열 구조가 다릅니다!")
        else:
            print("✅ 열 구조가 동일합니다.")