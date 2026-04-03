import pandas as pd
import numpy as np

# 데이터 생성 (n=100명)
np.random.seed(42)
n = 100

data = {
    'Patient_ID': range(1, n+1),
    # 독립변수: 치료군 (A: 기존치료, B: 신규 정밀의료 타겟 치료)
    'Group': np.random.choice(['A', 'B'], n),
    # 종속변수: 뇌암 마커 수치 (정규분포 가정)
    'Marker_Level': np.where(np.random.choice(['A', 'B'], n) == 'A', 
                             np.random.normal(50, 10, n), # Group A 평균 50
                             np.random.normal(60, 12, n)), # Group B 평균 60
    # 범주형 변수: 뇌암 병기 (1~4기)
    'Stage': np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.4, 0.2, 0.1])
}

df = pd.DataFrame(data)
print(df.head())

# 여기서 T-test나 ANOVA를 바로 돌려보실 수 있습니다.

