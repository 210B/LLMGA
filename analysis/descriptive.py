import pandas as pd
from scipy.stats import ttest_ind

# 데이터 불러오기
df = pd.read_csv('data/questionnaire/final_scores_with_immersion.csv')

# 숫자형 열만 선택
numeric_cols = df.select_dtypes(include='number').columns

# 1. 그룹별 기초 통계량
summary = df.groupby('Model')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median', 'count'])
print("기초 통계량:")
print(summary)

# 2. 그룹 간 차이 (t-test)
print("\n📊 그룹 간 평균 차이 (t-test):")
for col in numeric_cols:
    A = df[df['Model'] == 'A'][col]
    B = df[df['Model'] == 'B'][col]
    t, p = ttest_ind(A, B, equal_var=False)
    print(f"{col:25s} | t = {t:6.2f}, p = {p:.4f}")
