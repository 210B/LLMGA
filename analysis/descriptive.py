import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro
import math

# 1. 데이터 불러오기
df = pd.read_csv('data/questionnaire/final_scores_with_immersion.csv')

# 2. 숫자형 열만 선택
numeric_cols = df.select_dtypes(include='number').columns

# 3. 기초 통계량 저장
summary = df.groupby('Model')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median', 'count'])
summary.to_csv("processed_output/groupwise_descriptives.csv", encoding="utf-8-sig")
print("📊 기초 통계량 저장 완료")

# 4. 정규성 검사 저장
shapiro_results = []
for col in numeric_cols:
    for model in ['A', 'B']:
        stat, p = shapiro(df[df['Model'] == model][col])
        shapiro_results.append({
            'Variable': col,
            'Model': model,
            'Shapiro-W': round(stat, 4),
            'p-value': round(p, 4)
        })
shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df.to_csv("processed_output/shapiro_normality.csv", index=False, encoding="utf-8-sig")
print("📈 정규성 검사 결과 저장 완료")

# 5. t-test 결과 저장
ttest_results = []
for col in numeric_cols:
    A = df[df['Model'] == 'A'][col]
    B = df[df['Model'] == 'B'][col]
    t, p = ttest_ind(A, B, equal_var=False)
    ttest_results.append({
        'Variable': col,
        't-statistic': round(t, 4),
        'p-value': round(p, 4)
    })
ttest_df = pd.DataFrame(ttest_results)
ttest_df.to_csv("processed_output/ttest_results.csv", index=False, encoding="utf-8-sig")
print("📊 t-test 결과 저장 완료")

# 6. Boxplot 그리드 저장
sns.set(style="whitegrid")

n = len(numeric_cols)
cols = 2
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 6, rows * 4))

for idx, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, cols, idx)
    sns.boxplot(data=df, x='Model', y=col, palette='Set2')
    sns.stripplot(data=df, x='Model', y=col, color='black', alpha=0.5, jitter=True)
    plt.title(f"{col}")
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.suptitle("📦 변수별 Model 그룹 분포 (Box + Stripplot)", fontsize=16, y=1.02)
plt.savefig("figures/groupwise_boxplots.png", bbox_inches='tight', dpi=300)
plt.show()
print("📦 박스플롯 시각화 저장 완료")