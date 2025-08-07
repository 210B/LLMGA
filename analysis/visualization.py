import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 데이터 불러오기
df = pd.read_csv('data/questionnaire/final_scores.csv', encoding='utf-8-sig')

# 🔹 시각화 설정
sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))

# 🔹 산점도 (색상: 모델 구분)
scatter = sns.scatterplot(
    data=df,
    x='Hamlet_score',
    y='Venice_score',
    hue='Model',
    palette='Set1',
    s=100
)

# 🔹 각 점에 ID 텍스트 표시
for i in range(len(df)):
    plt.text(
        df['Hamlet_score'][i] + 0.1,  # 약간 오른쪽으로
        df['Venice_score'][i],
        df['ID'][i],
        fontsize=9
    )

# 🔹 제목 및 축 설정
plt.title('Hamlet vs Venice Total Scores by Model Group')
plt.xlabel('Hamlet_score')
plt.ylabel('Venice_score')
plt.legend(title='Model')
plt.tight_layout()

# 🔹 저장 (옵션)
plt.savefig('figures/scatter_by_model.png', dpi=300)

# 🔹 화면 표시
plt.show()

df = pd.read_csv('data/questionnaire/final_scores_with_immersion.csv', encoding='utf-8-sig')
df['ID'] = df['ID'].astype(str)

# 🔹 스타일 설정
sns.set(style='whitegrid')

# 🔹 박스플롯 그리기
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='Model', y='immersion_total', palette='Set2')
plt.title('Immersion Total Score by Model Group')
plt.xlabel('Model')
plt.ylabel('Immersion Total Score')
plt.tight_layout()

# 🔹 저장
plt.savefig('figures/boxplot_immersion_by_model.png', dpi=300)

# 🔹 화면 표시
plt.show()