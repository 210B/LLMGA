import pandas as pd

# 🔹 모델 구분 정보 불러오기 (이름, 게임 1/2 제외)
info_df = pd.read_excel('data/questionnaire/실험참가자최종.xlsx', usecols=['ID', '모델 구분'])
info_df['ID'] = info_df['ID'].astype(str).str.upper()

# 🔹 Hamlet 점수 불러오기
hamlet_df = pd.read_csv('data/questionnaire/scored_hamlet.csv', encoding='utf-8-sig')
hamlet_df['참가자 ID를 입력해주세요.'] = hamlet_df['참가자 ID를 입력해주세요.'].astype(str).str.upper()
hamlet_scores = hamlet_df[['참가자 ID를 입력해주세요.', 'score_sum']].rename(
    columns={'참가자 ID를 입력해주세요.': 'ID', 'score_sum': 'Hamlet_score'}
)

# 🔹 Venice 점수 불러오기
venice_df = pd.read_csv('data/questionnaire/scored_venice.csv', encoding='utf-8-sig')
venice_df['참가자 ID를 입력해주세요.'] = venice_df['참가자 ID를 입력해주세요.'].astype(str).str.upper()
venice_scores = venice_df[['참가자 ID를 입력해주세요.', 'score_sum']].rename(
    columns={'참가자 ID를 입력해주세요.': 'ID', 'score_sum': 'Venice_score'}
)

# 🔹 병합: 모델 구분 + Hamlet + Venice
merged = info_df.merge(hamlet_scores, on='ID', how='left')
merged = merged.merge(venice_scores, on='ID', how='left')

# 🔹 저장
merged.to_csv('data/questionnaire/final_scores.csv', index=False, encoding='utf-8-sig')