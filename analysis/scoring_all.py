import pandas as pd

# 🔹 파일 불러오기
immersion_df = pd.read_excel('data/questionnaire/[종합]플레이경험설문(응답).xlsx')  # ← 종합 설문 결과 파일 경로
final_scores = pd.read_csv('data/questionnaire/final_scores.csv', encoding='utf-8-sig')

# 🔹 참가자 ID 정리
id_col = '참가자 ID를 입력해주세요.'
immersion_df[id_col] = immersion_df[id_col].astype(str).str.upper()
final_scores['ID'] = final_scores['ID'].astype(str).str.upper()

# 🔹 역채점 항목
reverse_items = [
    '게임 중에도 현실 세계를 의식하고 있었다.',
    '주변에서 일어나는 일을 인식하고 있었다.',
    '주위를 살피기 위해 게임을 멈추고 싶은 충동이 들었다.'
]

# 🔹 하위 항목 정의
subscales = {
    'immersion_attention': [
        '게임이 내 주의를 끌었다.',
        '나는 게임에 집중하고 있었다.',
        '나는 게임을 하기 위해 노력을 기울였다.',
        '나는 최선을 다하고 있다고 느꼈다.'
    ],
    'immersion_time_loss': [
        '게임을 하면서 시간 가는 줄 몰랐다.',
        '게임 중에도 현실 세계를 의식하고 있었다.',      # 역채점
        '일상적인 걱정을 잊고 있었다.',
        '주변에서 일어나는 일을 인식하고 있었다.',        # 역채점
        '주위를 살피기 위해 게임을 멈추고 싶은 충동이 들었다.'  # 역채점
    ],
    'immersion_presence': [
        '현실 환경과 분리된 느낌이 들었다.',
        '게임은 활동이라기보다는 하나의 경험처럼 느껴졌다.',
        '게임 속 존재감이 현실보다 더 강하게 느껴졌다.',
        '게임에 너무 몰입해서 조작하고 있다는 것을 잊었다.',
        '내가 원하는 대로 게임 속에서 움직이고 있다는 느낌이 들었다.'
    ],
    'immersion_emotional': [
        '게임에 감정적으로 몰입되었다.',
        '게임의 전개가 어떻게 될지 궁금했다.',
        '시뮬레이션의 끝이나 최종 상태에 도달하는 데 관심이 있었다.',
        '게임에 너무 몰입해서 게임과 직접 대화하고 싶었다.'
    ],
    'immersion_enjoyment': [
        '게임 속 대화(채팅)를 즐겼다.',
        '게임하는 것이 즐거웠다.',
        '이 게임을 다시 하고 싶다.',
        '다른 결말을 보기 위해 게임을 다시 플레이하고 싶다.'
    ]
}

# 🔹 역채점 처리
def reverse_score(x):
    try:
        x = int(x)
        return 8 - x
    except:
        return x

for col in reverse_items:
    if col in immersion_df.columns:
        immersion_df[col] = immersion_df[col].apply(reverse_score)

# 🔹 하위 항목 점수 계산
score_df = pd.DataFrame()
score_df['ID'] = immersion_df[id_col]

for scale_name, items in subscales.items():
    missing = [item for item in items if item not in immersion_df.columns]
    if missing:
        raise KeyError(f"다음 항목이 설문 데이터에 없습니다: {missing}")
    score_df[scale_name] = immersion_df[items].sum(axis=1)

# 🔹 총점
subscale_cols = list(subscales.keys())
score_df['immersion_total'] = score_df[subscale_cols].sum(axis=1)

# 🔹 기존 점수 데이터와 병합
final_merged = final_scores.merge(score_df, on='ID', how='left')

# 🔹 저장
final_merged.to_csv('data/questionnaire/final_scores_with_immersion.csv', index=False, encoding='utf-8-sig')