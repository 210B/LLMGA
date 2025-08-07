import pandas as pd

# 파일 불러오기 (파일 경로에 따라 수정)
file_path = 'data/questionnaire/[베니스의상인]플레이설문(응답).xlsx'  # <- 각 파일마다 바꿔서 반복 사용
df = pd.read_excel(file_path)

# 리버스 스코어링이 필요한 문항 인덱스 또는 열 이름 지정
# 예시: 아래는 열 이름 기준이며, 문항이 다르면 수정 필요
reverse_items = [
    '캐릭터가 지나치게 차분하거나 논리적이거나 과장된 방식으로 행동해 어색했다.',
    '일관성 없는 캐릭터의 행동이 너무 자주 등장해 몰입이 방해되었다.',
    '캐릭터가 작품 배경 설정을 벗어나는 말을 했다.',
    '특정 대사 이후, 게임의 다음 전개에 대한 흥미가 떨어졌다.',
    '캐릭터가 자기모순적인 말을 했다.',
    '캐릭터가 지나치게 차분하거나 논리적이거나 과장된 방식으로 행동해 어색했다.',
    '부조리한 캐릭터의 행동이 너무 자주 등장해 몰입이 방해되었다.',
    '캐릭터의 대사가 감정적 몰입(공감, 긴장 등)을 방해했다.'
]

# 🔹 참가자 ID 열 이름
id_col = '참가자 ID를 입력해주세요.'

# 🔹 리버스 스코어링 함수
def reverse_score(x):
    try:
        x = int(x)
        return 8 - x
    except:
        return None

# 🔹 점수 열만 추출
score_columns = []
for col in df.columns:
    if col == id_col:
        continue
    try:
        vals = pd.to_numeric(df[col], errors='coerce')
        if vals.dropna().between(1, 7).all():
            score_columns.append(col)
    except:
        continue

# 🔹 점수 처리 + ID 포함
scored_df = df[[id_col] + score_columns].copy()
scored_df[id_col] = scored_df[id_col].astype(str).str.upper()  # ✅ ID 대문자화

for col in score_columns:
    if col in reverse_items:
        scored_df[col] = scored_df[col].apply(reverse_score)
    else:
        scored_df[col] = pd.to_numeric(scored_df[col], errors='coerce')

# 🔹 총점 계산
scored_df['score_sum'] = scored_df[score_columns].sum(axis=1, numeric_only=True)

# 🔹 저장 (점수 + 총합)
scored_df.to_csv('data/questionnaire/scored_venice.csv', index=False, encoding='utf-8-sig')

# 🔹 주관식 열만 추출 (ID 포함, 점수 열 제외)
text_columns = [col for col in df.columns if (
    ('이유' in col or '추가로' in col)
)]

text_df = df[[id_col] + text_columns].copy()
text_df[id_col] = text_df[id_col].astype(str).str.upper()

# 🔹 저장 (텍스트 응답)
text_df.to_csv('data/questionnaire/text_venice.csv', index=False, encoding='utf-8-sig')
