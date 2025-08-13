import pandas as pd

# 1. 파일 경로
id_file = "data/questionnaire/실험참가자최종.xlsx"        # 참가자 번호만 있는 파일
survey_file = "data/questionnaire/연구참가자모집설문(응답).xlsx"     # 설문 응답 데이터

# 2. 파일 불러오기
df_ids = pd.read_excel(id_file)           # 컬럼: ParticipantID
df_survey = pd.read_excel(survey_file)    # 컬럼: 이름, 나이, ...

# 3. 행 수가 같을 때만 가능
assert len(df_ids) == len(df_survey), "행 수가 일치하지 않습니다."

# 4. ParticipantID를 맨 앞에 붙이기
df_combined = pd.concat([df_ids, df_survey], axis=1)
columns_to_exclude = [
    "이름",
    "모델 구분",
    "게임 1",
    "게임 2",
    "Unnamed: 5",
    "Unnamed: 6",
    "Unnamed: 7",
    "Unnamed: 8",
    "타임스탬프",
    "[연구참가 동의]",
    "성함을 알려주세요.",
    "만 나이를 알려주세요.",
    "참가자로 선정될 시 안내 받으실 핸드폰 번호를 알려주세요. (010-****-**** 형식)",
    "실험에는 약 60분 정도 소요됩니다. 다음 링크에서 가능하신 시간을 선택해주세요. 구글 캘린더 시간 선택 후 구글 폼으로 돌아오셔서 ❗설문 응답 제출❗까지 부탁드립니다.\nhttps://calendar.app.google/LHKr82S2Rhi4SzYf9\n\n일정 조율이 필요하시다면 문의 오픈 카톡으로 연락주세요.\n※ 문의 오픈 카카오톡: https://open.kakao.com/o/s9jVzjHh",
    "이메일 주소"
]
df_combined = df_combined.drop(columns=columns_to_exclude)

column_rename_map = {
    "ID": "ID",
    "평소에 게임을 얼마나 자주 하십니까?": "game_frequency",
    "선호하는 게임 장르를 모두 선택해 주세요.": "game_genre",
    "시뮬레이션 게임을 플레이한 경험이 있습니까?": "simul_exp",
    "AI 또는 게임 내 캐릭터와의 상호작용이 중심인 게임을 플레이해 본 적이 있습니까?": "AIgame_exp",
    "앞의 문항에서 '예'라고 대답했다면 기억나는 게임 이름을 입력해주세요.": "AIgame_sample",
    "셰익스피어의 작품에 대해 들어본 적이 있습니까?": "shakespeare",
    "'햄릿' 작품을 어떤 방식으로 접하셨습니까?(중복 선택 가능)": "hamlet_approach",
    "햄릿 작품의 줄거리나 등장인물에 대해 어느 정도 알고 계십니까?": "hamlet_understanding",
    "'베니스의 상인' 작품을 어떤 방식으로 접하셨습니까?(중복 선택 가능)": "venice_approach",
    "베니스의 상인 작품의 줄거리나 등장인물에 대해 어느 정도 알고 계십니까?": "venice_understanding",
    "대형 언어 모델(LLM, 예: ChatGPT, Gemini, Claude 등)에 대해 들어본 적이 있습니까? ": "LLM",
    "대형 언어 모델을 사용해 본 적 있습니까?": "LLM_exp",
    "앞의 문항에서 '예'라고 대답했다면 주로 사용한 모델 이름을 입력해주세요.": "LLM_model",
    "평소에 언어 모델을 얼마나 자주 사용하십니까?": "LLM_frequency",
    "대형 언어 모델의 작동 원리에 대해 어느 정도 알고 계십니까?": "LLM_understanding",
    "대형 언어 모델이 없는 사실을 지어내서 말하는 현상에 대해 어느 정도 알고 계십니까?": "hallucination",
    "위와 같은 현상으로 인해 불편함을 경험하신 적이 있습니까?": "hallu_exp",
    "ChatGPT와 같이 대형 언어 모델 기반의 에이전트와 상호작용하는 시스템에 대해 알고 계셨습니까? ": "LLM_agent"
}

# 2. 데이터 불러오기 & 컬럼명 변경
df = pd.read_csv("processed_output/prior_survey.csv")
df.rename(columns=column_rename_map, inplace=True)

# 3. 매핑 딕셔너리
yes_no_map = {"예": 1, "아니오": 0}

frequency_map = {
    "자주(주 3회 이상)": 3,
    "보통(주 1~2회)": 2,
    "가끔(월 1~2회)": 1,
    "거의 하지 않음": 0
}

understanding_map = {
    "세부적인 원리까지 이해함": 3,
    "대략적인 원리 이해함": 2,
    "대략적인 작동 방식 이해함": 1,
    "이름과 개념만 들어봄": 0.5,
    "전혀 모름": 0
}

sunderstanding_map = {
    "세부적인 내용까지 잘 알고 있음": 3,
    "주요 인물과 줄거리를 대략적으로 알고 있음": 2,
    "작품의 이름과 대략적인 배경만 알고 있음": 1,
    "전혀 모름": 0
}
hallu_exp_map = {
    "전혀 없음": 0,
    "가끔 경험함": 1,
    "자주 경험함": 2
}
media_weights = {
    "원작 희곡 또는 번역본을 읽음": 3,
    "요약/해설 자료를 읽음": 2,
    "연극/영화/드라마로 감상": 2,
    "학교 수업 등 교육 과정": 1.5,
    "인터넷/미디어를 통해 간접적으로만 알고 있음": 1
}

# 4. 가중치 점수 계산 함수
def compute_weighted_score(response, weight_map):
    if pd.isna(response):
        return 0
    items = [x.strip() for x in str(response).split(",")]
    return sum(weight_map.get(item, 0) for item in items)

# 5. 인코딩 수행

# Binary 예/아니오
for col in ["simul_exp", "AIgame_exp", "shakespeare", "LLM", "LLM_exp", "LLM_agent"]:
    if col in df.columns:
        df[col] = df[col].map(yes_no_map)

# Frequency
df["game_frequency"] = df["game_frequency"].map(frequency_map)
df["LLM_frequency"] = df["LLM_frequency"].map(frequency_map)

# 이해도
df["LLM_understanding"] = df["LLM_understanding"].map(understanding_map)
df["hallucination"] = df["hallucination"].map(understanding_map)

# 셰익스피어 작품 이해도
df["hamlet_understanding"] = df["hamlet_understanding"].map(sunderstanding_map)
df["venice_understanding"] = df["venice_understanding"].map(sunderstanding_map)

# 접근 방식 점수
df["hamlet_approach"] = df["hamlet_approach"].apply(lambda x: compute_weighted_score(x, media_weights))
df["venice_approach"] = df["venice_approach"].apply(lambda x: compute_weighted_score(x, media_weights))

df["hallu_exp"] = df["hallu_exp"].map(hallu_exp_map)
# 6. 저장
df.to_csv("processed_output/prior_survey_processed.csv", index=False, encoding="utf-8-sig")
