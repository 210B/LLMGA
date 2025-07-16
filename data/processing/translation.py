import json
import os
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_translation"))

# 경로 설정
input_file_path = Path("data/finetuning/Gertrude_merged.jsonl")
output_file_path = Path("data/finetuning/Gertrude_translated.jsonl")

# 메시지 content만 번역하는 함수 (role은 그대로 유지)
def translate_content_only(role, content):
    prompt = f"""
다음 텍스트는 셰익스피어 스타일로 쓰인 대화 데이터셋입니다. 이를 한국어로 번역할 때 아래 지침을 반드시 따르세요:

1. 화자의 성격, 계급, 감정에 따라 말투를 조절할 것  
 - 귀족이나 고위 인물은 격식 있고 문학적인 말투  
 - 하층민이나 서민은 구어적이고 거친 말투 가능  
 - 감정(분노, 비탄, 기쁨 등)이 뚜렷하면 말투에 강하게 반영할 것  

2. 고전적 분위기를 유지할 것  
 - 중세풍/시적인 분위기와 운율을 살리되  
 - 현대 한국어 화자가 이해할 수 있도록 자연스럽게 구성할 것  

3. 은유, 상징, 시적 표현을 가능한 한 유지하거나 한국어에 맞게 변형할 것  
 - 직역이 부자연스러울 경우, 원문의 의도와 분위기를 살려 의역할 것  

4. 문맥(희극/비극/로맨스 등)에 맞는 번역 분위기를 유지할 것  
 - 희극: 유쾌하고 익살스럽게  
 - 비극: 장중하고 비장하게  
 - 로맨스: 서정적이고 낭만적으로  

5. 문체는 일관성 있게 유지할 것  
 - 현대적 문어체 또는 고전체 혼합 문체 중 하나를 선택하여 일관되게 유지  
 - 인물의 계급/성격에 따라 현대 회화체 혼용 가능 (특히 하층민)  

6. 화자의 역할명(assistant, user 등)은 번역하지 말 것  

7. 시스템 메시지는 등장인물에 따라 자연스럽게 한국어로 번역할 것. 예: "I want you to act like Ophelia from Hamlet." → "햄릿 작품의 오필리어처럼 행동해."

이 지침을 기준으로 각 문장을 자연스럽고 문학적으로 번역하세요.

대사:
{content}
    """.strip()

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    result =  response.choices[0].message.content.strip()
    print(f"→ 번역 결과: {result[:40]}...\n")  # 디버깅 출력
    return result

# JSONL 파일 읽기
with input_file_path.open("r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

translated = []

# 각 대화 처리
for convo in tqdm(lines, desc="번역 중"):
    translated_messages = []
    for msg in convo["messages"]:
        translated_text = translate_content_only(msg["role"], msg["content"])
        translated_messages.append({
            "role": msg["role"],  # 역할은 그대로 유지
            "content": translated_text
        })
    translated.append({"messages": translated_messages})

# 결과 저장
with output_file_path.open("w", encoding="utf-8") as f:
    for convo in translated:
        f.write(json.dumps(convo, ensure_ascii=False) + "\n")

print(f"번역 완료: {output_file_path}")
