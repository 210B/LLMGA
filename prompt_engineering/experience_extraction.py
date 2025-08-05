from openai import OpenAI
import json
import os
import re
from dotenv import load_dotenv
from collections import defaultdict
import pandas as pd

load_dotenv()  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

with open("prompts/experience_extraction.txt", "r", encoding="utf-8") as f:
    base_prompt = f.read()

# 캐릭터별 데이터 저장용 딕셔너리
character_experiences = defaultdict(list)
target_characters = {"Claudius", "Gertrude", "Ophelia"}

# 5개의 summary 파일에 대해 반복
for i in range(1, 6):
    summary_file = f"prompts/summary{i}.txt"

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_prompt = f.read()

    full_prompt = f"{base_prompt.strip()}\n\nInput Text:\n\"\"\"\n{summary_prompt.strip()}\n\"\"\""

    # 프롬프트 요청
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": full_prompt}],
        n=3
    )

    # 결과 저장
    for j, choice in enumerate(response.choices):
        content = choice.message.content.strip()
        print(f"\n--- Result {i}-{j+1} ---\n{content}\n")

        try:
            # 줄 단위로 JSON 객체 추출
            lines = re.findall(r'\{.*?\}(?=\n|$)', content, flags=re.DOTALL)
            parsed_objects = [json.loads(line) for line in lines]

            # 필터링 및 캐릭터별 저장
            for obj in parsed_objects:
                if obj.get("Character") in target_characters:
                    character_experiences[obj["Character"]].append(obj)

        except json.JSONDecodeError as e:
            print(f"[Warning] Result {i}-{j+1} is not valid JSONL. Skipped saving. Error: {e}")

# 캐릭터별 CSV 파일 저장
output_dir = "data/experience_extraction"
os.makedirs(output_dir, exist_ok=True)

for character, records in character_experiences.items():
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, f"{character}.csv"), index=False, encoding="utf-8-sig")