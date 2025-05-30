from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

with open("prompts/experience_extraction_baseprompt.txt", "r", encoding="utf-8") as f:
    base_prompt = f.read()

all_experiences = []

# 4개의 summary 파일에 대해 반복
for i in range(1, 5):
    summary_file = f"prompts/experience_extraction_hamletsummary{i}.txt"

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_prompt = f.read()

    full_prompt = f"{base_prompt.strip()}\n\nInput Text:\n\"\"\"\n{summary_prompt.strip()}\n\"\"\""

    # 프롬프트 요청
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        n=3
    )

    # 결과 저장
    for j, choice in enumerate(response.choices):
        content = choice.message.content.strip()
        print(f"\n--- Result {i}-{j+1} ---\n{content}\n")

        try:
            parsed = json.loads(content)
            experiences_list = parsed.get("experiences", [])
            all_experiences.extend(experiences_list)  # 전체 리스트에 추가
        except json.JSONDecodeError:
            print(f"[Warning] Result {i}-{j+1} is not valid JSON. Skipped saving.")

# 최종 JSONL 파일 저장
output_path = "data/experience_extraction/experiences.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for experience in all_experiences:
        json_line = json.dumps(experience, ensure_ascii=False)
        f.write(json_line + "\n")