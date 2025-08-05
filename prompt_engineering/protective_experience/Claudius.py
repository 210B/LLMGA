from openai import OpenAI
import json
import os
import csv
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

CHARACTER_NAME = "Claudius"

with open("prompts/protective_experience.txt", "r", encoding="utf-8") as f:
    base_prompt = f.read()

base_prompt = base_prompt.format(character_name=CHARACTER_NAME)

all_experience = []

# 5개의 summary 파일에 대해 반복
for i in range(1, 6):
    summary_file = f"prompts/summary{i}.txt"

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_prompt = f.read()

    full_prompt = f"Context:\n\"\"\"\n{summary_prompt.strip()}\n\"\"\"{base_prompt.strip()}\n\n"

    # 프롬프트 요청
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        n=3
    )

    # 결과 저장
    for j, choice in enumerate(response.choices):
        content = choice.message.content.strip()
        print(f"\n--- Result {i}-{j+1} ---\n{content}\n")
        all_experience.append({
            "Dialogue": content
        })

output_path = f"data/protective_experience/{CHARACTER_NAME}.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Dialogue"])

    for experience in all_experience:
        dialogue = experience["Dialogue"]
        writer.writerow([dialogue])