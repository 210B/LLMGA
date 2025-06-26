from openai import OpenAI
import os
import csv
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY_completion")

client = OpenAI(api_key=api_key)

# 프롬프트 로딩
with open("prompts/experience_completion.txt", "r", encoding="utf-8") as f:
    base_prompt = f.read().strip()

# 처리할 캐릭터 목록
characters = ["Claudius"]
input_dir = "data/experience_extraction"
output_dir = "data/experience_completion"
os.makedirs(output_dir, exist_ok=True)

for character in characters:
    input_path = os.path.join(input_dir, f"{character}.csv")
    output_path = os.path.join(output_dir, f"{character}.csv")

    # 캐릭터명을 반영한 프롬프트 구성
    prompt_with_character = base_prompt.replace("{character_name}", character)

    all_dialogues = []

    with open(input_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= 3:
                break  # 테스트용 상위 3개만 처리

            emotion = row["Emotion"].strip()
            description = row["Description"].strip()

            # 최종 프롬프트 조립
            full_prompt = (
                f"The setting is provided only for your reference and must NOT be included in your answer.\n\n"
                f"Setting:\n"
                f"- Type: Script\n"
                f"- Emotion: {emotion}\n"
                f"- Description: {description}\n\n"
                f"{prompt_with_character}\n\n"
            )

            # GPT 호출
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )

            content = response.choices[0].message.content.strip().replace("\n\n", "\n")
            print(f"\n--- {character} #{i+1} ---\n{content}\n")

            all_dialogues.append({
                "Character": character,
                "Emotion": emotion,
                "Description": description,
                "Dialogue": content
            })

    # CSV 저장
    with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Emotion", "Description", "Dialogue"])  # 헤더에서 Character 제거
        for item in all_dialogues:
            writer.writerow([
                item["Emotion"],
                item["Description"],
                item["Dialogue"]
            ])
