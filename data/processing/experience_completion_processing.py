import pandas as pd
import re
import json

df = pd.read_csv('data/experience_completion/complete_experience.csv')

def parse_dialogue(dialogue):
    lines = dialogue.strip().split('\n')
    messages = []
    current_role = None
    current_content = ""

    for line in lines:
        line = line.strip()
        match = re.match(r"^(.*?)[\s]*\((speaking|thinking)\)\s*$", line)
        if match:
            if current_role and current_content.strip():
                messages.append({
                    "role": current_role,
                    "content": current_content.strip().replace('\n', ' ')
                })
            character = match.group(1).strip()
            current_role = "assistant" if character.lower() == "white rabbit" else "user"
            current_content = ""
        else:
            current_content += " " + line.strip()

    if current_role and current_content.strip():
        messages.append({
            "role": current_role,
            "content": current_content.strip().replace('\n', ' ')
        })

    if messages and messages[-1]['role'] == 'user':
        messages.pop()

    return messages

fine_tune_data = []
for _, row in df.iterrows():
    conversation = parse_dialogue(row['Dialogue'])
    if conversation:
        fine_tune_data.append({"messages": conversation})

with open("fine_tune_data_complete.jsonl", "w", encoding="utf-8") as f:
    for item in fine_tune_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")