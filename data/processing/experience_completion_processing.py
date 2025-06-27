'''import pandas as pd
import re
import json

df = pd.read_csv('data/experience_completion/Claudius.csv')

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
        f.write(json.dumps(item, ensure_ascii=False) + "\n")'''



import csv
import json
import re

csv_file = 'data/experience_completion/Claudius.csv'
jsonl_file = 'data/finetuning/Claudius_complete.csv'

def parse_dialogue_to_messages(dialogue):
    messages = []

    # Add initial system prompt
    messages.append({
        "role": "system",
        "content": "You are Claudius, king of Denmark, speaking in a Shakespearean tone with fear, cunning, and royal authority."
    })

    # Split into individual lines and process
    for line in dialogue.strip().splitlines():
        # Match pattern: Speaker (optional tags): content
        match = re.match(r'^([\w\s]+?)(?:\s*\(.*?\))?:\s*(.+)', line.strip())
        if match:
            speaker = match.group(1).strip()
            content = match.group(2).strip()
            role = "assistant" if speaker == "Claudius" else "user"
            messages.append({"role": role, "content": content})

    return messages

with open(csv_file, 'r', encoding='utf-8-sig', newline='') as infile, open(jsonl_file, 'w', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    for row in reader:
        row = {k.strip(): v for k, v in row.items()}
        messages = parse_dialogue_to_messages(row['Dialogue'])
        json_obj = {
            "emotion": row['Emotion'],
            "description": row['Description'],
            "messages": messages
        }
        json.dump(json_obj, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"✅ Successfully converted to {jsonl_file}")

