'''import csv
import json
import re
import os

# Character names to process
base_names = ['Claudius', 'Gertrude', 'Ophelia']

# Directory paths
input_dir = 'data/experience_completion'
output_dir = 'data/finetuning'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def parse_dialogue_to_messages(dialogue, base_name):
    system_prompt = f"I want you to act like {base_name} from Hamlet."
    messages = [{"role": "system", "content": system_prompt}]
    
    for line in dialogue.strip().splitlines():
        match = re.match(r'^([\w\s]+?)(?:\s*\(.*?\))?:\s*(.+)', line.strip())
        if match:
            speaker = match.group(1).strip()
            content = match.group(2).strip()
            role = "assistant" if speaker == base_name else "user"
            messages.append({"role": role, "content": content})
    return messages

# Loop over each character
for base_name in base_names:
    csv_file = os.path.join(input_dir, f'{base_name}.csv')
    jsonl_file = os.path.join(output_dir, f'{base_name}_complete.jsonl')

    try:
        with open(csv_file, 'r', encoding='utf-8-sig', newline='') as infile, open(jsonl_file, 'w', encoding='utf-8') as outfile:
            reader = csv.DictReader(infile)
            for row in reader:
                row = {k.strip(): v for k, v in row.items()}
                messages = parse_dialogue_to_messages(row['Dialogue'], base_name)
                json_obj = {
                    "messages": messages
                }
                json.dump(json_obj, outfile, ensure_ascii=False)
                outfile.write('\n')
        print(f"✅ Successfully converted {base_name}.csv → {base_name}_complete.jsonl")
    except FileNotFoundError:
        print(f"⚠️ File not found: {csv_file} — skipping.")'''

import pandas as pd
import re
import json
import glob
import os

csv_files = glob.glob("data/experience_completion/*.csv")

speaker_map = {
    "Lily": "user",
    "Miles": "user",
    "Claudius": "assistant",
    "Gertrude": "assistant",
    "Ophelia": "assistant"
}

system_messages = {
    "Claudius": "I want you to act like Claudius from Hamlet.",
    "Gertrude": "I want you to act like Gertrude from Hamlet.",
    "Ophelia": "I want you to act like Ophelia from Hamlet."
}

pattern = r"^\**\s*([A-Za-z ]+)\s*\(speaking\)\**\s*[:：\-]?\s*(.*)$"

for file_path in csv_files:
    df = pd.read_csv(file_path)
    jsonl_lines = []

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if base_name in system_messages:
        system_message = system_messages[base_name]
    else:
        system_message = "You are a helpful assistant."
        print(f"⚠️ Warning: No system message defined for '{base_name}'. Using default.")

    for _, row in df.iterrows():
        dialogue_text = str(row.get('Dialogue', '')).strip()
        lines = dialogue_text.split('\n')

        messages = [{"role": "system", "content": system_message}]
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match = re.match(pattern, line)
            if match:
                speaker = match.group(1).strip()
                content = match.group(2).strip()
                role = speaker_map.get(speaker)
                if role:
                    if content:
                        messages.append({"role": role, "content": content})
                    else:
                        i += 1
                        if i < len(lines):
                            next_line = lines[i].strip()
                            if next_line:
                                messages.append({"role": role, "content": next_line})
            i += 1

        if len(messages) > 1:
            jsonl_line = json.dumps({"messages": messages}, ensure_ascii=False)
            jsonl_lines.append(jsonl_line)

    # 저장
    output_path = f"data/finetuning/{base_name}_complete.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(line + "\n")

    print(f"✅ Processed: {base_name}")
