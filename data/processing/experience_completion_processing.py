import csv
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
        print(f"⚠️ File not found: {csv_file} — skipping.")