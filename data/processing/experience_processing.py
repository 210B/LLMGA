import pandas as pd
import re
import json
import glob

protective_files = glob.glob("data/protective_experience/*.csv")
completion_files = glob.glob("data/experience_completion/*.csv")

csv_files = protective_files + completion_files

speaker_map = {
    "Lily": "user",
    "Miles": "user",
    "Claudius": "assistant",
    "Gertrude": "assistant",
    "Ophelia": "assistant"
}

pattern = r"^\**\s*([A-Za-z ]+)\s*\(speaking\)\s*(.*)$"

jsonl_lines = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    for idx, row in df.iterrows():
        dialogue_text = str(row['Dialogue'])

        lines = dialogue_text.strip().split('\n')

        system_message = f"I want you to act like White Rabbit from Alice's Adventures in Wonderland."
        messages = [{"role": "system", "content": system_message}]

        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                if speaker in speaker_map:
                    role = speaker_map[speaker]
                    messages.append({"role": role, "content": text})

        if len(messages) > 1:
            jsonl_line = json.dumps({"messages": messages}, ensure_ascii=False)
            jsonl_lines.append(jsonl_line)

output_file = 'data/finetuning/fine_tune_data.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for line in jsonl_lines:
        f.write(line + '\n')