'''import pandas as pd
import re
import json
import glob
import os

csv_files = glob.glob("data/protective_experience/*.csv")

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

# 정규식: 말하는 사람 이름 추출 + 같은 줄에 대사 있으면 캡처
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
                    if content:  # 같은 줄에 대사 포함
                        messages.append({"role": role, "content": content})
                    else:  # 다음 줄에서 대사 가져오기
                        i += 1
                        if i < len(lines):
                            next_line = lines[i].strip()
                            if next_line:
                                messages.append({"role": role, "content": next_line})
            i += 1

        if len(messages) > 1:
            jsonl_line = json.dumps({"messages": messages}, ensure_ascii=False)
            jsonl_lines.append(jsonl_line)

    output_path = f"data/finetuning/{base_name}_protective.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in jsonl_lines:
            f.write(line + '\n')

    print(f"[{base_name}] {len(jsonl_lines)} dialogues written to {output_path}")
'''

import pandas as pd
import re
import json
import glob
import os

csv_files = glob.glob("data/protective_experience/*.csv")

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
    output_path = f"data/protective_experience/jsonl/{base_name}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(line + "\n")

    print(f"✅ Processed: {base_name}")
