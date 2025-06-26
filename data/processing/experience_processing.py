import os
import glob
import pandas as pd
import re
import json

# 두 디렉터리에서 CSV 파일 목록 수집
protective_files = glob.glob("data/protective_experience/*.csv")
completion_files = glob.glob("data/experience_completion/*.csv")

# 파일명(확장자 제외)으로 매칭
protective_map = {os.path.splitext(os.path.basename(f))[0]: f for f in protective_files}
completion_map = {os.path.splitext(os.path.basename(f))[0]: f for f in completion_files}

# 공통된 파일명만 추출
common_keys = set(protective_map.keys()) & set(completion_map.keys())

# 화자 역할 매핑
speaker_map = {
    "Lily": "user",
    "Miles": "user",
    "Claudius": "assistant",
    "Gertrude": "assistant",
    "Ophelia": "assistant"
}

# 캐릭터별 시스템 메시지
system_messages = {
    "Claudius": "I want you to act like Claudius from Hamlet.",
    "Gertrude": "I want you to act like Gertrude from Hamlet.",
    "Ophelia": "I want you to act like Ophelia from Hamlet."
    }

# 정규표현식 패턴
pattern = r"^\**\s*([A-Za-z ]+)\s*\(speaking\)\s*(.*)$"

# 공통 키에 대해 병합 후 처리
for key in common_keys:
    df1 = pd.read_csv(protective_map[key])
    df2 = pd.read_csv(completion_map[key])
    df = pd.concat([df1, df2], ignore_index=True)

    jsonl_lines = []
    system_message = system_messages.get(key, "You are a helpful assistant.")

    for idx, row in df.iterrows():
        dialogue_text = str(row['Dialogue'])
        lines = dialogue_text.strip().split('\n')

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

    # 저장
    output_file = f"data/finetuning/{key}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in jsonl_lines:
            f.write(line + '\n')

    print(f"{len(jsonl_lines)} dialogues written to {output_file}")