import os

# 인물 리스트
base_names = ['Claudius', 'Gertrude', 'Ophelia']
base_names = ['Claudius']

# 디렉터리 경로
input_dir = 'data/finetuning'

# 각 인물에 대해 병합 수행
for base_name in base_names:
    complete_path = os.path.join(input_dir, f'{base_name}_complete.jsonl')
    protective_path = os.path.join(input_dir, f'{base_name}_protective.jsonl')
    merged_path = os.path.join(input_dir, f'{base_name}_merged.jsonl')

    try:
        with open(merged_path, 'w', encoding='utf-8') as outfile:
            for path in [complete_path, protective_path]:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
                else:
                    print(f"⚠️ 파일 없음: {path} — 생략됨.")
        print(f"✅ 병합 완료: {merged_path}")
    except Exception as e:
        print(f"❌ 병합 중 오류 ({base_name}): {e}")