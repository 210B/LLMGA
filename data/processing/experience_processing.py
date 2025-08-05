import os
import json

# 인물 리스트
base_names = ['Claudius', 'Gertrude', 'Ophelia']

# 디렉터리 경로
input_dir = 'data/finetuning'

# 각 인물에 대해 병합 수행
for base_name in base_names:
    complete_path = os.path.join(input_dir, f'{base_name}_complete.jsonl')
    protective_path = os.path.join(input_dir, f'{base_name}_protective.jsonl')
    merged_path = os.path.join(input_dir, f'{base_name}_merged.jsonl')

    try:
        # 모든 라인을 모아 리스트로 병합
        merged_lines = []
        for path in [complete_path, protective_path]:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as infile:
                    merged_lines.extend(infile.readlines())
            else:
                print(f"⚠️ 파일 없음: {path} — 생략됨.")

        '''# 마지막 메시지가 assistant인지 확인
        if merged_lines:
            try:
                last_json = json.loads(merged_lines[-1])
                if last_json.get("role") != "assistant":
                    print(f"⚠️ 마지막 메시지가 assistant가 아님. 삭제됨: {base_name}")
                    merged_lines = merged_lines[:-1]
            except json.JSONDecodeError:
                print(f"⚠️ 마지막 줄 JSON 파싱 실패. 삭제됨: {base_name}")
                merged_lines = merged_lines[:-1]'''

        # 병합 파일 저장
        with open(merged_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(merged_lines)

        print(f"✅ 병합 완료: {merged_path}")

    except Exception as e:
        print(f"❌ 병합 중 오류 ({base_name}): {e}")
