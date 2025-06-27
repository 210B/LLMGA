from openai import OpenAI
import json
import csv
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

training_file = client.files.create(
    file=open("data/finetuning/Ophelia_merged.jsonl", "rb"),
    purpose="fine-tune"
)

print(f"파일 업로드 완료: ID = {training_file.id}")

fine_tune_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo"
)

print(f"파인튜닝 작업 생성됨! Job ID = {fine_tune_job.id}")

while True:
    job_status = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
    print(f"상태: {job_status.status}")
    if job_status.status in ["succeeded", "failed", "cancelled"]:
        break
    import time
    time.sleep(30)

print(f"파인튜닝 완료 결과: {job_status}")