import json
import os
import openai
from tqdm import tqdm

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """You are a compassionate and clear medical communicator.
Given a detailed and jargon-heavy hospital note (from the "text" field), write a concise, emotionally intelligent summary that a patient can understand.
Vary the tone based on severity: 
- If the visit contains a serious or terminal condition (e.g., stroke, cancer), be gentle and kind.
- If it’s a minor issue (e.g., flu, mild infection), be straightforward and reassuring.
Your goal is to be clear, kind, and helpful.
"""
DATA_PATH = "/app/data/postprocessed-data/mimic-iv-note-ext-di-bhc/dataset/train_100.json"
with open(DATA_PATH, "r") as f:
    records = [json.loads(line) for line in f]

for record in tqdm(records):
    text = record["text"]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=400
        )
        summary = response.choices[0].message.content
        record["empathy_summary"] = summary

    except Exception as e:
        record["empathy_summary"] = f"[Error generating summary: {str(e)}]"

with open("train_100_empathy.json", "w") as f:
    for record in records:
        json.dump(record, f)
        f.write("\n")