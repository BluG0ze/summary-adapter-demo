import os
import json


LLM_JUDGE_PROMPT_TEMPLATE = """You are an expert in summarization. I will give you the original text and two candidate summaries. Please help me score these summaries.
The scoring criteria are:
1. Whether the summarization concisely and accurately expresses the meaning of the original text
2. Whether the language of the summarization is consistent with the original text.
3. The score needs to be between 0-10

Output format:
1. summary1: [[4]], summary2: [[5]]
2. do not output any explanation and other text

original text:
{original_text}

candidate summaries:
1. {summary}
2. {response0}
"""

def process_results(doc, results):
    p_id = doc["id"]
    original_text = doc["text"]
    summary = doc["summary"]
    prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
        original_text=original_text,
        summary=summary,
        response0=results[0],
    )
    with open("./llm_judge_prompts.jsonl", "a", encoding="utf-8") as json_file:
        json_file.write(json.dumps({"id": p_id, "prompt": prompt}, ensure_ascii=False) + "\n")

    return {
        "acc": 1
    }