import json
import os
import re
import time
import openai
from tqdm import tqdm

openai.base_url = "" # Your LLM judge API url
openai.api_key = ""  # Your API key
JUDGE_MODEL = "" # the model name of your LLM judge


def get_llm_judge(content):
    """
    Call the LLM judge to get scores for the summaries.
    Args:
        content (str): The prompt to send to the LLM judge.
    Returns:
        str: The response from the LLM judge.
    """
    # create a chat completion
    completion = openai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": content}]
    )
    # print the completion
    return completion.choices[0].message.content


def get_llm_judge_with_retry(prompt, max_retries=3):
    """
    Call the LLM judge to get scores for the summaries.
    If the result is not a valid Python list, retry.
    Args:
        prompt (str): The prompt to send to the LLM judge.
        max_retries (int): The maximum number of retries if the result is invalid.
    Returns:
        list: A list of scores for the summaries.
        None: If all retries fail.
    """
    for i in range(max_retries):
        result = get_llm_judge(prompt)
        try:
            scores = []
            matches = re.findall(r'\[(\d+)\]', result)  # find all numbers in square brackets
            for match in matches:
                try:
                    score = int(match)  # convert the matched string to an integer
                    scores.append(score)
                except ValueError:
                    print(f"WARN: Can't chagne '{match}' into int.")
                    pass
            return scores
        except ValueError:
            print(f"Try {i+1}: ValueError: {result}")
            time.sleep(1)
            continue
    print(f"All {max_retries} retries failed.")
    return None
