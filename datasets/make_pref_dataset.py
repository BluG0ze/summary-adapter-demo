import json
import os
import concurrent.futures
from tqdm import tqdm
from utils.llm_judge_utils import get_llm_judge_with_retry


SYSTEM_MSG = """You are an expert in summarization tasks. You are good at summarizing long texts into concise and accurate summarizations. The text you summarize needs to meet the following three requirements:
1. Your summarization must be concise and accurately express the meaning of the original text
2. The language of the summarization must be consistent with the original text
3. You only need to output the summary, without various polite words and other useless words"""
USER_PROMPT_PREFIX = "Summarize the following text: \n"


def transfer_to_pref_dataset(filename):
    """
    Given a JSONL file with LLM judge prompts, process it to create a preference dataset by using the LLM judge.
    Args:
        filename (str): The path to the JSONL file.
    Returns:
        int: The number of new lines generated.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm(lines, desc=f"Processing {filename}", unit="line"):
            data = json.loads(line.strip())

            # Extract the original text and the two summaries from the prompt
            prompt = data.get('prompt')
            original_text = prompt.split("original text:\n")[-1].split("\ncandidate summaries:")[0].strip()
            summary = prompt.split("candidate summaries:\n1. ")[-1].split("\n")[0].strip()
            response0 = prompt.split("\n2. ")[-1].split("\n")[0].strip()

            # get the LLM judge to compare the two summaries, making the preference data
            judge_result = get_llm_judge_with_retry(prompt)
            if judge_result:
                score1, score2 = judge_result
                chsn, rej = response0, summary
                if score1 > score2:
                    chsn, rej = summary, response0 
                new_data = {
                    "id": data["id"],
                    "chosen": [
                        {"role": "system", "content": SYSTEM_MSG}, 
                        {"role": "user", "content": USER_PROMPT_PREFIX + original_text}, 
                        {"role": "assistant", "content": chsn}
                    ],
                    "rejected": [
                        {"role": "system", "content": SYSTEM_MSG}, 
                        {"role": "user", "content": USER_PROMPT_PREFIX + original_text}, 
                        {"role": "assistant", "content": rej}
                    ],
                }
                new_lines.append(json.dumps(new_data, ensure_ascii=False) + "\n")

    if len(new_lines) > 0:
        with open(filename.replace(".jsonl", "_processed.jsonl"), 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return len(new_lines)
    else:
        print(f"File {filename} is empty after processing.")
        return 0


def main(directory):
    """
    Main function to transfer all LLM judge prompts into preference datasets in the given directory.
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jsonl')]
    print(f"checking the files: {file_paths}")

    # Multi-threading, one thread per file
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(transfer_to_pref_dataset, file_path): file_path for file_path in file_paths}

        with tqdm(total=len(file_paths), desc="Overall Progress", unit="file") as pbar:
            # wait for all threads to complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    len_newlines = future.result()
                    print(f"File {file_path} processed successfully, generated {len_newlines} new JSON lines.")
                except Exception as e:
                    print(f"Error happened in {file_path} : {e}")
                pbar.update(1)  # update progress bar


if __name__ == "__main__":
    directory = './'
    main(directory)
