import json
import os
import concurrent.futures
from tqdm import tqdm
from utils.llm_judge_utils import get_llm_judge_with_retry


def checking_win_rate(filename):
    """
    Process a JSONL file to compute the win rate of the summaries.
    Args:
        filename (str): The path to the JSONL file.
    Returns:
        tuple: A tuple containing the filename and the win rate.
    """
    win_count, valid_score_count = 0, 0
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Processing {filename}", unit="line"):
            try:
                data = json.loads(line.strip())
                prompt = data.get('prompt')
                judge_result = get_llm_judge_with_retry(prompt)
                if judge_result:
                    score1, score2 = judge_result
                    valid_score_count += 1
                    if score1 > score2:
                        win_count += 1
            except json.JSONDecodeError:
                print(f"JSONDecodeError: {line}")
            except Exception as e:
                print(f"Error happened: {e}")

    if valid_score_count > 0:
        win_rate = win_count / valid_score_count
        print(f"The win rate of {filename} is: {win_rate:.2f} ({win_count}/{valid_score_count})")
        return filename, win_rate
    else:
        print(f"There is no valid prompt in {filename}")
        return filename, None


def main(directory):
    """
    Main function to process all JSONL files in the given directory.
    Args:
        directory (str): The path to the directory containing JSONL files.
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jsonl')]
    print(f"checking the files: {file_paths}")

    # Multi-threading, one thread per file
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(checking_win_rate, file_path): file_path for file_path in file_paths}
        
        with tqdm(total=len(file_paths), desc="Overall Progress", unit="file") as pbar:
            # wait for all threads to complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    filename, win_rate = future.result()
                    if win_rate is not None:
                        result_filename = filename.replace('.jsonl', '_result.jsonl')
                        with open(result_filename, 'w', encoding='utf-8') as result_file:
                            result_file.write(json.dumps({
                                "filename": filename,
                                "win_rate": win_rate,
                            }, ensure_ascii=False))
                except Exception as e:
                    print(f"Error happened in {file_path} : {e}")
                pbar.update(1)  # update progress bar


if __name__ == "__main__":
    directory = './'
    main(directory)
