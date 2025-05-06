import json
import os
from utils.prompts import SYSTEM_MSG, USER_PROMPT_PREFIX

def making_completion_format(input_file, output_file):
    """
    transform the input JSONL file into completion format for SFT training.
    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path to the output JSONL file.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                data = json.loads(line.strip())

                # making completion format
                new_data = {
                    "id": data["id"],
                    "url": data["url"],
                    "title": data["title"],
                    "article": data["text"],
                    "summary": data["summary"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": USER_PROMPT_PREFIX + data["text"]},
                        {"role": "assistant", "content": data["summary"]}
                    ]
                }

                outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

            except json.JSONDecodeError:
                print(f"JSONDecodeError: {line.strip()}")
            except KeyError as e:
                print(f"KeyError: {e} in line: {line.strip()}")


def process_directory(directory):
    """
    Process all JSONL files in the given directory and transform them into a new format.
    Args:
        directory (str): The path to the directory containing JSONL files.
    """
    for lang in ["chinese_traditional", "english", "japanese", "korean"]:
        for subset in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
            input_file = os.path.join(directory, lang, subset)
            if os.path.exists(input_file):
                output_file = os.path.join(directory, lang, f"transformed_{subset}")
                print(f"transferring {input_file} -> {output_file}")
                making_completion_format(input_file, output_file)
            else:
                print(f"File not found: {input_file}")


if __name__ == "__main__":
    directory_to_process = "./xlsum_datasets/"
    process_directory(directory_to_process)
    print("All files have been processed.")

