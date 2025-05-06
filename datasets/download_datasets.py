import os
import json
from datasets import load_dataset


def save_dataset_to_jsonl(dataset, file_path):
    """
    Save a dataset to a JSONL file.
    Args:
        dataset (Dataset): The dataset to save.
        file_path (str): The path to the output JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dump(example, f, ensure_ascii=False) + "\n")


def download_subset(lang, lang_path, subset):
    """
    Download a specific subset of the dataset for a given language.
    Args:
        lang (str): The language code.
        lang_path (str): The path to save the dataset.
        subset (str): The subset to download (train, validation, test).
    """
    try:
        subdataset = load_dataset("csebuetnlp/xlsum", lang, split=subset)
        subdataset.save_to_disk(os.path.join(lang_path, subset))
        subdataset_path = os.path.join(lang_path, f"{subset}.jsonl")
        save_dataset_to_jsonl(subdataset, subdataset_path)
        print(f"successfully downloaded and saved {lang} subset's {subset}.")
    except ValueError as e:
        print(f"{lang} does not have {subset}: {e}")


if __name__ == "__main__":
    langs = ["chinese_traditional", "english", "korean", "japanese"]
    output_path = "./xlsum_datasets"
    os.makedirs(output_path, exist_ok=True)

    for lang in langs:
        try:
            lang_path = os.path.join(output_path, lang)
            os.makedirs(lang_path, exist_ok=True)

            for subset in ["train", "validation", "test"]:
                download_subset(lang, lang_path, subset)

        except Exception as e:
            print(f"Error downloading {lang} dataset: {e}")

    print("All datasets downloaded and saved successfully.")
