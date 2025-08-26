import fire
from datasets import load_dataset, get_dataset_split_names
import json
import os


def save_to_jsonl(dataset, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")


def download_dataset(dataset_path, subset=None, output_dir="."):
    try:
        # Get available splits
        splits = get_dataset_split_names(path=dataset_path, config_name=subset)

        # Check if splits is only one
        if len(splits) == 1:
            dataset = load_dataset(dataset_path, name=subset, split=splits[0])

            # Split the dataset into train and test (80:20)
            split_dataset = dataset.train_test_split(
                test_size=0.2, shuffle=True, seed=42
            )

            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save train split
            train_output = os.path.join(output_dir, "train.jsonl")
            save_to_jsonl(split_dataset["train"], train_output)
            print(f"Saved train split to {train_output}")

            # Save test split
            test_output = os.path.join(output_dir, "test.jsonl")
            save_to_jsonl(split_dataset["test"], test_output)
            print(f"Saved test split to {test_output}")
        else:
            for split in splits:
                # Load specific split
                dataset = load_dataset(dataset_path, name=subset, split=split)

                # Create output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_file = os.path.join(output_dir, f"{split}.jsonl")

                # Save to JSONL
                save_to_jsonl(dataset, output_file)
                print(f"Saved {split} split to {output_file}")

    except Exception as e:
        print(f"Error processing dataset: {str(e)}")


if __name__ == "__main__":
    fire.Fire(download_dataset)
