

import json
import argparse
import os
import random

def load_json(path):
    """Load JSON list from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    """Save list as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def split_dataset(data, train_ratio, seed):
    """Shuffle and split dataset into train and test sets."""
    rng = random.Random(seed)
    rng.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description="Split synthetic CS441 dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    data = load_json(args.input)

    # Train/test split
    train_set, test_set = split_dataset(data, args.train_ratio, args.seed)

    # Save files
    train_path = os.path.join(args.output_dir, "cs441_synthetic_train.json")
    test_path = os.path.join(args.output_dir, "cs441_synthetic_test.json")

    save_json(train_path, train_set)
    save_json(test_path, test_set)

    print(f"Dataset split completed.")
    print(f"Train samples: {len(train_set)} → {train_path}")
    print(f"Test samples:  {len(test_set)} → {test_path}")

if __name__ == "__main__":
    main()