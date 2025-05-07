import pandas as pd
import re
import json
import random
import argparse
import os

# Revised pipeline to accept CLI arguments:
#   1) input_file: path to discharge notes CSV
#   2) output_dir: directory to write train.json, val.json, test.json

# Section delimiters
DI_START = r'Discharge Instructions:'
DI_END = r'Followup Instructions:'
BHC_START = r'Brief Hospital Course:'
BHC_END = r'Discharge Instructions:'

# Cleanup patterns
GREETING_PATTERN = r'^Dear [^\n]*\n'
CLOSING_PATTERN = r'Your [^\n]* Team\.$'


def extract_summary(text: str) -> str:
    """
    Extract Discharge Instructions (DI) as the summary.
    """
    m = re.search(DI_START + r'(.*?)' + DI_END, text, flags=re.DOTALL)
    if not m:
        return None
    summary = m.group(1).strip()
    # Remove greetings and sign-off
    summary = re.sub(GREETING_PATTERN, '', summary)
    summary = re.sub(CLOSING_PATTERN, '', summary, flags=re.MULTILINE).strip()
    # Length filter
    if len(summary) < 350:
        return None
    # Sentence count filter
    sentences = re.split(r'[.!?]+', summary)
    if len([s for s in sentences if s.strip()]) < 3:
        return None
    # Blank-line filter
    if summary.count('\n\n') > 5:
        return None
    # De-identification token density
    deid_tokens = len(re.findall(r'_{2,}', summary))
    word_count = len(summary.split())
    if word_count > 0 and deid_tokens / (word_count / 10) > 1:
        return None
    return summary


def extract_context(text: str) -> str:
    """
    Extract Brief Hospital Course (BHC) as context.
    """
    m = re.search(BHC_START + r'(.*?)' + BHC_END, text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def preprocess(input_file: str) -> list:
    df = pd.read_csv(input_file)
    examples = []
    for _, row in df.iterrows():
        raw = row['text']
        summary = extract_summary(raw)
        if not summary:
            continue
        context = extract_context(raw)
        if not context:
            continue
        examples.append({'text': context, 'summary': summary})
    return examples


def split_and_save(examples: list, output_dir: str, seed: int):
    random.seed(seed)
    random.shuffle(examples)
    n = len(examples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.json')
    val_path = os.path.join(output_dir, 'val.json')
    test_path = os.path.join(output_dir, 'test.json')

    train_set = examples[:train_end]
    val_set = examples[train_end:val_end]
    test_set = examples[val_end:]

    print(f"Saving {len(train_set)} train / {len(val_set)} val / {len(test_set)} test examples to {output_dir}")

    with open(train_path, 'w') as f:
        json.dump(train_set, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_set, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_set, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Process discharge notes to produce train/test/validation JSON splits.'
    )
    parser.add_argument('input_file', help='CSV file with discharge notes')
    parser.add_argument('output_dir', help='Directory to save output JSON files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()

    examples = preprocess(args.input_file)
    print(f"Extracted {len(examples)} examples after filtering.")
    split_and_save(examples, args.output_dir, args.seed)


if __name__ == '__main__':
    main()
