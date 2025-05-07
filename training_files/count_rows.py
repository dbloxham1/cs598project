import pandas as pd
import argparse

def main(input_path):
    df = pd.read_csv(input_path)
    print(f"Number of rows: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    main(args.input_path)