import argparse


# Args parser
def parse_args():
    parser = argparse.ArgumentParser(description="CSAT_Kor settings")

    parser.add_argument("--seed", default=42, type=int, help="Reproducibility")
    parser.add_argument("--dataset", default="CSAT_Kor.csv", type=str, help="Dataset")

    args = parser.parse_args()

    return args
