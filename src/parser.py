import argparse


# Args parser
def parse_args():
    parser = argparse.ArgumentParser(description="CSAT_Kor settings")

    parser.add_argument("--seed", default=42, type=int, help="Reproducibility")
    parser.add_argument("--dataset", default="CSAT_Kor.csv", type=str, help="Dataset")
    parser.add_argument("--test_size", default=0.2, type=float, help="Test data split")
    parser.add_argument("--option", default=0, type=int, help="Option for template")
    parser.add_argument("--r", default=8, type=int, help="Lora attention dimension")
    parser.add_argument("--lora_alpha", default=8, type=int, help="Lora scaling")
    parser.add_argument("--lora_dropout", default=0.0, type=float, help="Dropout")
    parser.add_argument("--delta", default=0.3, type=float, help="Huber loss")

    args = parser.parse_args()

    return args
