import argparse


# Args parser
def parse_args():
    parser = argparse.ArgumentParser(description="CSAT_Kor settings")

    parser.add_argument("--seed", default=42, type=int, help="Reproducibility")
    parser.add_argument("--dataset", default="CSAT_Kor.csv", type=str, help="Dataset")
    parser.add_argument("--option", default=0, type=int, help="Option for template")
    parser.add_argument("--test_size", default=0.2, type=float, help="Test data split")
    parser.add_argument("--max_length", default=3072, type=int, help="Tokenizer")
    parser.add_argument("--delta", default=0.3, type=float, help="Huber loss")
    parser.add_argument("--r", default=8, type=int, help="Lora attention dimension")
    parser.add_argument("--lora_alpha", default=8, type=int, help="Lora scaling")
    parser.add_argument("--lora_dropout", default=0.0, type=float, help="Dropout")
    parser.add_argument("--batch_size", default=8, type=int, help="Train batch size")
    parser.add_argument("--cum_step", default=1, type=int, help="Accumulation steps")
    parser.add_argument("--epoch", default=3, type=int, help="Train epoch")
    parser.add_argument("--ls_type", default="linear", type=str, help="Scheduler type")
    parser.add_argument("--optim", default="adamw_torch", type=str, help="Optim type")

    args = parser.parse_args()

    return args
