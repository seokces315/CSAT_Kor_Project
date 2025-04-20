from parser import parse_args
from utils import set_seed, check_cuda_capability
from data import parent_dir, load_data, CSATPromptDataset
from models.exaone import load_model

from transformers import Trainer
from peft import LoraConfig

from sklearn.model_selection import train_test_split


# Define custom trainer for regression
class EXAONERegressionTrainer(Trainer):
    # Overriding
    def compute_loss(self, model, input_dicts, return_outputs=False):
        outputs = model(**input_dicts)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# Function for LoRA settings
def get_lora_config(r, lora_alpha, lora_dropout):
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )
    return lora_config


# Main flow
def main(args, debug=False):

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load dataset
    data_path = f"{parent_dir}/data/{args.dataset}"
    csat_kor_df = load_data(data_path)

    # Prepare train/test custom dataset
    csat_kor_train_df, csat_kor_test_df = train_test_split(
        csat_kor_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=csat_kor_df["difficulty"],
    )
    csat_kor_train_dataset = CSATPromptDataset(csat_kor_train_df, option=args.option)
    csat_kor_test_dataset = CSATPromptDataset(csat_kor_test_df, option=args.option)
    if debug is True:
        print(len(csat_kor_train_dataset))
        print(len(csat_kor_test_dataset))
        print()

    # Load tokenizer & model
    cap_flag = check_cuda_capability()
    model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    tokenizer, model = load_model(model_id, cap_flag=cap_flag, delta=args.delta)

    # Define training arguments
    peft_config = get_lora_config(args.r, args.lora_alpha, args.lora_dropout)


if __name__ == "__main__":
    args = parse_args()
    main(args, debug=False)

# # Model definition
# id_list = [model_id.split("/")[1], "CSAT_Kor"]
# tuned_model = "-".join(id_list)
