from parser import parse_args
from utils import set_seed, check_cuda_capability
from data import parent_dir, load_data, CSATPromptDataset
from models.exaone import EXAONERegressionModel, load_model

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from sklearn.model_selection import train_test_split
from functools import partial

import torch

import warnings
warnings.filterwarnings("ignore")


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


# Function to tokenize in custom settings
def collate_fn(batch, tokenizer, max_length):
    # Get data from batch
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    # Tokenizing
    tokenized_texts = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Check GPU quality
    cap_flag = check_cuda_capability()
    torch_dtype = torch.bfloat16 if cap_flag is True else torch.float16

    return {
        "input_ids": tokenized_texts["input_ids"],
        "attention_mask": tokenized_texts["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch_dtype),
    }


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

    # Get peft model
    peft_config = get_lora_config(args.r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_train_epochs=args.epoch,
        lr_scheduler_type=args.ls_type,
        warmup_steps=0,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=50,
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=args.optim,
        report_to="wandb",
    )

    # Define trainer for training
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=partial(collate_fn, tokenizer, args.max_length),
        train_dataset=csat_kor_train_dataset,
        eval_dataset=csat_kor_test_dataset,
    )
    if debug is True:
        model.print_trainable_parameters()

    # Train process
    trainer.train()

    # Reload the best model
    best_model = EXAONERegressionModel.from_pretrained("./results")
    if debug is True:
        print("Success!")
        print()


if __name__ == "__main__":
    args = parse_args()
    main(args, debug=False)
