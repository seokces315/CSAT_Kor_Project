from parser import parse_args
from utils import set_seed, check_cuda_capability
from data import parent_dir, CSATPromptDataset, label_to_diff, load_data

from models.model import load_model

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import torch
import numpy as np

import warnings
import logging
import wandb

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
wandb.init(project="Exaone-finetuning")


# Function to compute metrics
def compute_metrics(eval_preds):
    # Unpacking
    predictions, labels = eval_preds

    # # For difficulty columns
    # predictions = np.array([label_to_diff(p) for p in predictions])
    # labels = np.array([label_to_diff(l) for l in labels])

    # Dimension control
    predictions = torch.tensor(predictions.squeeze())
    labels = torch.tensor(labels.squeeze())

    # Compute accuracy with specific metric
    correct = (torch.abs(predictions - labels) <= 0.10).sum()
    # correct = (predictions == labels).sum()
    total = labels.shape[0]

    accuracy = 1.0 * correct / total
    mape = mean_absolute_percentage_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)

    return {"mse": mse, "mape": mape, "accuracy": accuracy}


# Function for LoRA settings
def get_lora_config(r, lora_alpha, lora_dropout):
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        target_modules=["q_proj", "v_proj"],  # GPT-series
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )
    return lora_config


# Function to tokenize in custom settings
def wrap_collate_fn(tokenizer, max_length):
    # Check GPU quality
    cap_flag = check_cuda_capability()
    torch_dtype = torch.bfloat16 if cap_flag is True else torch.float16

    def collate_fn(batch):
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

        return {
            "input_ids": tokenized_texts["input_ids"],
            "attention_mask": tokenized_texts["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch_dtype),
        }

    return collate_fn


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
    if args.model_id == "exaone":
        model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    elif args.model_id == "llama":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        model_id = "Qwen/Qwen3-8B"
    tokenizer, model = load_model(
        model_id, cap_flag=cap_flag, loss_cat=args.loss_cat, delta=args.delta
    )

    # Get peft model
    peft_config = get_lora_config(args.r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.cum_step,
        num_train_epochs=args.epoch,
        lr_scheduler_type=args.ls_type,
        warmup_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        data_seed=args.seed,
        dataloader_drop_last=True,
        eval_steps=50,
        run_name="Exaone-finetuning",
        disable_tqdm=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="mape",
        greater_is_better=False,
        optim=args.optim,
        report_to="wandb",
        full_determinism=True,
    )

    # Define trainer for training
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=wrap_collate_fn(tokenizer, args.max_length),
        train_dataset=csat_kor_train_dataset,
        eval_dataset=csat_kor_test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10, early_stopping_threshold=1e-9
            )
        ],
    )
    if debug is True:
        model.print_trainable_parameters()

    # Train process
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args, debug=False)
