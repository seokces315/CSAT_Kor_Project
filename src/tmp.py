# %pip install accelerate trl
from transformers import TrainingArguments
from trl import SFTTrainer
import huggingface_hub

huggingface_hub.login("Huggingface API KEY")

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    warmup_steps=0.03,
    push_to_hub=False,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# model.print_trainable_parameters()

trainer.train()

trainer.model.save_pretrained(new_model)
