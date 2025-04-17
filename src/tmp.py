# %pip install accelerate peft bitsandbytes transformers trl

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

import huggingface_hub


huggingface_hub.login("Huggingface API KEY")

if torch.cuda.get_device_capability()[0] >= 8:
    # !pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# Hugging Face Basic Model 한국어 모델
# base_model = "teddylee777/Llama-3-Open-Ko-8B-gguf"  # 테디님의 Llama3 한국어 파인튜닝 모델
base_model = "beomi/Llama-3-Open-Ko-8B"  # beomi님의 Llama3 한국어 파인튜닝 모델

# 주가 증권 보고서 gemini 데이터셋
hkcode_dataset = "uiyong/gemini_result_kospi_0517_jsonl"

# 새로운 모델 이름
new_model = "Llama3-owen-Ko-3-8B-kospi"

# llama 데이터 로드
dataset = load_dataset(hkcode_dataset, split="train")

# Weight freeze 된 Model을 4bit 혹은 8bit로 불러오되, 업데이트 되는 lora layer는 FP32 혹은 BF32로 불러옴
# 리소스 제약시 FP 16보다 BF16 을 통해 효율적으로 훈련
# https://www.youtube.com/watch?v=ptlmj9Y9iwE
# QLoRA config
# compute_dtype = getattr(torch, "float16"), torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    # bnb_4bit_use_double_quant=False,
)

### 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=quant_config, device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

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

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

from tensorboard import notebook

log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))

prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(
    task="text-generation", model=model, tokenizer=tokenizer, max_length=200
)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]["generated_text"])

# model.print_trainable_parameters()
