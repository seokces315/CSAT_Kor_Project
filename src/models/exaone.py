import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


# Function to define Huber loss
def compute_huber_loss(preds, labels, delta):
    assert len(preds) == len(labels)

    abs_error = torch.abs(preds - labels)
    delta = torch.tensor(delta).to(abs_error.device)
    mini = torch.minimum(abs_error, delta)
    linear = abs_error - mini
    loss = 0.5 * (mini**2) + delta * linear
    loss = loss.mean()

    return loss


# Custom model to fine-tune EXAONE on regression task
class EXAONERegressionModel(nn.Module):
    # Generator
    def __init__(self, model, delta):
        super().__init__()
        self.backbone = model
        hidden_size = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.delta = delta

    # Forward with last hidden state's last token
    def forward(self, input_dicts):
        labels = input_dicts["labels"]
        input_dicts = {
            k: v
            for k, v in input_dicts.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        outputs = self.backbone.base_model(**input_dicts)
        pooled = outputs.last_hidden_state[:, -1, :]
        preds = self.regressor(pooled).squeeze(-1)

        # Calculate huber loss
        loss = compute_huber_loss(preds, labels, self.delta)
        return {"loss": loss, "logits": preds}

    @property
    def config(self):
        return self.backbone.config


# Function for quantization
def ret_quant_config(torch_dtype):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
    )
    return quant_config


# Function to load tokenizer & model
def load_model(model_id, cap_flag, delta):
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Use cap_flag
    if cap_flag:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    # Quantization config
    quant_config = ret_quant_config(torch_dtype=torch_dtype)

    # Load model with upper options
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        quantization_config=quant_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Additional configs
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # model.gradient_checkpointing_enable()

    # Create a child model for specific task
    reg_model = EXAONERegressionModel(model, delta)

    return tokenizer, reg_model
