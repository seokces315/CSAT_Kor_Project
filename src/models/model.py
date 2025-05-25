from .pooling import AttentionPooling, mean_pooling
from .loss import compute_huber_loss

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


# Custom model to fine-tune EXAONE on regression task
class LLRegressionModel(nn.Module):
    # Generator
    def __init__(self, model, torch_dtype, loss_cat, delta, flag):
        super().__init__()
        self.backbone = model
        hidden_size = self.backbone.config.hidden_size
        self.attention_pooling = AttentionPooling(torch_dtype, hidden_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        # self.regressor = nn.Sequential(
        #     nn.Linear(hidden_size * 2, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_size, 1),
        #     nn.Sigmoid(),
        # )
        self.loss_cat = loss_cat
        self.delta = delta
        self.flag = flag
        self.to(dtype=torch_dtype)

    # Forward with last hidden state's last token
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        input_dicts = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.flag is True:
            outputs = self.backbone.base_model(**input_dicts)
        else:
            outputs = self.backbone(**input_dicts, output_hidden_states=True)

        # Pooling (Last token, Mean pooling, ...)
        if self.flag is True:
            # pooled = outputs.last_hidden_state[:, -1, :]
            # pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
            pooled = self.attention_pooling(outputs.last_hidden_state, attention_mask)
        else:
            last_hidden_state = outputs.hidden_states[-1]
            # pooled = last_hidden_state[:, -1, :]
            # pooled = mean_pooling(last_hidden_state, attention_mask)
            pooled = self.attention_pooling(last_hidden_state, attention_mask)

        # # Option : Hybrid
        # if self.flag is True:
        #     acum_token = outputs.last_hidden_state[:, -1, :]
        # else:
        #     acum_token = last_hidden_state[:, -1, :]
        # total_pooled = torch.cat([pooled, acum_token], dim=1)
        # preds = self.regressor(total_pooled).squeeze(-1)

        # Method : General
        preds = self.regressor(pooled).squeeze(-1)

        # Calculate loss
        if self.loss_cat == "mse":
            criterion = nn.MSELoss()
            loss = criterion(preds, labels)
        elif self.loss_cat == "mae":
            criterion = nn.L1Loss()
            loss = criterion(preds, labels)
        else:
            loss = compute_huber_loss(preds, labels, self.delta)

        return {"loss": loss, "logits": preds}

    # Dummy function
    def prepare_inputs_for_generation(
        self, input_ids=None, attention_mask=None, **kwargs
    ):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

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
def load_model(model_id, cap_flag, loss_cat, delta):
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
    flag = True if model_id == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" else False
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
    ll_reg_model = LLRegressionModel(model, torch_dtype, loss_cat, delta, flag)

    return tokenizer, ll_reg_model
