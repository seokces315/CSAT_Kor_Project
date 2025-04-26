import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


# Function for mean pooling
def mean_pooling(last_hidden_state, attention_mask):
    # Broadcasting
    hidden_size = last_hidden_state.size()
    expanded_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_size)

    # Zero-out & Sum given hidden states
    sum_embeddings = (last_hidden_state * expanded_attention_mask).sum(dim=1)

    # Count of non-masked tokens per sample
    sum_mask = expanded_attention_mask.sum(dim=1).clamp(min=1e-9)

    # Mean pooling
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings


# Function to define Huber loss
def compute_huber_loss(preds, labels, delta):
    assert len(preds) == len(labels)

    abs_error = torch.abs(preds - labels)
    delta = torch.tensor(delta, device=abs_error.device)

    is_mini_error = abs_error <= delta

    mini_error_loss = 0.5 * (abs_error**2)
    big_error_loss = delta * (abs_error - 0.5 * delta)
    loss = torch.where(is_mini_error, mini_error_loss, big_error_loss)

    return loss.mean()


# Custom model to fine-tune EXAONE on regression task
class EXAONERegressionModel(nn.Module):
    # Generator
    def __init__(self, model, torch_dtype, delta):
        super().__init__()
        self.backbone = model
        hidden_size = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        # self.regressor = self.regressor.to(dtype=torch_dtype)
        self.delta = delta

    # Forward with last hidden state's last token
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        input_dicts = {"input_ids": input_ids, "attention_mask": attention_mask}

        outputs = self.backbone.base_model(**input_dicts)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        # pooled = outputs.last_hidden_state[:, -1, :]
        preds = self.regressor(pooled).squeeze(-1)

        # Calculate huber loss
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
    reg_model = EXAONERegressionModel(model, torch_dtype, delta)

    return tokenizer, reg_model
