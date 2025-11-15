from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import Dataset, DataLoader
from logger.logger import WandbLogger
from lora.module import LoRALayer
from torch.amp import autocast, GradScaler


# --- PARAMETERS ---
MODEL_NAME = "EleutherAI/pythia-1.4b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MAX_LENGTH = 512
EPOCHS = 4
LORA_R = 4
LORA_ALPHA = 1
RUN_NAME = "lora_hh_rlhf_demo"


# --- DATA LOAD & SPLIT ---
def split_prompt_response(sample):
    idx = sample.rfind("Assistant:")
    prompt = sample[:idx + len("Assistant:")].strip()
    response = sample[idx + len("Assistant:"):].strip()
    return {"prompt": prompt, "response": response}

def preprocess(example):
    prompt = example['prompt']
    response = example['response']

    prompt_tokens = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    response_tokens = tokenizer(response, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    full_text = prompt + " " + response
    full_tokens = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding='max_length')["input_ids"]

    labels = full_tokens.copy()
    # Маскируем только реальные токены prompt (до их длины но не превышая MAX_LENGTH)
    prompt_len = min(len(prompt_tokens), MAX_LENGTH)
    labels[:prompt_len] = [-100] * prompt_len

    # Проверка: есть размаскированные токены? Если нет - дропаем
    if all(l == -100 for l in labels):
        return None

    return {
        "input_ids": full_tokens,
        "attention_mask": [1 if t != tokenizer.pad_token_id else 0 for t in full_tokens],
        "labels": labels
    }


# --- DATASET ---
class DialogDataset(Dataset):
    def __init__(self, items):
        self.input_ids = [torch.tensor(x["input_ids"]) for x in items]
        self.attention_mask = [torch.tensor(x["attention_mask"]) for x in items]
        self.labels = [torch.tensor(x["labels"]) for x in items]
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


raw_data = load_dataset("Anthropic/hh-rlhf", split="train")
samples = [split_prompt_response(item['chosen']) for item in raw_data if "Assistant:" in item['chosen']]
samples = samples[:2000]  # уменьшить для демо, иначе памяти не хватит

# --- TOKENIZATION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

data_tkn = [x for x in (preprocess(d) for d in samples) if x is not None]

train_ds = DialogDataset(data_tkn)
dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --- MODEL & LORA ---
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)

print(model)

# -- Патчинг Linear слоя Attention в Pythia (GPTNeoXForCausalLM) --
with torch.no_grad():
    for layer in model.gpt_neox.layers:
        orig = layer.attention.query_key_value
        lora_layer = LoRALayer(orig, r=LORA_R, alpha=LORA_ALPHA).to(DEVICE)
        layer.attention.query_key_value = lora_layer

for param in model.parameters():
    param.requires_grad = False
for layer in model.gpt_neox.layers:
    for param in layer.attention.query_key_value.parameters():
        param.requires_grad = True

# --- OPTIMIZER ---
lora_params = []
for layer in model.gpt_neox.layers:
    lora_params += list(layer.attention.query_key_value.parameters())
optimizer = torch.optim.Adam(lora_params, lr=5e-5)

# --- Wandb logger ---
logger = WandbLogger(project="nlp_dpo", run_name=RUN_NAME)

# --- WANDB CONFIG ---
logger.log_config({
    "model_name": MODEL_NAME,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "n_samples": len(train_ds)
})

scaler = GradScaler('cuda')

# --- TRAIN LOOP ---
logger.watch(model)
step_count = 0
for epoch in range(EPOCHS):
    epoch_losses = []
    for batch in dataloader:
        print(batch["input_ids"].shape)
        print(batch["input_ids"])
        print("Any nan?", torch.isnan(batch["input_ids"].float()).any())
        print("Max value:", batch["input_ids"].max())
        print("Min value:", batch["input_ids"].min())
        print("dtype:", batch["input_ids"].dtype)

        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        print("Batch input ids:", input_ids)
        print("Is nan in input_ids?", torch.isnan(input_ids.float()).any())

        # with autocast('cuda'):
        #     outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        #     loss = outputs.loss
        # # Масштабируем градиенты
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(loss.item())

        # print('Loss:', loss.item())
        # for name, p in model.named_parameters():
        #     if p.grad is not None:
        #         print(name, p.grad.abs().max().item())

        ppl = torch.exp(loss).item()
        logger.log_step(step_count, **{"lora_pythia_loss": loss.item(), 'lora_pythia_ppl': ppl})
        epoch_losses.append(loss.item())
        step_count += 1
    print(f"Epoch {epoch+1}: Mean loss {np.mean(epoch_losses):.4f}")

logger.finish()

# --- EVAL GENERATION ---
model.eval()
import random
for _ in range(5):
    idx = random.randint(0, len(samples)-1)
    prompt = samples[idx]['prompt']
    input_text = prompt
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.to(DEVICE)
    gen_ids = model.generate(input_ids, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print("-"*40)
    print("PROMPT:", prompt)
    print("GENERATED:", gen_text.replace(prompt, '').strip())
    print("TRUE:", samples[idx]['response'])
