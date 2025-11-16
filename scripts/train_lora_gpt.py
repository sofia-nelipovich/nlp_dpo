import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lora.module import LoRALayer
from logger.logger import WandbLogger

# --- Имя эксперимента через аргументы ---
parser = argparse.ArgumentParser(description="GPT2 LoRA vs FT comparison")
parser.add_argument('--run_name', type=str, required=True, help='Имя wandb-run (эксперимента)')
parser.add_argument('--epochs', type=str, required=True, help='Число эпох (эксперимента)')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = WandbLogger(project='nlp_dpo', run_name=args.run_name + '_ft')

# --- Подсчёт параметров ---
def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct_trainable = 100 * trainable_params / total_params
    return total_params, trainable_params, pct_trainable

def print_and_log_model_parameters(model, name):
    total_params, trainable_params, pct_trainable = get_model_parameters(model)
    print(f"\n{name} model:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Percent trainable:     {pct_trainable:.2f}%")
    logger.log_config({f"{name}_total_params": total_params, \
                       f"{name}_trainable_params": trainable_params, \
                       f"{name}_pct_trainable": pct_trainable})

# --- Данные ---
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Нужно для GPT2
inputs = [
    "Hello world!",
    "LoRA comparison test.",
    "Small batch for demo.",
    "Fine-tuning baseline.",
    "Deep learning rocks.",
    "Graduate homework example!"
]
encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=16)
input_ids = encodings["input_ids"].to(DEVICE)
labels = input_ids.clone()

epochs = int(args.epochs)

# --- Fine-Tuning (FT) ---
model_ft = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
for p in model_ft.parameters():
    p.requires_grad = True
print_and_log_model_parameters(model_ft, "FT")

optimizer = optim.Adam([p for p in model_ft.parameters() if p.requires_grad], lr=5e-4)
ft_fw_times, ft_bw_times, ft_mems = [], [], []

for epoch in range(epochs):
    epoch_loss = []
    for step in range(len(inputs)):
        optimizer.zero_grad()
        batch_input = input_ids[step].unsqueeze(0)
        batch_label = labels[step].unsqueeze(0)

        start_fw = time.time()
        outputs = model_ft(batch_input, labels=batch_label)
        fw_time = time.time() - start_fw

        loss = outputs.loss

        start_bw = time.time()
        loss.backward()
        bw_time = time.time() - start_bw
        optimizer.step()

        mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        ft_fw_times.append(fw_time)
        ft_bw_times.append(bw_time)
        ft_mems.append(mem)

        # === LOG STEP ===
        logger.log_step(epoch * len(inputs) + step, **{
            "loss": loss.item(),
            "forward_time_ms": fw_time * 1000,
            "backward_time_ms": bw_time * 1000,
            "memory_mb": mem,
        })

        epoch_loss.append(loss.item())
    print(f"Epoch {epoch+1}: Loss = {np.mean(epoch_loss):.4f}")

# Итоговая статистика FT
logger.log_summary("mean_forward_ms", np.mean(ft_fw_times) * 1000)
logger.log_summary("mean_backward_ms", np.mean(ft_bw_times) * 1000)
logger.log_summary("max_memory_mb", np.max(ft_mems))

logger.finish()

del model_ft
del optimizer
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- LoRA ---
logger = WandbLogger(project='nlp_dpo', run_name=args.run_name + '_lora')
model_lora = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
# Patch Conv1D -> LoRA (заменяем только первый mlp.c_fc)
with torch.no_grad():
    orig_conv = model_lora.transformer.h[0].mlp.c_fc  # GPT2: Conv1D weight [in, out], bias [out]
    lora_layer = LoRALayer(orig_conv, r=4, alpha=16).to(DEVICE)
    model_lora.transformer.h[0].mlp.c_fc = lora_layer

for param in model_lora.parameters():
    param.requires_grad = False
for param in lora_layer.parameters():
    param.requires_grad = True
print_and_log_model_parameters(model_lora, "LoRA")

optimizer = optim.Adam(lora_layer.parameters(), lr=5e-4)
lora_fw_times, lora_bw_times, lora_mems = [], [], []

for epoch in range(epochs):
    epoch_loss = []
    print("Эпоха", epoch, "B min/max", lora_layer.B.data.min().item(), lora_layer.B.data.max().item())
    print("Эпоха", epoch, "A min/max", lora_layer.A.data.min().item(), lora_layer.A.data.max().item())
    for step in range(len(inputs)):
        optimizer.zero_grad()
        batch_input = input_ids[step].unsqueeze(0)
        batch_label = labels[step].unsqueeze(0)

        start_fw = time.time()
        outputs = model_lora(batch_input, labels=batch_label)
        fw_time = time.time() - start_fw

        loss = outputs.loss

        start_bw = time.time()
        loss.backward()
        bw_time = time.time() - start_bw
        optimizer.step()

        mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        lora_fw_times.append(fw_time)
        lora_bw_times.append(bw_time)
        lora_mems.append(mem)

        # === LOG STEP ===
        logger.log_step(epoch * len(inputs) + step, **{
            "loss": loss.item(),
            "forward_time_ms": fw_time * 1000,
            "backward_time_ms": bw_time * 1000,
            "memory_mb": mem,
        })

        epoch_loss.append(loss.item())
    print(f"Epoch {epoch+1}: Loss = {np.mean(epoch_loss):.4f}")

# Итоговая статистика LoRA
logger.log_summary("mean_forward_ms", np.mean(lora_fw_times) * 1000)
logger.log_summary("mean_backward_ms", np.mean(lora_bw_times) * 1000)
logger.log_summary("max_memory_mb", np.max(lora_mems))

logger.finish()
