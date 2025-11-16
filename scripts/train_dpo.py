import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger.logger import WandbLogger
from lora.module import LoRALayer
from dpo.dataset import DPOPairDataset
from dpo.loss import dpo_loss

# --- PARAMETERS ---
MODEL_NAME = "EleutherAI/pythia-1.4b"
CHECKPOINT = "data/pythia_lora_sft_ref"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 1
BETA = 0.1
LORA_R = 8
LORA_ALPHA = 16
RUN_NAME = "dpo_demo"

def patch_lora(model, r, alpha, device):
    for layer in model.gpt_neox.layers:
        orig = layer.attention.query_key_value
        lora_layer = LoRALayer(orig, r=r, alpha=alpha).to(device)
        layer.attention.query_key_value = lora_layer

def get_lora_params(model):
    params = []
    for layer in model.gpt_neox.layers:
        params.extend(list(layer.attention.query_key_value.parameters()))
    return params


# --- Wandb logger ---
logger = WandbLogger(project="nlp_dpo", run_name=RUN_NAME)

artifact = logger.run.use_artifact('sofia-nelipovich-hse-university/nlp_dpo/pythia_lora_sft_ref:latest', type='model')
artifact_dir = artifact.download()  # путь к скачанному файлу/директории

print("Downloaded to:", artifact_dir)

# --- MODELS ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# ---- Reference (frozen) model ----
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
patch_lora(ref_model, LORA_R, LORA_ALPHA, DEVICE)
ref_sd = load_file(f"{artifact_dir}/model.safetensors")
ref_model.load_state_dict(ref_sd, strict=False)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
 
# ---- Main model for DPO fine-tuning ----
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
patch_lora(model, LORA_R, LORA_ALPHA, DEVICE)
sd = load_file(f"{artifact_dir}/model.safetensors")
model.load_state_dict(sd, strict=False)
for p in model.parameters():
    p.requires_grad = False
for p in get_lora_params(model):
    p.requires_grad = True

# --- DATLOADER ---
ds_hh = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
dataset = DPOPairDataset(ds_hh, tokenizer, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(get_lora_params(model), lr=5e-5)

# --- WANDB CONFIG ---
logger.log_config({
    "model_name": MODEL_NAME,
    "beta": BETA,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "n_samples": len(ds_hh)
})

logger.watch(model)

# --- DPO TRAIN LOOP ---
model.train()
ref_model.to('cpu')
step_count = 0
for epoch in range(EPOCHS):
    losses = []
    for batch in tqdm(dataloader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss = dpo_loss(model, ref_model, batch, tokenizer, BETA)
        loss.backward()
        optimizer.step()
        logger.log_step(step_count, **{"dpo_loss": loss.item()})
        losses.append(loss.item())
        del loss
        torch.cuda.empty_cache()
        step_count += 1
    print(f"Epoch {epoch+1}: Mean dpo loss {np.mean(losses):.4f}")

# --- EVAL GENERATION ---
def generate(model, prompt, max_new_tokens=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.to(DEVICE)
    gen_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text

import random
eval_indices = random.sample(range(len(dataset)), 5)
print("\n=== ANSWERS COMPARISON ===")
for idx in eval_indices:
    ex = dataset[idx]
    prompt = ex["prompt"]
    true_chosen = ex["chosen"]
    true_rejected = ex["rejected"]
    sft_out = generate(ref_model, prompt)
    dpo_out = generate(model, prompt)
    print("-"*40)
    print("PROMPT:", prompt)
    print("DPO_GEN:", dpo_out.replace(prompt, '').strip())
    print("SFT_GEN:", sft_out.replace(prompt, '').strip())
    print("TRUE_CHOSEN:", true_chosen)
    print("TRUE_REJECTED:", true_rejected)
