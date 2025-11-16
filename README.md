# nlp_dpo

# Direct Preference Optimization (DPO) Fine-Tuning with LoRA

Этот проект реализует обучение языковой модели с помощью **DPO** (Direct Preference Optimization), согласно [оригинальной статье](https://arxiv.org/pdf/2305.18290), с использованием LoRA-адаптеров для эффективного дообучения на ограниченных GPU-ресурсах.

## Описание

- SFT и DPO реализованы для моделей семейства GPT/EleutherAI (или других совместимых с HuggingFace).
- В качестве reference используется SFT-модель, для DPO дообучаются только LoRA-слои.
- Поддерживается сравнение с full fine-tuning и независимое логгирование метрик (loss, delta_model, delta_ref, память).


## Настройка данных

- Для SFT: датасет должен содержать пары "prompt" — "response".
- Для DPO: пары "prompt", "chosen", "rejected", причём желательно, чтобы между SFT и DPO датасеты **не пересекались**.

## Логгирование и мониторинг

Все метрики обучения автоматизировано логгируются в Wandb.  
**Ссылка на отчёт в Wandb:**  
[wandb report](https://api.wandb.ai/links/sofia-nelipovich-hse-university/ymzvuckg)

## Ключевые параметры

- LoRA: rank (LORA_R), alpha (LORA_ALPHA)
- DPO: beta коэффициент, learning rate, batch size, max_length
