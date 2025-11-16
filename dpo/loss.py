import torch
import torch.nn.functional as F

def dpo_loss(model, ref_model, batch, tokenizer, beta=0.1):
    prompt = batch['prompt']
    chosen = batch['chosen']
    rejected = batch['rejected']

    device = next(model.parameters()).device  # автоматически определяем cuda/cpu

    def tokenize_batch(prompts, responses):
        texts = [f"{p} {r}" for p, r in zip(prompts, responses)]
        tokens = tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        return tokens

    chosen_tokens = tokenize_batch(prompt, chosen)
    rejected_tokens = tokenize_batch(prompt, rejected)

    def mean_log_prob(model, tokens, device='cuda'):
        for k in tokens:
            tokens[k] = tokens[k].to(device)
        if device == 'cuda':
            logits = model(**tokens).logits
        else:
            with torch.no_grad():
                logits = model(**tokens).logits
        input_ids = tokens['input_ids']
        # Отбираем пер-токен логиты для ответа
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = F.cross_entropy(
            shift_logits.transpose(1,2), shift_labels, reduction="none"
        )
        # Cуммируем по всем токенам в ответе (или усредняем)
        # В итоге получаем log P(y|x) для каждой пары
        log_prob = -loss.sum(dim=1)
        return log_prob

    # Считаем log_prob для всех комбинаций
    chosen_logprob_model = mean_log_prob(model, chosen_tokens)
    rejected_logprob_model = mean_log_prob(model, rejected_tokens)
    chosen_logprob_ref = mean_log_prob(ref_model, chosen_tokens, device='cpu')
    rejected_logprob_ref = mean_log_prob(ref_model, rejected_tokens, device='cpu')

    # Считаем формулу DPO Loss
    delta_model = chosen_logprob_model - rejected_logprob_model
    delta_ref = chosen_logprob_ref - rejected_logprob_ref
    dpo_term = beta * (delta_model - delta_ref)
    loss = -torch.log(torch.sigmoid(dpo_term)).mean()
    return loss
