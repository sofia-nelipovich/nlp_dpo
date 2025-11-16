from torch.utils.data import Dataset


def split_prompt_response(sample):
    idx = sample.rfind("Assistant:")
    prompt = sample[:idx + len("Assistant:")].strip()
    response = sample[idx + len("Assistant:"):].strip()
    return prompt, response

class DPOPairDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.data = []
        for ex in hf_dataset:
            if "chosen" in ex and "rejected" in ex and ("Assistant:" in ex["chosen"]) and ("Assistant:" in ex["rejected"]):
                prompt, chosen_response = split_prompt_response(ex['chosen'])
                _, rejected_response = split_prompt_response(ex['rejected'])
                self.data.append({
                    'prompt': prompt,
                    'chosen': chosen_response,
                    'rejected': rejected_response
                })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
