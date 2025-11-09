import wandb


class WandbLogger:
    def __init__(self, project="lora_demo", run_name=None):
        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name)

    def log_config(self, params):
        self.wandb.config.update(params)

    def log_step(self, step, **kwargs):
        metrics = dict(kwargs)
        metrics["step"] = step
        self.wandb.log(metrics)

    def log_sample_generation(self, prompt, generation, ref=None, step=None):
        table = self.wandb.Table(columns=["prompt", "generation", "reference"])
        table.add_data(prompt, generation, ref)
        self.wandb.log({"sample_generation": table, "step": step if step is not None else 0})

    def watch(self, model):
        # Логирует градиенты и параметры
        self.wandb.watch(model, log="all")

    def log_summary(self, key, value):
        wandb.summary[key] = value

    def finish(self):
        self.wandb.finish()
