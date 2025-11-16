import wandb


class WandbLogger:
    def __init__(self, project="lora_demo", run_name=None):
        self.wandb = wandb
        self.run = wandb.init(
            project=project, 
            name=run_name,
            monitor_gym=True,
            save_code=True,
            settings=wandb.Settings(code_dir="..")
        )

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

    def log_model_artifact(self, model_dir, artifact_name="pythia_lora_sft_ref", description="SFT+LoRA reference model"):
        artifact = self.wandb.Artifact(
            name=artifact_name,
            type="model",
            description=description
        )
        artifact.add_dir(model_dir)
        self.run.log_artifact(artifact)

