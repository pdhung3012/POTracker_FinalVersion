from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig

# 1. Sample preference data
data = {
    "prompt": ["Why is the sky blue?", "What causes rain?"],
    "chosen": ["The sky is blue due to Rayleigh scattering.", "Rain is caused by condensation of water vapor."],
    "rejected": ["Because blue is a nice color.", "Rain comes from sadness in the sky."]
}

dataset = Dataset.from_dict(data)

# 2. Load tokenizer and base model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Define DPO training configuration
dpo_config = DPOConfig(
    beta=0.1,  # strength of the preference loss
    max_length=512,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1
)

# 4. Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./dpo-checkpoints",
    per_device_train_batch_size=dpo_config.per_device_train_batch_size,
    num_train_epochs=dpo_config.num_train_epochs,
    logging_steps=dpo_config.logging_steps,
    remove_unused_columns=False,
    logging_dir="./logs",
    save_strategy="no"
)

# 5. Initialize and train DPO model
trainer = DPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    beta=dpo_config.beta
)

trainer.train()
