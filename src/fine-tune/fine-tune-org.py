import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch

# Load sample data
df = pd.read_csv("sample_data.csv")
df["text"] = "### Instruction:\n" + df["prompt"] + "\n\n### Response:\n" + df["response"]
dataset = Dataset.from_pandas(df[["text"]])

# Load tokenizer and model
model_name = "tiiuae/falcon-rw-1b"  # You can also use "gpt2", "EleutherAI/pythia-70m", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # ensure pad token exists
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# Define training args
training_args = TrainingArguments(
    output_dir="./finetuned-llm",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save final model
trainer.save_model("./finetuned-llm")
