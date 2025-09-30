import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# 1. Load your dataset
dataset = load_dataset("json", data_files="path/to/your_data.jsonl")["train"]

# 2. Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # or any open LLM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Important for some models

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# 3. Prepare for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Tokenize the dataset
def tokenize_function(example):
    input_text = example["input"]  # Assuming 'input' and 'output' fields
    target_text = example["output"]
    full_text = f"### Input:\n{input_text}\n\n### Response:\n{target_text}"
    
    model_inputs = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./llm-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    save_steps=100,
    logging_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=2,
    report_to="none",
    run_name="llm-finetune",
)

# 6. Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=None,
)

# 7. Start training
trainer.train()

# 8. Save the LoRA adapter
model.save_pretrained("./llm-finetuned-lora")
tokenizer.save_pretrained("./llm-finetuned-lora")
