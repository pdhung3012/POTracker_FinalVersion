import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import difflib
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# XML evaluation metric
# -----------------------
def extract_text_and_tags(xml_string):
    root = etree.fromstring(xml_string)
    texts, tags = [], []
    for elem in root.iter():
        tags.append(elem.tag)
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
    return ' '.join(texts), ' '.join(tags)

def text_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf = vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf[0], tfidf[1])[0, 0]

def tag_similarity(tags1, tags2):
    return difflib.SequenceMatcher(None, tags1, tags2).ratio()

def combined_similarity(xml1, xml2, alpha=0.5):
    try:
        text1, tags1 = extract_text_and_tags(xml1)
        text2, tags2 = extract_text_and_tags(xml2)
        sim_text = text_similarity(text1, text2)
        sim_tags = tag_similarity(tags1, tags2)
        return {
            'text_similarity': sim_text,
            'tag_similarity': sim_tags,
            'combined_similarity': alpha * sim_text + (1 - alpha) * sim_tags
        }
    except Exception:
        return {
            'text_similarity': 0.0,
            'tag_similarity': 0.0,
            'combined_similarity': 0.0
        }

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scores = [combined_similarity(p, l) for p, l in zip(decoded_preds, decoded_labels)]
    return {
        "text_similarity": sum(s["text_similarity"] for s in scores) / len(scores),
        "tag_similarity": sum(s["tag_similarity"] for s in scores) / len(scores),
        "combined_similarity": sum(s["combined_similarity"] for s in scores) / len(scores),
    }

# -----------------------
# Load sample data
# -----------------------
df = pd.read_csv("sample_data.csv")  # columns: input, target (XML)
def format_prompt(example):
    return f"### Instruction:\n{example['input']}\n\n### Response:\n{example['target']}"
df["prompt_formatted"] = df.apply(format_prompt, axis=1)
dataset = Dataset.from_pandas(df[["prompt_formatted", "target"]])

# -----------------------
# Model + Tokenizer
# -----------------------
model_name = "tiiuae/falcon-rw-1b"  # or "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit model loading with bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model = prepare_model_for_kbit_training(base_model)

# -----------------------
# Apply LoRA
# -----------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust per model
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# -----------------------
# Tokenization
# -----------------------
def tokenize_supervised(example):
    prompt = example["prompt_formatted"]
    response = example["target"]
    full_text = prompt + response

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")

    labels = full_ids["input_ids"].copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    full_ids["labels"] = labels
    return full_ids

tokenized_dataset = dataset.map(tokenize_supervised, remove_columns=dataset.column_names)

# -----------------------
# TrainingArguments
# -----------------------
training_args = TrainingArguments(
    output_dir="./sft-lora-xml-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    learning_rate=2e-4,
    bf16=True,
    report_to="none",
    evaluation_strategy="epoch",
    predict_with_generate=True,
)

# -----------------------
# Trainer
# -----------------------
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

# -----------------------
# Train
# -----------------------
trainer.train()
model.save_pretrained("./sft-lora-xml-model")
