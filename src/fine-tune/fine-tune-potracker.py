import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import torch
import difflib
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# XML similarity metric functions
# -------------------------------
def extract_text_and_tags(xml_string):
    root = etree.fromstring(xml_string)
    texts = []
    tags = []
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

# -------------------------------
# Load sample data
# -------------------------------
df = pd.read_csv("sample_data.csv")  # columns: input, target (XML)
dataset = Dataset.from_pandas(df)

# -------------------------------
# Tokenizer and model
# -------------------------------
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# -------------------------------
# Tokenization
# -------------------------------
def tokenize(example):
    full_text = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['target']}"
    inputs = tokenizer(full_text, max_length=512, truncation=True, padding="max_length")
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(tokenize)

# -------------------------------
# Custom compute_metrics
# -------------------------------
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scores = [combined_similarity(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    return {
        "text_similarity": sum(s["text_similarity"] for s in scores) / len(scores),
        "tag_similarity": sum(s["tag_similarity"] for s in scores) / len(scores),
        "combined_similarity": sum(s["combined_similarity"] for s in scores) / len(scores),
    }

# -------------------------------
# Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./xml-llm",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    evaluation_strategy="epoch",
    predict_with_generate=True,
    report_to="none",
    fp16=torch.cuda.is_available()
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # use same set for demonstration
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

# -------------------------------
# Train and Save
# -------------------------------
trainer.train()
trainer.save_model("./xml-llm")
