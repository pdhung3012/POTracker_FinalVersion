import torch
import difflib
from lxml import etree
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import DPOTrainer


# ======================== XML Similarity Logic ========================
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


def combined_similarity(xml1, xml2, alpha=0.7):
    text1, tags1 = extract_text_and_tags(xml1)
    text2, tags2 = extract_text_and_tags(xml2)
    sim_text = text_similarity(text1, text2)
    sim_tags = tag_similarity(tags1, tags2)
    return alpha * sim_text + (1 - alpha) * sim_tags


# ======================== Load Model with LoRA ========================
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # adapt to model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)


# ======================== Prepare Sample Dataset ========================
data = {
    "prompt": [
        "<root><title>Hello</title><body>What is XML?</body></root>",
        "<doc><header>Rain</header><content>Causes of rain</content></doc>"
    ],
    "chosen": [
        "<root><title>Hello</title><body>XML stands for eXtensible Markup Language.</body></root>",
        "<doc><header>Rain</header><content>Rain is caused by water vapor condensing.</content></doc>"
    ],
    "rejected": [
        "<root><title>Hi</title><body>XML is a file format like JPG.</body></root>",
        "<doc><header>Rain</header><content>Rain is when the sky is crying.</content></doc>"
    ]
}
dataset = Dataset.from_dict(data)


def preprocess(example):
    prompt = example["prompt"]
    example["prompt_texts"] = prompt
    example["chosen_texts"] = example["chosen"]
    example["rejected_texts"] = example["rejected"]
    return example


dataset = dataset.map(preprocess)


# ======================== Custom DPOTrainer with XML Loss ========================
class XMLDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_texts = inputs["chosen_texts"]
        rejected_texts = inputs["rejected_texts"]
        reference_xmls = inputs["prompt_texts"]
        alpha = 0.7

        # Compute reward difference using XML similarity
        rewards_chosen = []
        rewards_rejected = []

        for ref, ch, rj in zip(reference_xmls, chosen_texts, rejected_texts):
            rewards_chosen.append(combined_similarity(ref, ch, alpha))
            rewards_rejected.append(combined_similarity(ref, rj, alpha))

        rewards_chosen = torch.tensor(rewards_chosen).to(model.device)
        rewards_rejected = torch.tensor(rewards_rejected).to(model.device)

        logits_diff = rewards_chosen - rewards_rejected
        dpo_loss = -torch.nn.functional.logsigmoid(logits_diff).mean()

        return (dpo_loss, {"loss": dpo_loss}) if return_outputs else dpo_loss


# ======================== Training Setup ========================
training_args = TrainingArguments(
    output_dir="./mistral-xml-dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="no",
    remove_unused_columns=False,
    bf16=True
)

trainer = XMLDPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
