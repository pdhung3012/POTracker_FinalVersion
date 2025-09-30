import json
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_json_examples(json_path):
    with open(json_path, 'r') as f:
        examples = json.load(f)
    formatted_examples = ""
    for ex in examples:
        formatted_examples += f"Input: {ex['input']}\nOutput:\n{ex['output']}\n\n"
    return formatted_examples

def load_pdf_context(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def build_prompt(examples_text, pdf_text, query):
    prompt = "You are an expert XML generator. Given a query, you must return a valid XML based on examples and context.\n\n"
    if pdf_text:
        prompt += f"### Context from PDF:\n{pdf_text}\n\n"
    prompt += f"### Examples:\n{examples_text}\n"
    prompt += f"### Now, generate XML for the following query:\n{query}\nOutput:\n"
    return prompt

def generate_xml(model_name, prompt, max_new_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated.split("Output:", 1)[-1].strip()

# === User Configuration ===
json_path = "examples.json"    # JSON file with [{'input': ..., 'output': ...}, ...]
pdf_path = "context.pdf"       # Optional: PDF to provide context
user_query = "Generate XML for a list of products with name, price, and availability"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# === Run ===
examples_text = load_json_examples(json_path)
pdf_text = load_pdf_context(pdf_path)
prompt = build_prompt(examples_text, pdf_text, user_query)
generated_xml = generate_xml(model_name, prompt)

print("=== Generated XML ===\n")
print(generated_xml)
