import argparse
import json
import traceback

import torch
import re
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath("../"))
import random
from prompt.prompt_template import *


# Format chat prompt based on model family
def format_chat(model_id, system_msg,  user_prompt):
    full_prompt = f"{user_prompt}"

    if "mistral" in model_id.lower() or "llama" in model_id.lower() or "sarashina" in model_id.lower():
        return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n{full_prompt} [/INST]"
    elif "qwen" in model_id.lower():
        return f"<|im_start|>system\n{system_msg}\n<|im_start|>user\n{full_prompt}\n<|im_end|>"
    elif "gemma" in model_id.lower():
        return f"<start_of_turn>user\n{system_msg}\n{full_prompt}<end_of_turn><start_of_turn>model\n"
    else:
        return f"{system_msg}\n{full_prompt}"

# Check if string is valid XML
def is_valid_xml(text):
    try:
        ET.fromstring(text)
        return True
    except ET.ParseError:
        return False

# Try to extract XML content from messy string
def extract_possible_xml(text):
    matches = re.findall(r"<\s*([a-zA-Z0-9:_-]+)[^>]*>.*?</\s*\1\s*>", text, flags=re.DOTALL)
    if matches:
        for tag in matches:
            candidate = re.search(rf"<{tag}[^>]*>.*?</{tag}>", text, flags=re.DOTALL)
            if candidate and is_valid_xml(candidate.group(0)):
                return candidate.group(0)
    return None

# Generate response with retry and validation
def generate_response(
    model, tokenizer, model_id,
    system_msg,  user_prompt,
    max_attempts=3, max_input_tokens=None
):
    for attempt in range(max_attempts):
        prompt = format_chat(model_id, system_msg,  user_prompt)

        # Tokenize prompt only to get its length
        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens or tokenizer.model_max_length
        ).to(model.device)

        prompt_len = prompt_inputs.input_ids.shape[-1]

        with torch.no_grad():
            output = model.generate(
                **prompt_inputs,
                # max_input_tokens=max_input_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )

        # Trim the output to only include new tokens after the prompt
        generated_ids = output[0][prompt_len:]
        # , skip_special_tokens = True
        # print('length {}'.format(prompt_len))
        answer = tokenizer.decode(generated_ids, skip_special_tokens = True).strip()

        if is_valid_xml(answer):
            return answer

        filtered = extract_possible_xml(answer)
        if filtered:
            return filtered
    # print('answer {}'.format(answer))
    # input('bbb')
    return answer  # fallback after max_attempts


# Main processing loop
def main(model_id, input_train_path, input_test_path,context_path, output_path, max_attempts=3, max_input_tokens=None):
    print(f"🔄 Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    json_context=json.load(open(context_path,'rb'))


    # with open(input_path, "r") as infile:
    #     input_data = [json.loads(line) for line in infile]
    input_train_data = json.load(open(input_train_path, 'rb'))[:100]
    input_test_data=json.load(open(input_test_path,'rb'))[:10]
    output_data = []



    index_item=-1
    for item in tqdm(input_test_data, desc="🧠 Generating responses"):
        index_item+=1
        system_prompt = SamplePromptTemplate.system_template
        # context_prompt = 'empty'
        # user_prompt = 'What is capital of Vietnam?'
        index_train, item_train = random.choice(list(enumerate(input_train_data)))
        str_example="""2.1. Example 1: \n- Input: \n'''\n{}\n'''\n- Expected Output: \n'''\n{}\n'''\n""".format(item_train['input'],item_train['output'])

        user_prompt=SamplePromptTemplate.prompt_template.replace('{input_XML}',item.get('input')).replace('{examples}',str_example)
        # print('user input:\n{}\nEnd'.format(user_prompt))
        try:
            answer = generate_response(
                model, tokenizer, model_id,
                system_prompt, user_prompt,
                max_input_tokens=max_input_tokens
            )
            # print('valid answer {}'.format(answer))
            # input('aaa')
        except Exception as e:
            answer = f"Error: {str(e)}"
            print(f"⚠️ Error on input: {e}")
            traceback.print_exc()

        item_with_answer = dict(item)
        item_with_answer["user_prompt"] = user_prompt
        item_with_answer["index_sample"] = index_train
        item_with_answer["answer"] = answer
        output_data.append(item_with_answer)
        print('get answer {}'.format(index_item))

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as outfile:
        json.dump(output_data, outfile, indent=2)

    print(f"\n✅ Output written to: {output_path}")

# CLI parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate XML-valid LLM responses.")
    parser.add_argument("--model_id", type=str, required=True, help="Model path or Hugging Face model ID")
    parser.add_argument("--input_train", type=str, required=True, help="Path to input JSONL train file")
    parser.add_argument("--input_test", type=str, required=True, help="Path to input JSONL test file")
    parser.add_argument("--context_path", type=str, required=True, help="Path to contextJSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--max_attempts", type=int, default=3, help="Max attempts to retry for valid XML")
    parser.add_argument("--max_input_tokens", type=int, default=None, help="Maximum input prompt token length")
    args = parser.parse_args()

    main(
        args.model_id,
        args.input_train,
        args.input_test,
        args.context_path,
        args.output,
        max_attempts=args.max_attempts,
        max_input_tokens=args.max_input_tokens
    )
