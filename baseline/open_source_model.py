import os
import logging
import argparse
import torch
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(find_dotenv())

client = OpenAI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_MAPPINGS = {
    "gpt": "gpt-4o",  # This will still use OpenAI's API
    "ds-8b": "/ibex/user/feic/pjs/model/DeepSeek-R1-Distill-Llama-8B",
    "llama-3.1-8b": "/ibex/user/feic/pjs/model/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "/ibex/user/feic/pjs/model/Qwen2.5-7B-Instruct-1M"
    # Add more models as needed
}


def get_model_name(model_type):
    return MODEL_MAPPINGS.get(model_type, MODEL_MAPPINGS["gpt"])


def load_model_and_tokenizer(model_type):
    """Load model and tokenizer from HuggingFace."""
    if model_type == "gpt":
        return None, None  # OpenAI API will be used instead

    model_name = get_model_name(model_type)

    logger.info(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate stories using different LLM models')
    parser.add_argument('--model', type=str, default='gpt', choices=MODEL_MAPPINGS.keys(),
                        help='Model type to use for story generation (e.g., gpt, llama, deepseek, qwen)')
    return parser.parse_args()


def generate_text_with_model(model, tokenizer, system_prompt, user_prompt, additional_content=None):
    """Generate text using HuggingFace models."""
    if additional_content:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n{additional_content}\n<|assistant|>"
    else:
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract just the assistant's response based on the prompt format
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    return response


def generate_story(model, tokenizer, prompt: str) -> str:
    system_prompt = "Generate a story about the event given in the prompt."
    story = generate_text_with_model(model, tokenizer, system_prompt, prompt)

    return story


def main():
    args = parse_arguments()
    model_type = args.model

    cnt = 0

    model, tokenizer = load_model_and_tokenizer(model_type)

    with open("../prompts/events", "r") as f:
        prompts = f.read().split("\n")

    os.makedirs(f'./{model_type}', exist_ok=True)

    for prompt in prompts:
        question = f'Generate the story about the event: {prompt}'
        story = generate_story(model, tokenizer, question)
        with open(f'./{model_type}/{cnt}.txt', "w") as f:
            f.write(story)
        cnt += 1


if __name__ == "__main__":
    main()
