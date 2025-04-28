import os
import logging
import argparse
import torch
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

client = OpenAI()
sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# Dictionary mapping model types to their HuggingFace model IDs
MODEL_MAPPINGS = {
    "gpt": "gpt-4o",  # This will still use OpenAI's API
    "ds-8b": "/ibex/user/feic/pjs/model/DeepSeek-R1-Distill-Llama-8B",
    "llama-3.1-8b": "/ibex/user/feic/pjs/model/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "/ibex/user/feic/pjs/model/Qwen2.5-7B-Instruct-1M"
    # Add more models as needed
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate stories using different LLM models')
    parser.add_argument('--model', type=str, default='gpt', choices=MODEL_MAPPINGS.keys(),
                        help='Model type to use for story generation (e.g., gpt, llama, deepseek, qwen)')
    return parser.parse_args()


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


def generate_sparql_query(prompt: str) -> str:
    # Always use GPT-4o for SPARQL query generation
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "Generate SPARQL query based on the prompt to get information from DBpedia. Only output the sparql query."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    res = completion.choices[0].message.content
    res = res.replace("```sparql", "").replace("```", "")
    res = res + "\nLIMIT 10"
    logger.info(f"Generated SPARQL query: {res}")
    return res


def get_info_from_dbpedia(prompt: str) -> dict:
    success = False
    ans = {}
    fail_count = 0

    while not success and fail_count < 3:
        fail_count += 1
        try:
            query = generate_sparql_query(prompt)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            success = True
            if "results" in results and "bindings" in results["results"]:
                for result in results["results"]["bindings"]:
                    for var in result:
                        value = result[var].get('value', '')
                        if value.startswith("http://") or value.startswith("https://"):
                            value = value.split("/")[-1]
                        ans[var] = value

            if not ans:
                success = False
        except Exception as e:
            print(f"Query failed, retrying... Error: {e}")

    return ans


def get_subgraph_from_dbpedia(prompt: str) -> list:
    success = False
    ans = []

    while not success:
        try:
            query = generate_sparql_query(prompt)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            success = True
            if "results" in results and "bindings" in results["results"]:
                for result in results["results"]["bindings"]:
                    for var in result:
                        value = result[var].get('value', '')
                        if value.startswith("http://") or value.startswith("https://"):
                            value = value.split("/")[-1]
                        ans.append(f"{var}: {value}")

        except Exception as e:
            print(f"Query failed, retrying... Error: {e}")

    return ans


def get_story(model, tokenizer, prompt: str, model_type: str, num_parts: int = 3) -> str:
    appendix_info = get_subgraph_from_dbpedia(prompt)
    parts = generate_story_parts(
        prompt, appendix_info, model_type, model, tokenizer, num_parts)
    full_story = combine_story_parts(
        parts, prompt, model_type, model, tokenizer)

    return full_story


def generate_story_parts(prompt: str, appendix_info: list, model_type: str, model=None, tokenizer=None, num_parts: int = 3) -> list:
    chunk_size = max(1, len(appendix_info) // num_parts)
    parts = []

    for i in range(num_parts):
        chunk = appendix_info[i * chunk_size: (i + 1) * chunk_size]
        chunk_text = "\n".join(chunk)
        sub_prompt = f"{prompt} (Part {i+1})"

        system_prompt = "Generate a part of a story based on the prompt and additional information. Only output the story content."

        if model_type == "gpt":
            # Use OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": sub_prompt
                },
                {
                    "role": "user",
                    "content": chunk_text
                }
            ]

            completion = client.chat.completions.create(
                model=get_model_name(model_type),
                messages=messages
            )
            res = completion.choices[0].message.content.strip().replace(
                "```", "")
        else:
            # Use Hugging Face model
            res = generate_text_with_model(
                model,
                tokenizer,
                system_prompt,
                sub_prompt,
                chunk_text
            )

        # Log first 100 chars
        logger.info(f"Generated part {i+1} using {model_type}: {res[:100]}...")
        parts.append(res)

    return parts


def combine_story_parts(parts: list, prompt: str, model_type: str, model=None, tokenizer=None) -> str:
    combined_input = "\n\n".join(
        [f"Part {i+1}:\n{part}" for i, part in enumerate(parts)])

    system_prompt = "Combine the following story parts into one coherent story. Only output the story."
    user_prompt = f"Original prompt: {prompt}\n\n{combined_input}"

    if model_type == "gpt":
        # Use OpenAI API
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        completion = client.chat.completions.create(
            model=get_model_name(model_type),
            messages=messages
        )
        res = completion.choices[0].message.content.strip().replace("```", "")
    else:
        # Use Hugging Face model
        res = generate_text_with_model(
            model,
            tokenizer,
            system_prompt,
            user_prompt
        )

    return res


if __name__ == "__main__":
    args = parse_arguments()
    model_type = args.model
    logger.info(f"Using model type: {model_type} for story generation")

    with open(f'../prompts/events', 'r') as f:
        prompts = f.read().split("\n")

    # make directory: ../baseline/{model_type}_our_method
    os.mkdir(f'../baseline/{model_type}_our_method', exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_type)

    cnt = 0
    for prompt in prompts:
        question = f'Generate the story about the event: {prompt}'
        story = get_story(model, tokenizer, question, model_type, num_parts=3)
        with open(f'../baseline/{model_type}_our_method/{cnt}.txt', "w") as f:
            f.write(story)
        cnt += 1
