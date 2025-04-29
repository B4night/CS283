import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = OpenAI()


def distinct_n(corpus, n):
    total_ngrams = 0
    unique_ngrams = set()
    for sentence in corpus:
        tokens = sentence.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams = list(ngrams)
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


def evaluate_distinct(stories):
    distinct1 = distinct_n(stories, 1)
    distinct2 = distinct_n(stories, 2)
    print(f"Distinct-1: {distinct1:.4f}")
    print(f"Distinct-2: {distinct2:.4f}")


def calculate_perplexity(stories, model_name="gpt2-xl"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()

    total_loss = 0
    total_length = 0

    for story in stories:
        inputs = tokenizer(story, return_tensors="pt",
                           truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        total_loss += loss.item() * inputs["input_ids"].size(1)
        total_length += inputs["input_ids"].size(1)

    if total_length == 0:
        return float('inf')

    avg_loss = total_loss / total_length
    ppl = math.exp(avg_loss)
    print(f"Perplexity: {ppl:.2f}")


def get_score(prompt: str) -> str:
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "You are an expert story critic and evaluator.\n\nYour task is to evaluate the following story and provide a comprehensive score from 0 to 100, based on multiple dimensions of storytelling quality.\n\nSpecifically, assess the story according to the following criteria:\n\n1. **Structure and Coherence (25 points)**  \n   - Does the story have a clear beginning, development, and ending?\n   - Are the plot points logically connected without abrupt jumps or inconsistencies?\n\n2. **Creativity and Originality (20 points)**  \n   - Is the story conceptually novel or imaginative?\n   - Does it avoid clichés and offer fresh ideas or perspectives?\n\n3. **Language Fluency and Style (20 points)**  \n   - Is the language smooth, grammatically correct, and pleasant to read?\n   - Does the writing style suit the tone and genre of the story?\n\n4. **Emotional Impact (15 points)**  \n   - Does the story evoke emotions (e.g., excitement, sadness, joy) in the reader?\n   - Is the emotional arc believable and well-executed?\n\n5. **Character Development and Consistency (10 points)**  \n   - Are the characters distinct, believable, and consistent in their behavior?\n   - Do the characters' motivations make sense within the story?\n\n6. **World-Building and Setting (10 points)**  \n   - Is the story world vivid, detailed, and immersive?\n   - Does the setting enhance the story rather than distract from it?\n\n---\n\n**Scoring Instructions:**\n- Assign a score for each criterion individually according to the maximum points listed.\n- Then sum the individual scores to get a final score out of 100.\n- Be strict and fair. A perfect 100 should be extremely rare and only given to exceptionally outstanding stories.\n- If any dimension is missing or poorly executed, deduct points accordingly.\n\n**Output format:**\n```markdown\n# Story Evaluation\n\n## 1. Structure and Coherence (out of 25): \nScore: X/25\n\n## 2. Creativity and Originality (out of 20):\nScore: X/20\n\n## 3. Language Fluency and Style (out of 20):\nScore: X/20\n\n## 4. Emotional Impact (out of 15):\nScore: X/15\n\n## 5. Character Development and Consistency (out of 10):\nScore: X/10\n\n## 6. World-Building and Setting (out of 10):\nScore: X/10\n\n## Final Score:\nXX/100\n\n## Overall Feedback:\n(Brief overall comment about the story's strengths and areas for improvement.)\n"
            },
            {
                "role": "user",
                "content": f'Story: {prompt}'
            }
        ]
    )
    res = completion.choices[0].message.content
    return res


if __name__ == "__main__":
    dirs = ['ds-8b', 'llama', 'qwen']

    for dire in dirs:
        sub_dirs = os.listdir(f'{dire}')

        baseline_dir = sub_dirs[0] if len(
            sub_dirs[0]) > len(sub_dirs[1]) else sub_dirs[1]
        new_dir = sub_dirs[1] if baseline_dir == sub_dirs[0] else sub_dirs[0]

        baseline_dir = f'./{dire}/{baseline_dir}'
        new_dir = f'./{dire}/{new_dir}'

        baseline_stories = []
        for file in os.listdir(f'{baseline_dir}'):
            with open(f'{baseline_dir}/{file}', 'r') as f:
                story = f.read()
                if story == "":
                    continue
                response = get_score(story)
                baseline_stories.append(story)
            with open(f'{baseline_dir}/{file}.score', 'w') as f:
                f.write(response)
            # break

        print(f'Metrics for {baseline_dir}')
        evaluate_distinct(baseline_stories)
        calculate_perplexity(baseline_stories)

        new_stories = []
        for file in os.listdir(f'{new_dir}'):
            with open(f'{new_dir}/{file}', 'r') as f:
                story = f.read()
                if story == "":
                    continue
                response = get_score(story)
                new_stories.append(story)
            with open(f'{new_dir}/{file}.score', 'w') as f:
                f.write(response)
            # break

        print(f'Metrics for {new_dir}')
        evaluate_distinct(new_stories)
        calculate_perplexity(new_stories)
