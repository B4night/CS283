import logging
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = OpenAI()


def generate_story(prompt: str) -> str:
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "Generate a story about the event given in the prompt."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    res = completion.choices[0].message.content
    return res


def main():
    with open("../prompts/events", "r") as f:
        prompts = f.read().split("\n")

    cnt = 0
    for prompt in prompts:
        question = f'Generate the story about the event: {prompt}'
        story = generate_story(question)
        with open(f'./pure_gpt_4o/{cnt}.txt', "w") as f:
            f.write(story)
        cnt += 1


if __name__ == "__main__":
    main()
