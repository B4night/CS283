import logging
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

client = OpenAI()
sparql = SPARQLWrapper("http://dbpedia.org/sparql")


def generate_sparql_query(prompt: str) -> str:
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


def generate_story_parts(prompt: str, appendix_info: list, num_parts: int = 3) -> list:
    # Split appendix_info into `num_parts` chunks
    chunk_size = max(1, len(appendix_info) // num_parts)
    parts = []

    for i in range(num_parts):
        chunk = appendix_info[i * chunk_size: (i + 1) * chunk_size]
        chunk_text = "\n".join(chunk)
        sub_prompt = f"{prompt} (Part {i+1})"

        messages = [
            {
                "role": "system",
                "content": "Generate a part of a story based on the prompt and additional information. Only output the story content."
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
            model='gpt-4o',
            messages=messages
        )
        res = completion.choices[0].message.content.strip().replace("```", "")
        logger.info(f"Generated part {i+1}: {res}")
        parts.append(res)

    return parts


def get_story(prompt: str, num_parts: int = 3) -> str:
    appendix_info = get_subgraph_from_dbpedia(prompt)
    parts = generate_story_parts(prompt, appendix_info, num_parts)
    full_story = combine_story_parts(parts, prompt)
    return full_story


def combine_story_parts(parts: list, prompt: str) -> str:
    combined_input = "\n\n".join(
        [f"Part {i+1}:\n{part}" for i, part in enumerate(parts)])
    messages = [
        {
            "role": "system",
            "content": "Combine the following story parts into one coherent story. Only output the story."
        },
        {
            "role": "user",
            "content": f"Original prompt: {prompt}"
        },
        {
            "role": "user",
            "content": combined_input
        }
    ]
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=messages
    )
    res = completion.choices[0].message.content.strip().replace("```", "")
    return res


if __name__ == "__main__":
    history = 'France revolution'
    prompt = f"I am interested in the history about {history}"
    story = get_story(prompt, num_parts=3)
    logger.info(f"Generated full story: \n{story}")
