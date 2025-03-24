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


def get_info_from_dbpedia(prompt: str) -> str:
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


def get_subgraph_from_dbpedia(prompt: str) -> str:
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


def get_story(prompt: str) -> str:
    appendix_info = "\n".join(get_subgraph_from_dbpedia(prompt=prompt))

    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "Generate a story based on the prompt and additional information. Only output the story."
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "user",
                "content": appendix_info
            }
        ]
    )
    res = completion.choices[0].message.content
    res = res.replace("```", "")
    return res


if __name__ == "__main__":
    # history = input("Enter the history you want to know about: ")
    history = 'France revolution'
    prompt = f"I am intrested in the history about {history}"
    story = get_story(prompt)
    logger.info(f"Generated story: \n{story}")
