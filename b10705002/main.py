from openai import OpenAI
import json
import os
from utils import create_vector_store, create_update_assistant
import pandas as pd
from tqdm import tqdm
import re

client = OpenAI()

QUERY_CSV_PATH = "final_project_query.csv" 
ASSISTANT_ID = None
VECTOR_STORE_ID = None

if os.path.exists("openai_id.json"):
    with open("openai_id.json", "r") as f:
        data = json.load(f)
        ASSISTANT_ID = data["assistant_id"]
        VECTOR_STORE_ID = data["vector_store_id"]

else:
	with open("openai_id.json", "w") as f:
		json.dump({"assistant_id": None, "vector_store_id": None}, f)

print("Assistant ID:", ASSISTANT_ID)
print("Vector Store ID:", VECTOR_STORE_ID)

if not VECTOR_STORE_ID:
    print("Creating vector store...")
    VECTOR_STORE_ID = create_vector_store()
    print("Vector store has been created with ID:", VECTOR_STORE_ID)

print("Creating or updating assistant...")
ASSISTANT_ID = create_update_assistant(VECTOR_STORE_ID)
print("Assistant has been created with ID:", ASSISTANT_ID)
    
print("Assistant ID:", ASSISTANT_ID)
print("Vector Store ID:", VECTOR_STORE_ID)

def query(query: str) -> str:
    """
    Query the assistant with a given query string.
    """
    thread = client.beta.threads.create(
    messages=[
        {
			"role": "user",
			"content": query,
        }
    ]
    )
    
    # Use the create and poll SDK helper to create a run and poll the status of
    # the run until it's in a terminal state.

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=ASSISTANT_ID,
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    response = message_content.value + "\n".join(citations)
    return response

def post_process(response: str) -> bool:
    """
    Post-process the response to format it correctly.
    """
    # Here you can add any post-processing logic if needed
    match = re.search(r"違法[:：]\s*([YyNn])", response)
    percentage = 'N'
    if match:
        percentage = match.group(1)
        # print(f"Extracted percentage: {percentage}%")
    
    return percentage.upper() == 'Y'  # Return True if 'Y', False if 'N'


if __name__ == "__main__":
    df = pd.read_csv(QUERY_CSV_PATH)
    ids = df["ID"].tolist()
    queries = df["Question"].tolist()

    TEST_MODE = False  # Set to True for testing, False for full run
    if TEST_MODE:
        ids = ids[:5]  # Limit to first 10 for testing
        queries = queries[:5]  # Limit to first 10 for testing
    answers = []
    records = []
    for id, query_text in tqdm(zip(ids, queries), total=len(ids), colour="green", desc="Processing queries"):
        query_arg = f"請幫我分析以下的廣告是否違法:\n{query_text}"
        response = query(query_arg)
        answer = post_process(response)
        answers.append({
            "ID": id - 1,
            "Answer": int(answer)
        })
        records.append({
            "ID": id - 1,
            "Query": query_arg,
            "Response": response,
            "Answer": int(answer)
        })
    
    result_df = pd.DataFrame(answers)
    result_df.to_csv("final_project_result.csv", index=False)

    with open("final_project_records.txt", "w", encoding="utf-8") as f:
        for record in records:
            f.write(f"ID:\n{record['ID']}\n")
            f.write(f"Query:\n{record['Query']}\n")
            f.write(f"Response:\n{record['Response']}\n")
            f.write(f"Answer:\n{str(record['Answer'])}\n")
            f.write("-"*30 + "\n")