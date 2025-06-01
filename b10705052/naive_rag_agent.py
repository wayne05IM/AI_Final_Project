from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
import json
import os
from dotenv import load_dotenv
import csv # Added for CSV handling
import asyncio # Added for async operations
import time # Added for timing
from pydantic import BaseModel, Field # Added for structured output
from typing import Optional # For optional fields if needed later

# Load environment variables from .env file
load_dotenv()

# Disable tracing
set_tracing_disabled(True)

# --- Configuration ---
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))  # Adjust based on API rate limits
VIOLATION_THRESHOLD = int(os.getenv("VIOLATION_THRESHOLD", "80"))  # Probability threshold for violation

# --- Custom Model Provider Setup ---
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, and EXAMPLE_MODEL_NAME environment variables."
    )

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        # Use the model_name passed in RunConfig, or default to MODEL_NAME from env
        effective_model_name = model_name or MODEL_NAME
        return OpenAIChatCompletionsModel(model=effective_model_name, openai_client=client)

CUSTOM_MODEL_PROVIDER = CustomModelProvider()

# --- Define Structured Output Model ---
class AdComplianceOutput(BaseModel):
    violation_probability: int = Field(..., alias="違法機率", description="Violation probability from 0 to 100. Example: 違法機率：80 -> 80")
    analysis: str = Field(..., alias="分析", description="Analysis of compliance, including logic and evidence")
    regulations: str = Field(..., alias="法規", description="Relevant legal articles")
    cases: str = Field(..., alias="案例", description="Relevant case judgments")

    class Config:
        populate_by_name = True # Allows using alias for population

# --- Helper function to load JSON data ---
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Skipping this file.")
        return None
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}. Skipping this file.")
        return None

# --- 1. Construct the Agent's Instructions ---

# Load baseline prompt
try:
    with open("baseline_prompt.txt", 'r', encoding='utf-8') as f:
        baseline_instructions = f.read()
except FileNotFoundError:
    print("Error: baseline_prompt.txt not found. Exiting.")
    exit()
except Exception as e:
    print(f"Error reading baseline_prompt.txt: {e}. Exiting.")
    exit()

# Load data from the '法規及案例 Vector Stores' directory
# data_directory = "法規及案例 Vector Stores"
# all_json_data_content = []
# try:
#     for filename in os.listdir(data_directory):
#         if filename.endswith(".json"):
#             file_path = os.path.join(data_directory, filename)
#             data = load_json_data(file_path)
#             if data:
#                 # Convert JSON object to string and add to list
#                 all_json_data_content.append(json.dumps(data, ensure_ascii=False, indent=2))
# except FileNotFoundError:
#     print(f"Error: Directory '{data_directory}' not found. Exiting.")
#     exit()
# except Exception as e:
#     print(f"Error reading files from '{data_directory}': {e}. Exiting.")
#     exit()

# Combine baseline instructions with the JSON data
# if not all_json_data_content:
#     print(f"Warning: No JSON data found in '{data_directory}'. Agent will only use baseline_prompt.txt.")
#     agent_instructions = baseline_instructions
# else:
#     combined_data_string = "\\n\\n--- Data Files ---" + "\\n\\n".join(all_json_data_content)
#     agent_instructions = baseline_instructions + combined_data_string

agent_instructions = baseline_instructions # Use only baseline_prompt.txt

# --- 2. Define the Agent ---
NaiveRAGAdComplianceAgent = Agent(
    name="NaiveRAGAdComplianceAgent",
    instructions=agent_instructions,
    # model parameter is removed, will be set in RunConfig
    output_type=AdComplianceOutput
)

# --- 4. Process Single Query (Helper Function) ---
async def process_single_query(semaphore, query_id, advertisement_text, violation_threshold=VIOLATION_THRESHOLD, progress_callback=None):
    """Process a single query with semaphore to control concurrency"""
    async with semaphore:
        try:
            result = await Runner.run(
                NaiveRAGAdComplianceAgent,
                advertisement_text,
                run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER, model=MODEL_NAME)
            )

            if result and result.final_output:
                try:
                    structured_response = result.final_output_as(AdComplianceOutput)
                    probability = structured_response.violation_probability
                    answer = 1 if probability <= violation_threshold else 0
                    submission_id = int(query_id) - 1
                    
                    if progress_callback:
                        progress_callback(query_id, probability, answer)
                    
                    return submission_id, answer

                except Exception as e_parse:
                    print(f"    Error parsing structured output for ID {query_id}: {e_parse}")
                    submission_id = int(query_id) - 1
                    if progress_callback:
                        progress_callback(query_id, 0, 0, error=str(e_parse))
                    return submission_id, 0

            else:
                error_message = result.error if result and result.error else "No content in agent response."
                print(f"    Error: No valid response for ID {query_id}: {error_message}")
                submission_id = int(query_id) - 1
                if progress_callback:
                    progress_callback(query_id, 0, 0, error=error_message)
                return submission_id, 0

        except Exception as e:
            print(f"    Error processing query ID {query_id}: {e}")
            submission_id = int(query_id) - 1
            if progress_callback:
                progress_callback(query_id, 0, 0, error=str(e))
            return submission_id, 0

# --- 5. Process Queries and Generate Submission File (Concurrent Version) ---
async def process_queries():
    input_csv_file = "final_project_query.csv"
    output_csv_file = "submission.csv"
    violation_threshold = VIOLATION_THRESHOLD
    max_concurrent_requests = MAX_CONCURRENT_REQUESTS

    try:
        # Read all queries first
        queries = []
        with open(input_csv_file, 'r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            
            if 'ID' not in reader.fieldnames or 'Question' not in reader.fieldnames:
                print(f"Error: CSV file '{input_csv_file}' must contain 'ID' and 'Question' columns.")
                return

            for row in reader:
                query_id = row.get("ID")
                advertisement_text = row.get("Question")

                if not query_id or not advertisement_text:
                    print(f"Warning: Skipping row with missing ID or Question: {row}")
                    continue

                queries.append((query_id, advertisement_text))

        print(f"Processing {len(queries)} queries from {input_csv_file} with max {max_concurrent_requests} concurrent requests...")
        
        start_time = time.time()
        completed_count = 0
        
        def progress_callback(query_id, probability, answer, error=None):
            nonlocal completed_count
            completed_count += 1
            if error:
                print(f"  [{completed_count}/{len(queries)}] ID {query_id}: ERROR - {error}")
            else:
                status = 'Legal' if answer == 1 else 'Illegal'
                print(f"  [{completed_count}/{len(queries)}] ID {query_id}: {probability}% -> {status}")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Process all queries concurrently
        tasks = [
            process_single_query(semaphore, query_id, advertisement_text, violation_threshold, progress_callback)
            for query_id, advertisement_text in queries
        ]

        # Wait for all tasks to complete
        print("Starting concurrent processing...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nConcurrent processing completed in {processing_time:.2f} seconds")
        print(f"Average time per query: {processing_time/len(queries):.2f} seconds")
        print(f"Throughput: {len(queries)/processing_time:.2f} queries/second")

        # Write results to output file
        with open(output_csv_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["ID", "Answer"])

            # Sort results by submission_id to maintain order
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"    Task {i} failed with exception: {result}")
                    # Use the original query_id for this failed task
                    query_id = queries[i][0]
                    submission_id = int(query_id) - 1
                    valid_results.append((submission_id, 0))
                else:
                    valid_results.append(result)

            # Sort by submission_id and write to file
            valid_results.sort(key=lambda x: x[0])
            for submission_id, answer in valid_results:
                writer.writerow([submission_id, answer])

        print(f"Processing complete. Output saved to {output_csv_file}")

    except FileNotFoundError:
        print(f"Error: Input CSV file '{input_csv_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred during CSV processing: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Naive RAG Ad Compliance Agent (Concurrent Version)...")
    print(f"Configuration:")
    print(f"  - Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"  - Violation Threshold: {VIOLATION_THRESHOLD}%")
    print()

    asyncio.run(process_queries())

    print("\n--- Instructions for running this script: ---")
    print("1. Make sure you have the OpenAI Python library, OpenAI Agents SDK, and python-dotenv installed:")
    print("   pip install openai python-dotenv openai-agents")
    print("2. Create a .env file in the same directory as this script with your API credentials:")
    print("       BASE_URL='your_openai_compatible_base_url'")
    print("       API_KEY='your_api_key'")
    print("       MODEL_NAME='your_model_name'")
    print("   Optional configuration:")
    print("       MAX_CONCURRENT_REQUESTS=10  # Adjust based on API rate limits")
    print("       VIOLATION_THRESHOLD=50      # Probability threshold for violation")
    print("3. Ensure 'baseline_prompt.txt' and 'final_project_query.csv' are in the same directory.")
    print("4. Run the script: python naive_rag_agent.py")
    print("5. The output will be saved in 'submission.csv'.")
    print()
    print("Note: This version processes queries concurrently for better performance.")
    print("Adjust MAX_CONCURRENT_REQUESTS in your .env file based on your API rate limits.")