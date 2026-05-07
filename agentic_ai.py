from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from azure.storage.blob import BlobServiceClient
import json
import os 
from dotenv import load_dotenv
load_dotenv()
blob_service_client = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "PatientAcumenView/")
llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.2,
        max_tokens=None
    )
schema_name = "MOH_Summit_ADAM_001_adam"


sql_server_conn_str = os.getenv("DATABASE_URL_FILES")
db = SQLDatabase.from_uri(sql_server_conn_str, schema=schema_name, sample_rows_in_table_info=0)
print(f"Dialect: {db.dialect}")
print(f"Connection string: {sql_server_conn_str}")
print(f"Available tables: {db.get_usable_table_names()}")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


system_prompt = f"""
Role: You are a Senior Clinical Data Scientist and CDISC Expert with deep expertise in SDTM and ADaM datasets with 10+yrs of experience.

Task: Retrieve precise answers from clinical study databases by analyzing SDTM (raw tabulation) and ADaM (analysis-ready) datasets.

Instructions:

1. Understand the Question

* Identify intent (e.g., patient count, adverse events, lab results, efficacy)
* Extract key elements: population, treatment, timepoints, conditions

2. Identify the Appropriate Domain

* Determine whether SDTM or ADaM is required:

  * Use SDTM for raw events (AE, LB, VS, DM)
  * Use ADaM for analysis (ADAE, ADLB, ADSL)
* Prefer ADaM unless raw data is explicitly needed

3. Validate Using Metadata

* Verify all variable names against metadata before using them
* Do not assume variable existence
* Common variables include: USUBJID, PARAMCD, AVAL, TRTxxA

4. Handle Terminology and Standardization

* Normalize synonyms and variations:

  * Example: "Heart Attack" → "Myocardial Infarction"
* Use standardized variables such as:

  * AEDECOD (coded term)
  * AETERM (reported term)
* Ensure case-insensitive comparisons where applicable

5. Build Query Logic

* Construct precise SQL queries using correct schema and table names
* Use USUBJID as the primary key for joins
* Apply appropriate filters, joins, and aggregations
* Ensure:

  * No duplicate subject counts
  * Proper null handling
  * Correct population selection logic

6. Apply Clinical Logic Validation

* Ensure correctness of:

  * Treatment-emergent flags (e.g., TRTEMFL)
  * Baseline definitions
  * Analysis population (e.g., ADSL filtering)
* Cross-check assumptions before finalizing

7. Output Format (Strict)
   Return ONLY:
   a) Final SQL query
   b) Brief explanation of logic (concise)
   c) Assumptions made (if any)

8. Constraints

* Do NOT hallucinate domains or variables
* If required data, variable, or domain is unclear or missing:

  * Explicitly state what is unclear (e.g., "AEDECOD not found in ADAE")
* If ambiguity exists, do NOT guess — flag it

Behavior:

* Be precise, structured, and domain-aware
* Think like a Clinical SAS Programmer and Biostatistician
* Prioritize correctness over completeness

"""

clinical_data_Agent = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt
)
query = "patient list with serious adverse event"
for step in clinical_data_Agent.stream(
    {"messages":[{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()

# ─────────────────────────────────────────────
# 2. Helper — fetch and parse a single JSON blob
# ─────────────────────────────────────────────
def fetch_json_blob(blob_name:str) -> dict | list:
    """Download a blob and parse it as JSON."""
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)
    raw = blob_client.download_blob().readall()
    return json.loads(raw)


# ─────────────────────────────────────────────
# 3. Tools for the Agent
# ─────────────────────────────────────────────
#@tool
def list_json_files() -> str:
    """
    List all JSON files inside the PatientAcumenView folder
    in Azure Blob Storage container.
    Returns a newline-separated list of full blob paths.
    """
    try:
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blobs = container_client.list_blobs(name_starts_with=BLOB_PREFIX)
        json_files = [b.name for b in blobs if b.name.endswith(".json")]

        if not json_files:
            return f"No JSON files found under '{BLOB_PREFIX}'."
        return "\n".join(json_files)
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def read_json_file(file_name: str) -> str:
    """
    Read and return the full content of a specific JSON file from Azure Blob Storage.
    Args:
        file_name: Exact name of the JSON file (e.g., 'data/patients.json')
    Returns:
        Pretty-printed JSON content as a string.
    """
    try:
        data = fetch_json_blob(file_name)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading file '{file_name}': {str(e)}"


@tool
def search_json_file(file_name: str, search_key: str, search_value: str) -> str:
    """
    Search inside a JSON file for records matching a key-value pair.
    Useful when the JSON is a list of records (e.g., list of patients or events).

    Args:
        file_name: Exact name of the JSON file (e.g., 'data/patients.json')
        search_key: The field/key to search on (e.g., 'USUBJID', 'AEDECOD')
        search_value: The value to match (case-insensitive)
    Returns:
        Matching records as a JSON string, or a message if nothing found.
    """
    try:
        data = fetch_json_blob(file_name)

        # Handle both list of records and single dict
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return "JSON structure is not a list of records — cannot search."

        matches = [
            record for record in data
            if isinstance(record, dict)
            and search_key in record
            and str(record[search_key]).lower() == search_value.lower()
        ]

        if not matches:
            return f"No records found where '{search_key}' = '{search_value}'."
        return json.dumps(matches, indent=2)

    except Exception as e:
        return f"Error searching file '{file_name}': {str(e)}"


# @tool
# def summarize_json_file(file_name: str) -> str:
#     """
#     Summarize a JSON file — show its structure, field names, record count,
#     and sample of first 2 records. Useful before doing a deep read.

#     Args:
#         file_name: Exact name of the JSON file
#     Returns:
#         Summary string with schema info and sample data.
#     """
#     try:
#         data = fetch_json_blob(file_name)

#         if isinstance(data, dict):
#             return (
#                 f"Type: Single JSON object\n"
#                 f"Keys: {list(data.keys())}\n"
#                 f"Sample:\n{json.dumps(data, indent=2)[:1000]}"
#             )

#         if isinstance(data, list):
#             keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
#             return (
#                 f"Type: List of records\n"
#                 f"Total records: {len(data)}\n"
#                 f"Fields: {keys}\n"
#                 f"First 2 records:\n{json.dumps(data[:2], indent=2)}"
#             )

#         return f"Unexpected JSON structure: {type(data)}"

#     except Exception as e:
#         return f"Error summarizing file '{file_name}': {str(e)}"
azure_system_prompt = """
You are a Clinical Data Analyst with expertise in reading and interpreting 
JSON-based clinical datasets stored in Azure Blob Storage.

Your capabilities:
- List all available JSON files in the storage container
- Summarize a file's structure before reading it fully
- Read and parse full JSON file content
- Search within JSON files for specific records by field/value

Workflow to follow:
1. Always use 'list_json_files' first if the user doesn't specify a file
2. Use 'read_json_file' for full content when needed
3. Use 'search_json_file' to find specific records

Be concise and structured in your responses.
"""


# ─────────────────────────────────────────────
# 6. Create the Agent
# ─────────────────────────────────────────────
azure_blob_agent = create_agent(
    llm,
    tools=[list_json_files,read_json_file, search_json_file],
    system_prompt=azure_system_prompt

)
def run_agent(query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    for step in azure_blob_agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()
# file_name="PatientAcumenView"
# Test 1 — what files exist?
run_agent(f"read the MOH_Summit_ADAM_004 folder files and summarize it.")


    


    
