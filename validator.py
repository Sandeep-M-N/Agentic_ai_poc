import os
import json
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

load_dotenv()

# ─────────────────────────────────────────────
# 1. LLM
# ─────────────────────────────────────────────
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.2,
    max_tokens=None
)

SQL_CONN_STR = os.getenv("DATABASE_URL_FILES")


# ─────────────────────────────────────────────
# 2. Helper — schema-scoped DB connection
# ─────────────────────────────────────────────
def get_db(schema: str) -> SQLDatabase:
    """Return a SQLDatabase scoped to a specific schema."""
    return SQLDatabase.from_uri(
        SQL_CONN_STR,
        schema=schema,
        sample_rows_in_table_info=0
    )


def run_query(schema: str, query: str) -> str:
    """Execute a SQL query against a specific schema and return results as string."""
    try:
        db = get_db(schema)
        result = db.run(query)
        return result if result else "No results returned."
    except Exception as e:
        return f"Error querying schema '{schema}': {str(e)}"


# ─────────────────────────────────────────────
# 3. Schema-aware Tools
# ─────────────────────────────────────────────
@tool
def list_tables_in_schema(schema_name: str) -> str:
    """
    List all tables available in a given SQL schema.

    Args:
        schema_name: The database schema name
                     
    Returns:
        Comma-separated list of table names in that schema.
    """
    try:
        db = get_db(schema_name)
        tables = db.get_usable_table_names()
        if not tables:
            return f"No tables found in schema '{schema_name}'."
        return f"Tables in '{schema_name}': {', '.join(tables)}"
    except Exception as e:
        return f"Error listing tables in schema '{schema_name}': {str(e)}"


@tool
def get_distinct_studyid(schema_name: str, table_name: str) -> str:
    """
    Get all DISTINCT STUDYID values from a specific table in a schema.

    Args:
        schema_name: The database schema 
        table_name: The table to query 
    Returns:
        List of distinct STUDYID values found, or error message.
    """
    query = f"SELECT DISTINCT STUDYID FROM [{schema_name}].[{table_name}]"
    result = run_query(schema_name, query)
    return f"STUDYID values in [{schema_name}].[{table_name}]: {result}"


@tool
def compare_studyids(
    schema1: str,
    table1: str,
    schema2: str,
    table2: str
) -> str:
    """
    Compare STUDYID values between two schema+table combinations.
    Returns whether any STUDYID values match (case-insensitive).

    Args:
        schema1: First schema  
        table1:  Table in first schema 
        schema2: Second schema 
        table2:  Table in second schema 
    Returns:
        MATCHES or DOES NOT MATCH with the actual STUDYID values found.
    """
    try:
        query1 = f"SELECT DISTINCT UPPER(STUDYID) AS STUDYID FROM [{schema1}].[{table1}]"
        query2 = f"SELECT DISTINCT UPPER(STUDYID) AS STUDYID FROM [{schema2}].[{table2}]"

        result1 = run_query(schema1, query1)
        result2 = run_query(schema2, query2)

        # Parse results into sets for comparison
        def parse_ids(raw: str) -> set:
            ids = set()
            for line in raw.strip().splitlines():
                line = line.strip().strip("(),'\" ")
                if line and line.upper() != "STUDYID":
                    ids.add(line.upper())
            return ids

        ids1 = parse_ids(result1)
        ids2 = parse_ids(result2)

        common = ids1 & ids2

        if common:
            return (
                f"MATCHES\n"
                f"Common STUDYID(s): {', '.join(common)}\n"
                f"{schema1}.{table1} STUDYIDs: {', '.join(ids1)}\n"
                f"{schema2}.{table2} STUDYIDs: {', '.join(ids2)}"
            )
        else:
            return (
                f"DOES NOT MATCH\n"
                f"{schema1}.{table1} STUDYIDs: {', '.join(ids1) or 'none'}\n"
                f"{schema2}.{table2} STUDYIDs: {', '.join(ids2) or 'none'}"
            )

    except Exception as e:
        return f"Error comparing STUDYIDs: {str(e)}"


@tool
def run_sql_on_schema(schema_name: str, sql_query: str) -> str:
    """
    Run a raw SQL query on a specific schema. Use only when other tools
    are insufficient.

    Args:
        schema_name: The schema to run the query against
        sql_query: A valid SQL SELECT statement
    Returns:
        Query results as a string.
    """
    return run_query(schema_name, sql_query)


# ─────────────────────────────────────────────
# 4. System Prompt
# ─────────────────────────────────────────────
system_prompt = """
Role: You are a Study Alignment Validator Agent in a clinical data system.

Purpose:
Determine whether two clinical study projects belong to the SAME STUDY
based on matching STUDYID values in their database schemas.

Schema Naming Convention:
- SDTM schema: {project_name}_sdtm  
- ADaM schema: {project_name}_adam   

VALIDATION WORKFLOW — Follow these steps in ORDER:

STEP 1: List tables in both SDTM schemas
- Use list_tables_in_schema for:
  - {old_project}_sdtm
  - {new_project}_sdtm
- Find tables common to BOTH

STEP 2: Select the best validation table using this priority:
  1. DM
  2. AE
  3. LB
  4. VS
- If no common SDTM tables → try ADaM schemas with ADSL

STEP 3: Compare STUDYIDs
- Use compare_studyids tool with the selected table from STEP 2
- This handles case-insensitive comparison automatically

STEP 4: Return final result
- Return ONLY: MATCHES or DOES NOT MATCH
- Do NOT explain reasoning
- Do NOT include SQL
- Do NOT proceed to any other task
"""


# ─────────────────────────────────────────────
# 5. Create Agent
# ─────────────────────────────────────────────
Study_Alignment_Validator_Agent = create_agent(
    llm,
    tools=[
        list_tables_in_schema,
        compare_studyids,
    ],
    system_prompt=system_prompt
)


# ─────────────────────────────────────────────
# 6. Runner + Test
# ─────────────────────────────────────────────
# def run_agent(query: str):
#     print(f"\n{'='*60}")
#     print(f"Query: {query}")
#     print('='*60)
query = "Check if STUDYID matches between arul_project_dec_new and arul_project_sep_old."

for step in Study_Alignment_Validator_Agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()
# if __name__ == "__main__":
#     run_agent(
#         "Check if STUDYID matches between "
#         "arul_project_dec_new and arul_project_sep_old."
#     )