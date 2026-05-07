import os
import ast
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase

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
PROJECT_DB_URL = os.getenv("DATABASE_URL")

# ─────────────────────────────────────────────
# 2. Single shared DB connection
# ─────────────────────────────────────────────
_db = SQLDatabase.from_uri(SQL_CONN_STR, sample_rows_in_table_info=0)
_project_db = SQLDatabase.from_uri(PROJECT_DB_URL, sample_rows_in_table_info=0)

def run_raw_query(query: str, use_project_db: bool = False):
    """
    Run SQL and return parsed list of tuples via ast.literal_eval.
    Returns [] on error or empty result.
    """
    try:
        db = _project_db if use_project_db else _db
        result = db.run(query)
        if not result or result.strip() == "No results returned.":
            return []
        # pyodbc returns strings like: [('val1', 'val2'), ('val3', 'val4')]
        parsed = ast.literal_eval(result)
        # Normalize: always return list of tuples
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return [parsed]
        return []
    except Exception as e:
        print(f"[ERROR] run_raw_query failed: {e}\nQuery: {query}")
        return []


# ─────────────────────────────────────────────
# 3. System schemas to exclude
# ─────────────────────────────────────────────
SYSTEM_SCHEMAS = {
    "dbo", "guest", "information_schema", "sys",
    "db_owner", "db_accessadmin", "db_securityadmin",
    "db_ddladmin", "db_backupoperator", "db_datareader",
    "db_datawriter", "db_denydatareader", "db_denydatawriter"
}


# ─────────────────────────────────────────────
# 4. Tools
# ─────────────────────────────────────────────
@tool
def get_new_project_studyid(new_project_name: str) -> str:
    """
    Get the STUDYID of the new project from its _sdtm DM table.

    Args:
        new_project_name: Project name without suffix
                          (e.g., 'arul_project_dec_new')
    Returns:
        STUDYID string or 'NONE'.
    """
    schema = f"{new_project_name}_sdtm"
    query  = f"SELECT DISTINCT UPPER(STUDYID) AS STUDYID FROM [{schema}].[DM]"
    rows   = run_raw_query(query)

    if not rows:
        print(f"[WARN] No STUDYID found in [{schema}].[DM]")
        return "NONE"

    studyid = str(rows[0][0]).strip().upper()
    print(f"[INFO] New project STUDYID: {studyid}")
    return studyid


@tool
def find_matching_old_projects(new_project_name: str) -> str:
    """
    Find all old _sdtm schemas whose DM.STUDYID matches the new project.
    Uses batched queries to avoid SQL Server query size limits.

    Args:
        new_project_name: Project name without suffix
    Returns:
        Matching old project names and STUDYID, or DOES NOT MATCH.
    """
    # ── Step 1: Get new project STUDYID ─────────────────────────────
    new_studyid = get_new_project_studyid.run(new_project_name)
    if new_studyid == "NONE":
        return (
            f"Could not extract STUDYID from '{new_project_name}_sdtm'. "
            f"Check if the schema exists and has a DM table."
        )
    print(f"[INFO] Matching against STUDYID: '{new_studyid}'")

    # ── Step 2: Find all old _sdtm schemas with a DM table ──────────
    schemas_query = """
        SELECT DISTINCT TABLE_SCHEMA
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME = 'DM'
          AND TABLE_SCHEMA LIKE '%_sdtm'
        ORDER BY TABLE_SCHEMA
    """
    schema_rows = run_raw_query(schemas_query)
    if not schema_rows:
        return "No old _sdtm schemas with a DM table found."

    old_schemas = [
        row[0] for row in schema_rows
        if row[0].lower() not in SYSTEM_SCHEMAS
        and not row[0].lower().startswith(new_project_name.lower())
    ]

    if not old_schemas:
        return f"No old _sdtm schemas found (excluding '{new_project_name}_sdtm')."

    print(f"[INFO] Found {len(old_schemas)} candidate schemas")

    # ── Step 3: Batch UNION ALL — 10 schemas per query ──────────────
    BATCH_SIZE = 10  # Safe batch size for SQL Server
    matches = []

    for i in range(0, len(old_schemas), BATCH_SIZE):
        batch = old_schemas[i : i + BATCH_SIZE]
        print(f"[INFO] Batch {i//BATCH_SIZE + 1}: checking {batch}")

        union_parts = [
            f"SELECT '{s}' AS SCHEMA_NAME, "
            f"LTRIM(RTRIM(UPPER(CAST(STUDYID AS NVARCHAR(MAX))))) AS STUDYID "
            f"FROM [{s}].[DM]"
            for s in batch
        ]
        print(f"the union parts are : {union_parts}")
        batch_query = (
            "SELECT DISTINCT SCHEMA_NAME, STUDYID FROM ("
            + " UNION ALL ".join(union_parts)
            + ") AS batch_result "
            f"WHERE LTRIM(RTRIM(UPPER(STUDYID))) = '{new_studyid.strip()}'"
        )
        print(f"[DEBUG] Executing batch query:\n{batch_query}")

        rows = run_raw_query(batch_query)
        print(f"[INFO] Batch result: {rows}")

        for row in rows:
            schema  = str(row[0]).strip()
            studyid = str(row[1]).strip().upper()
            base    = schema.rsplit("_sdtm", 1)[0]
            matches.append({
                "project": base,
                "schema":  schema,
                "studyid": studyid
            })

    # ── Step 4: Validate against Project table ──────────────────────
    if not matches:
        return (
            f"DOES NOT MATCH\n"
            f"New project STUDYID '{new_studyid}' did not match "
            f"any of the {len(old_schemas)} old schemas scanned."
        )

    validated_matches = []
    for m in matches:
        project_name = m['project']
        check_query = f"""
            SELECT ProjectNumber 
            FROM Project 
            WHERE ProjectNumber = '{project_name}' 
              AND ProjectStatus = 'InProgress'
        """
        result = run_raw_query(check_query, use_project_db=True)
        if result:
            validated_matches.append(m)
            print(f"[INFO] ✓ Validated: {project_name} (InProgress)")
        else:
            print(f"[INFO] ✗ Skipped: {project_name} (not InProgress or not found)")

    if not validated_matches:
        return (
            f"DOES NOT MATCH\n"
            f"Found {len(matches)} STUDYID matches, but none have ProjectStatus='InProgress' in Project table."
        )

    lines = [
        f"MATCHES FOUND",
        f"New project : {new_project_name}",
        f"STUDYID     : {new_studyid}",
        ""
    ]
    for m in validated_matches:
        lines.append(f"  ✓ Old Project : {m['project']}")
        lines.append(f"    Schema      : {m['schema']}")
        lines.append(f"    STUDYID     : {m['studyid']}")
        lines.append("")

    lines.append(f"Total matches: {len(validated_matches)}")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# 5. System Prompt
# ─────────────────────────────────────────────
system_prompt = """
You are a Study Alignment Validator Agent in a clinical data system.

Purpose:
Given ONLY a new project name, find which old _sdtm projects share the same STUDYID.

WORKFLOW:

STEP 1: Call find_matching_old_projects with the new project name.
        This single tool handles everything internally.

STEP 2: Return the result exactly as the tool returns it.
        - Do NOT reformat or reinterpret the tool output
        - Do NOT add extra project names not in the tool result
        - If DOES NOT MATCH → state that clearly

CONSTRAINTS:
- Always call find_matching_old_projects first — do not call other tools first
- Do NOT hallucinate schema or project names
- Do NOT assume STUDYID values
- Do NOT include raw SQL in final output
"""


# ─────────────────────────────────────────────
# 6. Create Agent
# ─────────────────────────────────────────────
Study_Alignment_Validator_Agent = create_agent(
    llm,
    tools=[
        get_new_project_studyid,
        find_matching_old_projects,
    ],
    system_prompt=system_prompt
)


# ─────────────────────────────────────────────
# 7. Runner + Test
# ─────────────────────────────────────────────
def run_agent(query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    for step in Study_Alignment_Validator_Agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    run_agent(
        "Find which old projects match the STUDYID of arul_project_dec_new"
    )