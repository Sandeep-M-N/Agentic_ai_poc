import os
import ast
import re
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

PROJECT_DB_URL = os.getenv("DATABASE_URL")

# ─────────────────────────────────────────────
# 2. Shared DB connection
# ─────────────────────────────────────────────
_project_db = SQLDatabase.from_uri(PROJECT_DB_URL, sample_rows_in_table_info=0)

def run_query(query: str):
    """
    Run SQL and return parsed list of tuples.
    Returns [] on error or empty result.
    """
    try:
        result = _project_db.run(query)
        if not result or result.strip() in ("", "No results returned."):
            return []
        parsed = ast.literal_eval(result)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return [parsed]
        return []
    except Exception as e:
        print(f"[ERROR] run_query failed: {e}\nQuery: {query}")
        return []


# ─────────────────────────────────────────────
# 3. Tools
# ─────────────────────────────────────────────
@tool
def get_project_id(old_project_number: str) -> str:
    """
    Get ProjectId from the Project table using the old project number.

    Args:
        old_project_number: e.g., 'arul_project_sep_old'
    Returns:
        ProjectId as string, or 'NONE' if not found.
    """
    query = f"""
        SELECT ProjectId
        FROM Project
        WHERE ProjectNumber = '{old_project_number}'
    """
    rows = run_query(query)
    if not rows:
        print(f"[WARN] No ProjectId found for '{old_project_number}'")
        return "NONE"

    project_id = str(rows[0][0]).strip()
    print(f"[INFO] ProjectId for '{old_project_number}': {project_id}")
    return project_id


@tool
def get_patient_list_query_id(project_id: str) -> str:
    """
    Get PatientListQueryID and GlobalDatasetType from PatientListQuery table using ProjectId.

    Args:
        project_id: ProjectId from the Project table
    Returns:
        PatientListQueryID and GlobalDatasetType as string, or 'NONE' if not found.
    """
    query = f"""
        SELECT PatientListQueryID, GlobalDatasetType
        FROM PatientListQuery
        WHERE ProjectId = '{project_id}'
    """
    rows = run_query(query)
    if not rows:
        print(f"[WARN] No PatientListQueryID found for ProjectId '{project_id}'")
        return "NONE"

    query_id = str(rows[0][0]).strip()
    global_dataset_type = str(rows[0][1]).strip() if rows[0][1] else ""
    print(f"[INFO] PatientListQueryID: {query_id}, GlobalDatasetType: {global_dataset_type}")
    return f"{query_id}|{global_dataset_type}"


@tool
def get_active_subqueries(patient_list_query_id: str) -> str:
    """
    Get all active subqueries from PatientListSubQuery table
    using PatientListQueryID where Status = 'Active'.

    Args:
        patient_list_query_id: PatientListQueryID from PatientListQuery table
    Returns:
        JSON-like string of active subquery records, or 'NONE' if not found.
    """
    query = f"""
        SELECT 
            PatientListSubQueryID,
            QueryTitle,
            [FreeText],
            DatasetType,
            Logic
        FROM PatientListSubQuery
        WHERE PatientListQueryID = '{patient_list_query_id}'
          AND Status = 'A'
    """
    rows = run_query(query)
    if not rows:
        print(f"[WARN] No active subqueries for PatientListQueryID '{patient_list_query_id}'")
        return "NONE"

    print(f"[INFO] Found {len(rows)} active subqueries")
    # Return as clean string representation for the next tool to parse
    return str(rows)


@tool
def migrate_subquery_logic(
    old_project_number: str,
    new_project_number: str,
    patient_list_query_id: str,
    global_dataset_type: str
) -> str:
    """
    Fetch all active subqueries and replace the old project name
    with the new project name in the Logic field.

    Handles schema suffixes: _sdtm, _adam, _ADaM, _SDTM (case-insensitive).

    Args:
        old_project_number: 
        new_project_number: 
        patient_list_query_id: PatientListQueryID from PatientListQuery table
    Returns:
        All active subqueries with updated Logic fields.
    """
    # ── Step 1: Fetch active subqueries ─────────────────────────────
    query = f"""
        SELECT 
            PatientListSubQueryID,
            QueryTitle,
            [FreeText],
            DatasetType,
            Logic
        FROM PatientListSubQuery
        WHERE PatientListQueryID = '{patient_list_query_id}'
          AND Status = 'A'
    """
    rows = run_query(query)
    if not rows:
        return f"No active subqueries found for PatientListQueryID '{patient_list_query_id}'."

    print(f"[INFO] Migrating logic for {len(rows)} active subqueries")

    # ── Step 2: Replace old project name with new in Logic ──────────
    # Pattern matches old_project_number followed by optional _sdtm/_adam
    # suffix variations (case-insensitive)
    pattern = re.compile(
        re.escape(old_project_number) + r'(_[a-zA-Z]+)?',
        re.IGNORECASE
    )

    results = []
    for row in rows:
        subquery_id  = str(row[0]).strip()
        query_title  = str(row[1]).strip() if row[1] else ""
        free_text    = str(row[2]).strip() if row[2] else ""
        dataset_type = str(row[3]).strip() if row[3] else ""
        logic        = str(row[4]).strip() if row[4] else ""

        # Replace old project name preserving the suffix
        # e.g., arul_project_sep_old_ADaM → arul_project_dec_new_ADaM
        def replace_with_suffix(match):
            suffix = match.group(1) or ""
            return f"{new_project_number}{suffix}"

        updated_logic = pattern.sub(replace_with_suffix, logic)

        print(f"[INFO] SubQuery {subquery_id}:")
        print(f"       Original : {logic}")
        print(f"       Updated  : {updated_logic}")

        results.append({
            "PatientListSubQueryID": subquery_id,
            "QueryTitle":            query_title,
            "FreeText":              free_text,
            "DatasetType":           dataset_type,
            "OriginalLogic":         logic,
            "UpdatedLogic":          updated_logic
        })

    # ── Step 3: Format output ────────────────────────────────────────
    lines = [
        f"MIGRATION COMPLETE",
        f"Old Project : {old_project_number}",
        f"New Project : {new_project_number}",
        f"global_dataset_type:{global_dataset_type}",
        f"Total Active Subqueries Migrated: {len(results)}",
        ""
    ]
    for r in results:
        lines.append(f"  SubQueryID  : {r['PatientListSubQueryID']}")
        lines.append(f"  QueryTitle  : {r['QueryTitle']}")
        lines.append(f"  DatasetType : {r['DatasetType']}")
        lines.append(f"  FreeText    : {r['FreeText']}")
        lines.append(f"  OriginalLogic  : {r['OriginalLogic']}")
        lines.append(f"  UpdatedLogic   : {r['UpdatedLogic']}")
        lines.append("")

    return "\n".join(lines)


@tool
def migrate_patient_list_queries(
    old_project_number: str,
    new_project_number: str
) -> str:
    """
    Full pipeline: Given old and new project numbers, fetch all active
    subqueries from the old project and return them with updated Logic
    fields replacing old project name with new project name.

    Pipeline:
        Project (ProjectNumber) → ProjectId
        → PatientListQuery (ProjectId) → PatientListQueryID
        → PatientListSubQuery (PatientListQueryID, Status=Active)
        → Replace old_project_number with new_project_number in Logic

    Args:
        old_project_number: e.g., 'arul_project_sep_old'
        new_project_number: e.g., 'arul_project_dec_new'
    Returns:
        All migrated subqueries with updated Logic, or error message.
    """
    # ── Step 1: Get ProjectId ────────────────────────────────────────
    project_id = get_project_id.run(old_project_number)
    if project_id == "NONE":
        return (
            f"Could not find ProjectId for '{old_project_number}'. "
            f"Check if the project exists in the Project table."
        )
    print(f"[INFO] ProjectId: {project_id}")

    # ── Step 2: Get PatientListQueryID ───────────────────────────────
    result = get_patient_list_query_id.run(project_id)
    if result == "NONE":
        return (
            f"Could not find PatientListQueryID for ProjectId '{project_id}'. "
            f"Check if a PatientListQuery exists for this project."
        )
    patient_list_query_id, global_dataset_type = result.split("|")
    print(f"[INFO] PatientListQueryID: {patient_list_query_id}, GlobalDatasetType: {global_dataset_type}")

    # ── Step 3: Migrate Logic ────────────────────────────────────────
    result = migrate_subquery_logic.run({
        "old_project_number":    old_project_number,
        "new_project_number":    new_project_number,
        "patient_list_query_id": patient_list_query_id,
        "global_dataset_type": global_dataset_type
    })

    return result


# ─────────────────────────────────────────────
# 4. System Prompt
# ─────────────────────────────────────────────
system_prompt = """
You are a Patient List Query Migration Agent in a clinical data system.

Purpose:
Given an old project number and a new project number, retrieve all active
patient list subqueries from the old project and migrate their Logic fields
to reference the new project's schemas.

WORKFLOW — Always follow this exact order:

STEP 1: Call migrate_patient_list_queries with both project numbers.
        This single tool handles the full pipeline:
        - Looks up ProjectId from old project number
        - Gets PatientListQueryID from ProjectId  
        - Fetches all active subqueries
        - Replaces old project name with new project name in Logic

STEP 2: Return the result exactly as the tool returns it.
        - Show each subquery with its QueryTitle, DatasetType, FreeText
        - Show both OriginalLogic and UpdatedLogic side by side
        - Do NOT reformat or reinterpret results
        - If any step fails, report the exact error

CONSTRAINTS:
- Always call migrate_patient_list_queries first
- Do NOT hallucinate project names, IDs, or logic
- Do NOT modify the Logic beyond replacing the project name
- Do NOT include internal SQL in final output
"""


# ─────────────────────────────────────────────
# 5. Create Agent
# ─────────────────────────────────────────────
Patient_Query_Migration_Agent = create_agent(
    llm,
    tools=[
        # get_project_id,
        # get_patient_list_query_id,
        # get_active_subqueries,
        # migrate_subquery_logic,
        migrate_patient_list_queries,
    ],
    system_prompt=system_prompt
)


# ─────────────────────────────────────────────
# 6. Runner + Test
# ─────────────────────────────────────────────
def run_migration_agent(query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    for step in Patient_Query_Migration_Agent.stream(
        {"messages": [{"role": "user", "content": query}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()


if __name__ == "__main__":
    run_migration_agent(
        "Migrate patient list queries from old project "
        "'arul_project_sep_old' to new project 'arul_project_dec_new'"
    )