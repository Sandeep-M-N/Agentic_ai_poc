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
        # Check if result starts with '[' or '(' - valid tuple/list representation
        result = result.strip()
        if not (result.startswith('[') or result.startswith('(')):
            print(f"[ERROR] Unexpected result format: {result[:100]}")
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
        # print(f"the union parts are : {union_parts}")
        batch_query = (
            "SELECT DISTINCT SCHEMA_NAME, STUDYID FROM ("
            + " UNION ALL ".join(union_parts)
            + ") AS batch_result "
            f"WHERE LTRIM(RTRIM(UPPER(STUDYID))) = '{new_studyid.strip()}'"
        )
        # print(f"[DEBUG] Executing batch query:\n{batch_query}")

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

@tool
def get_latest_old_project_with_query(old_projects: str) -> str:
    """
    Given a list of old project names (comma-separated or newline-separated),
    find the one with the latest CreatedAt that also has a PatientListQueryID.

    Pipeline:
        old_project names
        → Project table: get ProjectId + CreatedAt for each
        → Pick the one with latest CreatedAt
        → Check PatientListQuery: does that ProjectId have a PatientListQueryID?
        → If yes → return ProjectId + ProjectNumber
        → If no  → try next latest, and so on

    Args:
        old_projects: Comma or newline separated old project names
                      e.g., 'arul_project_sep_old, arul_project_jan_old'
    Returns:
        The single best old ProjectId and ProjectNumber, or error.
    """
    # ── Step 1: Parse input project names ───────────────────────────
    raw = old_projects.replace("\n", ",")
    project_names = [p.strip() for p in raw.split(",") if p.strip()]

    if not project_names:
        return "No old project names provided."

    print(f"[INFO] Checking {len(project_names)} old projects: {project_names}")

    # ── Step 2: Get ProjectId + CreatedAt for all old projects ───────
    # Single query using IN clause
    names_in = ", ".join(f"'{n}'" for n in project_names)
    query = f"""
        SELECT 
        ProjectId,
        ProjectNumber,
        CONVERT(VARCHAR, CreatedAt, 126) AS CreatedAt
    FROM Project
    WHERE ProjectNumber IN ({names_in})
    ORDER BY CreatedAt DESC
    """
    rows = run_raw_query(query, use_project_db=True)

    if not rows:
        return (
            f"No projects found in Project table for: {project_names}"
        )

    print(f"[INFO] Found {len(rows)} projects in Project table:")
    for row in rows:
        print(f"       ProjectId={row[0]} | ProjectNumber={row[1]} | CreatedAt={row[2]}")

    # ── Step 3: Sort by CreatedAt descending (latest first) ──────────
    # rows already ordered by CreatedAt DESC from SQL
    # but parse just in case
    from datetime import datetime

    def parse_dt(val):
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val))
        except Exception:
            return datetime.min

    sorted_rows = sorted(rows, key=lambda r: parse_dt(r[2]), reverse=True)

    print(f"[INFO] Sorted by CreatedAt (latest first):")
    for row in sorted_rows:
        print(f"       ProjectId={row[0]} | ProjectNumber={row[1]} | CreatedAt={row[2]}")

    # ── Step 4: Find the latest one that has a PatientListQueryID ────
    for row in sorted_rows:
        project_id     = str(row[0]).strip()
        project_number = str(row[1]).strip()
        created_at     = str(row[2]).strip()

        check_query = f"""
            SELECT PatientListQueryID
            FROM PatientListQuery
            WHERE ProjectId = '{project_id}'
        """
        result = run_raw_query(check_query, use_project_db=True)

        if result:
            patient_list_query_id = str(result[0][0]).strip()
            print(
                f"[INFO] ✓ Selected: ProjectNumber={project_number} | "
                f"ProjectId={project_id} | CreatedAt={created_at} | "
                f"PatientListQueryID={patient_list_query_id}"
            )
            return (
                f"SELECTED OLD PROJECT\n"
                f"  ProjectNumber      : {project_number}\n"
                f"  ProjectId          : {project_id}\n"
                f"  CreatedAt          : {created_at}\n"
                f"  PatientListQueryID : {patient_list_query_id}\n"
            )
        else:
            print(
                f"[INFO] ✗ Skipped: ProjectNumber={project_number} | "
                f"ProjectId={project_id} — no PatientListQueryID found"
            )

    # ── Step 5: None had a PatientListQueryID ────────────────────────
    return (
        f"NONE FOUND\n"
        f"Checked {len(sorted_rows)} old projects sorted by latest CreatedAt.\n"
        f"None of them have a PatientListQueryID in PatientListQuery table.\n"
        f"Projects checked: {[str(r[1]) for r in sorted_rows]}"
    )

# ─────────────────────────────────────────────
# 5. System Prompt
# ─────────────────────────────────────────────
system_prompt = """
You are a Study Alignment Validator Agent in a clinical data system.

Purpose:
Given ONLY a new project name, find the single best matching old project
that shares the same STUDYID and has patient list queries configured.

WORKFLOW — Always follow this exact order:

STEP 1: Call find_matching_old_projects with the new project name.
        Returns all old projects whose STUDYID matches and are InProgress.

STEP 2: Take the list of old project names from STEP 1 and call
        get_latest_old_project_with_query with those project names.
        This finds the single best old project by:
        - Getting ProjectId + CreatedAt for each from Project table
        - Picking the latest CreatedAt
        - Confirming it has a PatientListQueryID
        - Returns ProjectNumber, ProjectId, CreatedAt, PatientListQueryID

STEP 3: Return ONLY the result from STEP 2:
        - ProjectNumber (the single best old project)
        - ProjectId
        - CreatedAt
        - PatientListQueryID
        - Do NOT list all matched projects — only the final selected one

CONSTRAINTS:
- Always follow STEP 1 → STEP 2 → STEP 3 in order
- Do NOT skip get_latest_old_project_with_query
- Do NOT hallucinate project names or IDs
- Do NOT return all matched projects — only the single best one
- Do NOT include raw SQL in final output
"""

Study_Alignment_Validator_Agent = create_agent(
    llm,
    tools=[
        get_new_project_studyid,
        find_matching_old_projects,
        get_latest_old_project_with_query,   # ← new tool
    ],
    system_prompt=system_prompt,
    name="Study_Alignment_Validator_Agent"
)


# # ─────────────────────────────────────────────
# # 7. Runner + Test
# # ─────────────────────────────────────────────
# def run_agent(query: str):
#     print(f"\n{'='*60}")
#     print(f"Query: {query}")
#     print('='*60)
#     for step in Study_Alignment_Validator_Agent.stream(
#         {"messages": [{"role": "user", "content": query}]}
#     ):
#         for update in step.values():
#             for message in update.get("messages", []):
#                 message.pretty_print()


# if __name__ == "__main__":
#     run_agent(
#         "Find which old project match the STUDYID of arul_project_dec_new"
#     )