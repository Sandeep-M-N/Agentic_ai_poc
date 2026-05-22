import os
import ast
import re
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text as sql_text
import pandas as pd
from datetime import datetime
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
FILES_DB_URL = os.getenv("DATABASE_URL_FILES")
_files_engine = create_engine(FILES_DB_URL)

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
          AND ToggleState = 1
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
@tool
def execute_patient_list_query(
    old_project_number: str,
    new_project_number: str,
    global_dataset_type: str,
    subqueries: str   # JSON string: [{SubQueryID, QueryTitle, UpdatedLogic}, ...]
) -> str:
    """
    Execute patient list query directly against DATABASE_URL_FILES.
    Replicates the /ExecuteQuery API logic but runs DB queries directly.

    Args:
        old_project_number:  e.g., 'arul_project_sep_old'
        new_project_number:  e.g., 'arul_project_dec_new'
        global_dataset_type: 'ADaM' or 'SDTM'
        subqueries:          JSON string list of dicts with keys:
                             SubQueryID, QueryTitle, UpdatedLogic
    Returns:
        Patient list result with status comparison.
    """
    import json

    # ── Step 1: Parse subqueries ─────────────────────────────────────
    try:
        sq_list = json.loads(subqueries)
    except json.JSONDecodeError as e:
        return f"Invalid subqueries JSON: {e}"

    if not sq_list:
        return "No subqueries provided."

    # ── Step 2: Determine schema and demo table ───────────────────────
    # Mirrors API logic exactly:
    # schema = {new_project}_{folder_name}
    # is_adam = folder_name.upper() == 'ADAM'
    # demo_table = 'ADAE' if is_adam else 'DM'
    folder_name = global_dataset_type.strip()           # 'ADaM' or 'SDTM'
    schema      = f"{new_project_number}_{folder_name}" # e.g., arul_project_dec_new_ADaM
    is_adam     = folder_name.upper() == "ADAM"
    demo_table  = "ADAE" if is_adam else "DM"

    print(f"[INFO] Schema     : {schema}")
    print(f"[INFO] Demo table : {demo_table}")
    print(f"[INFO] Subqueries : {len(sq_list)}")

    # ── Step 3: Build SELECT columns, subquery parts, WHERE conditions ─
    select_columns  = [
        f"{demo_table}.USUBJID",
        # f"{demo_table}.AGE",
        # f"{demo_table}.SEX",
        # f"{demo_table}.RACE"
    ]
    subquery_parts   = []
    where_conditions = []

    for idx, sq in enumerate(sq_list):
        logic      = sq.get("UpdatedLogic") or sq.get("Logic", "")
        sub_title  = sq.get("QueryTitle", f"Query_{sq.get('SubQueryID', idx)}")

        if not logic.strip():
            print(f"[WARN] Skipping subquery {idx+1} — empty Logic")
            continue

        # Remove DISTINCT from inner logic (mirrors API)
        logic = re.sub(r'\bDISTINCT\b', '', logic, flags=re.IGNORECASE)

        # Replace old project reference with new project in Logic
        # (already done in migration, but apply as safety net)
        logic = logic.replace(
            f'[{old_project_number}_',
            f'[{new_project_number}_'
        )

        # Column name from sub_title (mirrors API)
        col_name = sub_title.replace(' ', '_').replace('-', '_').upper()
        cnt_col  = f"{col_name}_CNT"

        # Wrap as subquery with COUNT (mirrors API exactly)
        subquery = (
            f"(SELECT USUBJID, COUNT(*) AS [{cnt_col}] "
            f"FROM ({logic}) t GROUP BY USUBJID) SQ{idx}"
        )
        subquery_parts.append(subquery)

        # Flag and count columns (mirrors API)
        select_columns.append(
            f"CASE WHEN COALESCE(SQ{idx}.[{cnt_col}],0)>0 "
            f"THEN 'X' ELSE '' END AS [{sub_title}]"
        )
        select_columns.append(
            f"COALESCE(SQ{idx}.[{cnt_col}],0) AS [{sub_title} Count]"
        )
        where_conditions.append(f"COALESCE(SQ{idx}.[{cnt_col}],0)>0")

    if not subquery_parts:
        return "All subqueries had empty Logic — nothing to execute."

    # ── Step 4: Build final query (mirrors API exactly) ───────────────
    base_table_query = (
        f"(SELECT DISTINCT USUBJID, AGE, SEX, RACE "
        f"FROM [{schema}].[{demo_table}]) {demo_table}"
    )
    final_query = f"SELECT {', '.join(select_columns)} FROM {base_table_query} "
    for idx, sq_part in enumerate(subquery_parts):
        final_query += f"LEFT JOIN {sq_part} ON {demo_table}.USUBJID = SQ{idx}.USUBJID "
    final_query += f"WHERE ({' OR '.join(where_conditions)}) ORDER BY {demo_table}.USUBJID"

    # print(f"[INFO] Final query:\n{final_query}")

    # ── Step 5: Execute directly against DATABASE_URL_FILES ───────────
    try:
        with _files_engine.connect() as conn:
            result_proxy = conn.execute(sql_text(final_query))
            columns      = list(result_proxy.keys())
            rows         = result_proxy.fetchall()
            result       = [
                {col: ("" if val is None else val)
                 for col, val in zip(columns, row)}
                for row in rows
            ]

        print(f"[INFO] Query returned {len(result)} patients")

    except Exception as e:
        return f"Query execution failed: {str(e)}\nSQL:\n{final_query}"

    if not result:
        return (
            f"Query executed successfully but returned 0 patients.\n"
            f"Schema: {schema} | Demo table: {demo_table}"
        )

    # # ── Step 6: Format output ─────────────────────────────────────────
    # total   = len(result)
    # headers = list(result[0].keys())

    # lines = [
    #     f"EXECUTION COMPLETE",
    #     f"New Project  : {new_project_number}",
    #     f"Old Project  : {old_project_number}",
    #     f"Schema       : {schema}",
    #     f"Demo Table   : {demo_table}",
    #     f"Total Patients Found: {total}",
    #     ""
    # ]

    # # Column headers
    # display_headers = headers[:6]
    # lines.append("  " + " | ".join(f"{h:<15}" for h in display_headers))
    # lines.append("  " + "-" * 80)

    # # First 10 rows
    # for rec in result[:10]:
    #     row_vals = [str(rec.get(h, ""))[:15] for h in display_headers]
    #     lines.append("  " + " | ".join(f"{v:<15}" for v in row_vals))

    # if total > 10:
    #     lines.append(f"  ... and {total - 10} more patients")

    # return "\n".join(lines)
     # ── Step 6: Save results to JSON file ──────────────────────────
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patient_results_{new_project_number}.parquet"
    
    output_data = {
        "new_project": new_project_number,
        "old_project": old_project_number,
        "schema": schema,
        "demo_table": demo_table,
        "total_patients": len(result),
        # "columns": columns,
        # "patients": result
    }
    
    # with open(filename, 'w') as f:
    #     json.dump(output_data, f, indent=2, default=str)

    # Convert patient records to DataFrame and save as Parquet
    df = pd.DataFrame(result)
    df.to_parquet(filename, index=False, engine="pyarrow")
    
    print(f"[INFO] Results saved to {filename}")
    return filename

@tool
def full_migration_and_execute(
    old_project_number: str,
    new_project_number: str
) -> str:
    """
    Complete end-to-end pipeline:
    1. Fetch active subqueries from old project and migrate Logic
    2. Execute migrated queries directly against DATABASE_URL_FILES
    3. Return patient list results

    Args:
        old_project_number: e.g., 'arul_project_sep_old'
        new_project_number: e.g., 'arul_project_dec_new'
    """
    import json, re as re_mod

    # ── Step 1: ProjectId ────────────────────────────────────────────
    project_id = get_project_id.run(old_project_number)
    if project_id == "NONE":
        return f"Could not find ProjectId for '{old_project_number}'."

    # ── Step 2: PatientListQueryID + GlobalDatasetType ───────────────
    query_result = get_patient_list_query_id.run(project_id)
    if query_result == "NONE":
        return f"Could not find PatientListQueryID for ProjectId '{project_id}'."

    patient_list_query_id, global_dataset_type = query_result.split("|", 1)
    print(f"[INFO] QueryID: {patient_list_query_id} | DatasetType: {global_dataset_type}")

    # ── Step 3: Fetch + migrate subquery Logic ───────────────────────
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
    rows = run_query(query)   # ← hits PROJECT_DB (metadata DB)
    if not rows:
        return f"No active subqueries for PatientListQueryID '{patient_list_query_id}'."

    pattern = re_mod.compile(
        re_mod.escape(old_project_number) + r'(_[a-zA-Z]+)?',
        re_mod.IGNORECASE
    )

    sq_list        = []
    migration_lines = [
        f"MIGRATION SUMMARY",
        f"Old Project : {old_project_number}",
        f"New Project : {new_project_number}",
        f"Total Active Subqueries: {len(rows)}",
        ""
    ]

    for row in rows:
        logic = str(row[4]).strip() if row[4] else ""

        def replace_suffix(match):
            suffix = match.group(1) or ""
            return f"{new_project_number}{suffix}"

        updated_logic = pattern.sub(replace_suffix, logic)

        sq_list.append({
            "SubQueryID":   str(row[0]).strip(),
            "QueryTitle":   str(row[1]).strip() if row[1] else "",
            "UpdatedLogic": updated_logic
        })

        migration_lines.append(f"  SubQueryID : {row[0]}")
        migration_lines.append(f"  QueryTitle : {row[1]}")
        migration_lines.append(f"  Original   : {logic}")
        migration_lines.append(f"  Updated    : {updated_logic}")
        migration_lines.append("")

    # ── Step 4: Execute directly against DATABASE_URL_FILES ──────────
    json_filename = execute_patient_list_query.run({
        "old_project_number":  old_project_number,
        "new_project_number":  new_project_number,
        "global_dataset_type": global_dataset_type,
        "subqueries":          json.dumps(sq_list)
    })

    migration_lines.append("=" * 60)
    migration_lines.append(f"Results saved to: {json_filename}")
    return "\n".join(migration_lines)


# ─────────────────────────────────────────────
# 4. System Prompt
# ─────────────────────────────────────────────
system_prompt = """
You are a Patient List Query Migration and Execution Agent.

Purpose:
Given an old and new project number:
1. Migrate all active subquery Logic fields from old → new project
2. Execute the migrated queries via the API
3. Return the patient list result

WORKFLOW — Always follow this exact order:

STEP 1: Call full_migration_and_execute with both project numbers.
        This single tool handles everything:
        - Fetches ProjectId → PatientListQueryID → Active SubQueries
        - Replaces old project name with new in all Logic fields
        - Calls /ExecuteQuery API with migrated subqueries
        - Returns patient list

STEP 2: Return the COMPLETE result exactly as the tool returns it.
        - Show migration summary (subqueries migrated)
        - Show patient counts and status breakdown
        - Report any errors exactly as returned

CONSTRAINTS:
- Always call full_migration_and_execute first
- Do NOT hallucinate project names, patient data, or logic
- Do NOT modify Logic beyond replacing the project name
- Do NOT include raw SQL in final output
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
        # migrate_patient_list_queries,
        execute_patient_list_query,
        full_migration_and_execute
    ],
    system_prompt=system_prompt,
    name="Patient_Query_Migration_Agent"
)


# # ─────────────────────────────────────────────
# # 6. Runner + Test
# # ─────────────────────────────────────────────
# def run_migration_agent(query: str):
#     print(f"\n{'='*60}")
#     print(f"Query: {query}")
#     print('='*60)
#     for step in Patient_Query_Migration_Agent.stream(
#         {"messages": [{"role": "user", "content": query}]}
#     ):
#         for update in step.values():
#             for message in update.get("messages", []):
#                 message.pretty_print()


# if __name__ == "__main__":
#     run_migration_agent(
#         "Migrate patient list queries from old project "
#         "'arul_project_sep_old' to new project 'arul_project_dec_new'"
#     )