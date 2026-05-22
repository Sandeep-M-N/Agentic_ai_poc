# ─────────────────────────────────────────────
# diff_agent.py
# ─────────────────────────────────────────────
import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine, text as sql_text

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

_project_engine = create_engine(os.getenv("DATABASE_URL"))

# ─────────────────────────────────────────────
# 2. Tools
# ─────────────────────────────────────────────

@tool
def convert_parquet_to_json(parquet_file_path: str) -> str:
    """
    Convert a Parquet file to JSON format and save it.

    Args:
        parquet_file_path: Full path to the .parquet file
    Returns:
        Path to the generated JSON file, or error message.
    """
    try:
        print(f"the parquet file path is : {parquet_file_path}")
        if not os.path.exists(parquet_file_path):
            return f"ERROR: Parquet file not found at '{parquet_file_path}'"

        if not parquet_file_path.endswith(".parquet"):
            return "ERROR: Provided file is not a .parquet file."

        df = pd.read_parquet(parquet_file_path)
        json_file_path = parquet_file_path.replace(".parquet", ".json")
        df.to_json(json_file_path, orient="records", indent=4)
        

        # Normalize: strip whitespace from string columns
        # df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # json_file_path = parquet_file_path.replace(".parquet", "_new.json")
        # records = df.to_dict(orient="records")

        # with open(json_file_path, "w") as f:
        #     json.dump(records, f, indent=2, default=str)

        print(f"[INFO] Converted '{parquet_file_path}' → '{json_file_path}'")
        # print(f"[INFO] Total records: {len(records)}")
        return json_file_path

    except Exception as e:
        return f"ERROR during parquet→json conversion: {str(e)}"


@tool
def fetch_patient_data_from_db(old_project_id: str,old_project_name:str) -> str:
    """
    Fetch PatientData JSON from the PatientList table using old ProjectId.

    Args:
        old_project_id: ProjectId to match in PatientList table
    Returns:
        Path to saved JSON file with DB patient data, or error message.
    """
    try:
        query = f"""
            SELECT PatientData
            FROM PatientList
            WHERE ProjectId = '{old_project_id}'
        """

        with _project_engine.connect() as conn:
            result = conn.execute(sql_text(query))
            rows   = result.fetchall()

        if not rows:
            return f"ERROR: No records found in PatientList for ProjectId '{old_project_id}'"

        # PatientData column is already JSON — parse and flatten
        all_patients = []
        for row in rows:
            patient_data = row[0]
            if isinstance(patient_data, str):
                parsed = json.loads(patient_data)
            else:
                parsed = patient_data  # already dict/list

            if isinstance(parsed, list):
                all_patients.extend(parsed)
            elif isinstance(parsed, dict):
                all_patients.append(parsed)

        # Remove 'Status' key from each patient record
        for patient in all_patients:
            patient.pop('Status', None)

        # Save to file
        json_file_path = f"patient_db_{old_project_name}_old.json"
        with open(json_file_path, "w") as f:
            json.dump(all_patients, f, indent=2, default=str)

        print(f"[INFO] Fetched {len(all_patients)} patient records from DB")
        print(f"[INFO] Saved to '{json_file_path}'")
        return json_file_path

    except Exception as e:
        return f"ERROR fetching patient data: {str(e)}"


@tool
def compare_patient_json(
    old_json_file: str,
    new_json_file: str,
    key_column: str = "USUBJID"
) -> str:
    """
    Compare old (DB) and new (parquet-converted) patient JSON files.
    Finds:
        - ADDED   : records in new but not in old
        - DELETED : records in old but not in new
        - MODIFIED: records in both but with changed field values

    Args:
        old_json_file: Path to DB patient JSON file
        new_json_file: Path to parquet-converted JSON file
        key_column:    Unique patient identifier column (default: USUBJID)
    Returns:
        Path to diff result JSON file + summary string.
    """
    try:
        # ── Load both JSONs ──────────────────────────────────────────
        with open(old_json_file, "r") as f:
            old_records = json.load(f)
        with open(new_json_file, "r") as f:
            new_records = json.load(f)

        if not isinstance(old_records, list) or not isinstance(new_records, list):
            return "ERROR: Both JSON files must contain a list of records."

        # ── Index by key_column ──────────────────────────────────────
        def index_by_key(records, key):
            indexed = {}
            for rec in records:
                k = str(rec.get(key, "")).strip()
                if k:
                    indexed[k] = rec
                else:
                    print(f"[WARN] Record missing key '{key}': {rec}")
            return indexed

        old_index = index_by_key(old_records, key_column)
        new_index = index_by_key(new_records, key_column)

        old_keys = set(old_index.keys())
        new_keys = set(new_index.keys())

        # ── ADDED ────────────────────────────────────────────────────
        added_keys   = new_keys - old_keys
        added_records = [new_index[k] for k in sorted(added_keys)]

        # ── DELETED ──────────────────────────────────────────────────
        deleted_keys    = old_keys - new_keys
        deleted_records = [old_index[k] for k in sorted(deleted_keys)]

        # ── MODIFIED ─────────────────────────────────────────────────
        modified_records = []
        common_keys      = old_keys & new_keys

        for k in sorted(common_keys):
            old_rec = old_index[k]
            new_rec = new_index[k]

            # Compare all fields present in either record
            all_fields = set(old_rec.keys()) | set(new_rec.keys())
            field_diffs = {}

            for field in all_fields:
                old_val = str(old_rec.get(field, "")).strip()
                new_val = str(new_rec.get(field, "")).strip()

                if old_val != new_val:
                    field_diffs[field] = {
                        "old": old_val,
                        "new": new_val
                    }

            if field_diffs:
                modified_records.append({
                    key_column:   k,
                    "differences": field_diffs
                })

        # ── Build diff result ────────────────────────────────────────
        diff_result = {
            "summary": {
                "total_old_records":      len(old_records),
                "total_new_records":      len(new_records),
                "total_added":            len(added_records),
                "total_deleted":          len(deleted_records),
                "total_modified":         len(modified_records),
                "total_unchanged":        len(common_keys) - len(modified_records)
            },
            "added":    added_records,
            "deleted":  deleted_records,
            "modified": modified_records
        }

        # ── Save diff to file ────────────────────────────────────────
        diff_file = "patient_diff_result.json"
        with open(diff_file, "w") as f:
            json.dump(diff_result, f, indent=2, default=str)

        # ── Human-readable summary ───────────────────────────────────
        summary_lines = [
            "=" * 60,
            "PATIENT DATA DIFF SUMMARY",
            "=" * 60,
            f"  Old (DB) Records     : {len(old_records)}",
            f"  New (Parquet) Records: {len(new_records)}",
            "",
            f"  ✅ ADDED    : {len(added_records)} patients",
            f"  ❌ DELETED  : {len(deleted_records)} patients",
            f"  ✏️  MODIFIED : {len(modified_records)} patients",
            f"  ✔️  UNCHANGED: {len(common_keys) - len(modified_records)} patients",
            "",
        ]

        # Added detail
        if added_records:
            summary_lines.append("── ADDED PATIENTS ──────────────────────────────────")
            for rec in added_records[:10]:
                summary_lines.append(f"  + {rec.get(key_column)} → {rec}")
            if len(added_records) > 10:
                summary_lines.append(f"  ... and {len(added_records) - 10} more")
            summary_lines.append("")

        # Deleted detail
        if deleted_records:
            summary_lines.append("── DELETED PATIENTS ────────────────────────────────")
            for rec in deleted_records[:10]:
                summary_lines.append(f"  - {rec.get(key_column)} → {rec}")
            if len(deleted_records) > 10:
                summary_lines.append(f"  ... and {len(deleted_records) - 10} more")
            summary_lines.append("")

        # Modified detail
        if modified_records:
            summary_lines.append("── MODIFIED PATIENTS ───────────────────────────────")
            for rec in modified_records[:10]:
                summary_lines.append(f"  ~ {rec[key_column]}:")
                for field, diff in rec["differences"].items():
                    summary_lines.append(
                        f"      {field}: '{diff['old']}' → '{diff['new']}'"
                    )
            if len(modified_records) > 10:
                summary_lines.append(f"  ... and {len(modified_records) - 10} more")
            summary_lines.append("")

        summary_lines.append(f"Full diff saved to: {diff_file}")
        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)

    except FileNotFoundError as e:
        return f"ERROR: File not found — {str(e)}"
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON — {str(e)}"
    except Exception as e:
        return f"ERROR during comparison: {str(e)}"


@tool
def full_diff_pipeline(
    parquet_file_path: str,
    old_project_id: str,
    old_project_name: str,
    key_column: str = "USUBJID"
) -> str:
    """
    End-to-end diff pipeline:
    1. Convert Parquet → JSON (new data)
    2. Fetch PatientData from DB using old_project_id (old data)
    3. Compare both JSONs → find ADDED, DELETED, MODIFIED records

    Args:
        parquet_file_path: Path to the .parquet file from migration agent
        old_project_id:    ProjectId to query PatientList table in DB
        old_project_name:  Project name for file naming
        key_column:        Unique patient identifier (default: USUBJID)
    Returns:
        Full diff summary with added/deleted/modified patient records.
    """
    # ── Step 1: Convert Parquet → JSON ──────────────────────────────
    print(f"[INFO] Step 1: Converting parquet to JSON...")
    new_json_path = convert_parquet_to_json.run(parquet_file_path)
    if new_json_path.startswith("ERROR"):
        return f"Pipeline failed at Step 1 (Parquet→JSON): {new_json_path}"

    # ── Step 2: Fetch old patient data from DB ───────────────────────
    print(f"[INFO] Step 2: Fetching old patient data from DB...")
    old_json_path = fetch_patient_data_from_db.run({"old_project_id": old_project_id, "old_project_name": old_project_name})
    if old_json_path.startswith("ERROR"):
        return f"Pipeline failed at Step 2 (DB Fetch): {old_json_path}"

    # ── Step 3: Compare ──────────────────────────────────────────────
    print(f"[INFO] Step 3: Comparing old vs new patient data...")
    diff_result = compare_patient_json.run({
        "old_json_file": old_json_path,
        "new_json_file": new_json_path,
        "key_column":    key_column
    })

    return (
        f"FILES USED\n"
        f"  Parquet File : {parquet_file_path}\n"
        f"  New JSON     : {new_json_path}\n"
        f"  Old JSON     : {old_json_path}\n\n"
        f"{diff_result}"
    )


# ─────────────────────────────────────────────
# 3. System Prompt
# ─────────────────────────────────────────────
system_prompt = """
You are a Patient Data Diff Agent.

Purpose:
Given a Parquet file path, old ProjectId, and old project name:
1. Convert the Parquet file to JSON (new patient data)
2. Fetch existing PatientData from the PatientList DB table (old patient data)
3. Compare both and report ADDED, DELETED, and MODIFIED patient records

WORKFLOW — Always follow this exact order:

STEP 1: Call full_diff_pipeline with:
        - parquet_file_path : path to the .parquet file
        - old_project_id    : ProjectId to query PatientList table
        - old_project_name  : Project name for file naming
        - key_column        : unique identifier field (default: USUBJID)

STEP 2: Return the COMPLETE result exactly as the tool returns it:
        - Files used (parquet, new JSON, old JSON)
        - Summary counts (added / deleted / modified / unchanged)
        - Detailed breakdown of each change with field-level diffs

CONSTRAINTS:
- Always call full_diff_pipeline as the primary tool
- Do NOT guess or fabricate patient data or project IDs
- Do NOT modify the comparison logic
- If key_column is not specified, default to USUBJID
- Ask the user for missing inputs (parquet path, project ID, or project name) before proceeding
"""

# ─────────────────────────────────────────────
# 4. Create Agent
# ─────────────────────────────────────────────
Patient_Diff_Agent = create_agent(
    llm,
    tools=[
        # convert_parquet_to_json,
        # fetch_patient_data_from_db,
        # compare_patient_json,
        full_diff_pipeline
    ],
    system_prompt=system_prompt,
    name="Patient_Diff_Agent"
)

# # ─────────────────────────────────────────────
# # 7. Runner + Test
# # ─────────────────────────────────────────────
# def run_agent(query: str):
#     print(f"\n{'='*60}")
#     print(f"Query: {query}")
#     print('='*60)
#     for step in Patient_Diff_Agent.stream(
#         {"messages": [{"role": "user", "content": query}]}
#     ):
#         for update in step.values():
#             for message in update.get("messages", []):
#                 message.pretty_print()


# if __name__ == "__main__":
#     run_agent(
#         "this is my parquet file path : patient_results_arul_project_dec_new.parquet, the old project name is : GK_Summit_Dec and the project id is : 346"
#     )