import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from patient_query_migration_agent import Patient_Query_Migration_Agent
from validator2 import Study_Alignment_Validator_Agent
from Patient_diff_agent import Patient_Diff_Agent
from langgraph_supervisor import create_supervisor
load_dotenv()

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.2,
    max_tokens=None
)

# ═════════════════════════════════════════════════════════════════════
# SUPERVISOR — Main Orchestrator
# ═════════════════════════════════════════════════════════════════════

supervisor_prompt = """
You are the Clinical Study Migration Supervisor Agent.

You orchestrate four specialized agents to automate clinical study migration:

AGENTS UNDER YOUR CONTROL:
  1. study_alignment_agent   — Finds the best matching old project for a new project
  2. patient_migration_agent — Migrates patient list queries from old → new project
  3. patient_diff_agent      — Compares migrated parquet data vs existing DB patient data
  

YOUR WORKFLOW — Always follow this EXACT order:

STEP 1: STUDY ALIGNMENT
  - Call study_alignment_agent with the new project name
  - It will return:
      ProjectNumber      (the best matching old project)
      ProjectId          (used later for diff comparison)
      CreatedAt
      PatientListQueryID
  - Extract BOTH the old ProjectNumber AND ProjectId from its response
  - If response says DOES NOT MATCH → STOP and report to user

STEP 2: PATIENT QUERY MIGRATION
  - Call patient_migration_agent with:
      old_project_number = ProjectNumber from STEP 1
      new_project_number = the new project name given by the user
  - It will:
      - Fetch active subqueries from the old project
      - Replace old project name with new in all Logic fields
      - Execute the final patient list query
      - Save full results to a Parquet file
  - Extract the Parquet file path from its response
  - If migration fails → report the exact error and STOP

STEP 3: PATIENT DATA DIFF
  - Call patient_diff_agent with:
      parquet_file_path = Parquet file path from STEP 2
      old_project_id    = ProjectId from STEP 1
      key_column        = "USUBJID" (default)
  - It will:
      - Convert the Parquet file to JSON (new patient data)
      - Fetch existing PatientData from PatientList table using old ProjectId
      - Compare both and produce a full diff report
      - Return counts and details of:
          ✅ ADDED    — patients in new project not in old
          ❌ DELETED  — patients in old project not in new
          ✏️  MODIFIED — patients present in both but with changed field values
          ✔️  UNCHANGED — patients identical in both
      - Save full diff to patient_diff_result.json
  - If diff fails → report the exact error but DO NOT stop (migration already succeeded)

STEP 4: RETURN FINAL SUMMARY
  Report ALL of the following in a clean structured format:

  ── ALIGNMENT ──────────────────────────────────────────
  - Old project selected, its ProjectId
  - Reason for selection (latest CreatedAt + has PatientListQueryID)

  ── MIGRATION ──────────────────────────────────────────
  - Number of active subqueries migrated
  - Old project name → New project name
  - Path to saved Parquet results file

  ── DIFF REPORT ────────────────────────────────────────
  - Total old (DB) records vs new (Parquet) records
  - ✅ Added    : count + list of added USUBJIDs
  - ❌ Deleted  : count + list of deleted USUBJIDs
  - ✏️  Modified : count + per-patient field-level changes (old value → new value)
  - ✔️  Unchanged: count
  - Path to saved diff JSON file

ROUTING RULES:
  - ALWAYS run agents in order: alignment → migration → diff
  - NEVER skip study_alignment_agent — ProjectId from it is required for diff
  - NEVER skip patient_migration_agent — Parquet file from it is required for diff
  - If study_alignment_agent returns DOES NOT MATCH → STOP immediately
  - If patient_migration_agent fails → STOP and report exact error
  - If patient_diff_agent fails → report the error but still return migration summary
  - email_report_agent is OPTIONAL — only invoke if user explicitly requests an email

CONSTRAINTS:
  - Do NOT hallucinate project names, patient data, or diff results
  - Do NOT run migration without a validated old project from alignment
  - Do NOT run diff without a valid Parquet file from migration
  - Do NOT include raw SQL in your final output
  - Always pass ProjectId (not ProjectNumber) to patient_diff_agent
  - Always confirm ALL steps completed before returning the final answer
"""
workflow = create_supervisor(
    agents=[Study_Alignment_Validator_Agent,Patient_Query_Migration_Agent,Patient_Diff_Agent],
    model=model,
    prompt=supervisor_prompt
)

app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Find which old project match the STUDYID of arul_project_dec_new"
        }
    ]
})
for message in result["messages"]:
    message.pretty_print()



