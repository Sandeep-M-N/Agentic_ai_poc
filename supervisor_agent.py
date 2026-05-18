import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from patient_query_migration_agent import Patient_Query_Migration_Agent
from validator2 import Study_Alignment_Validator_Agent
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

You orchestrate two specialized agents to automate clinical study migration:

AGENTS UNDER YOUR CONTROL:
  1. study_alignment_agent   — Finds the best matching old project for a new project
  2. patient_migration_agent — Migrates patient list queries from old → new project

YOUR WORKFLOW — Always follow this EXACT order:

STEP 1: STUDY ALIGNMENT
  - Call study_alignment_agent with the new project name
  - It will return:
      ProjectNumber  (the best matching old project)
      ProjectId
      CreatedAt
      PatientListQueryID
  - Extract the old ProjectNumber from its response

STEP 2: PATIENT QUERY MIGRATION
  - Call patient_migration_agent with:
      old_project_number = ProjectNumber from STEP 1
      new_project_number = the new project name given by the user
  - It will:
      - Fetch active subqueries from the old project
      - Replace old project name with new in all Logic fields
      - Execute the final patient list query
      - Save full results to a JSON file

STEP 3: RETURN FINAL SUMMARY
  - Old project selected and why (latest CreatedAt + has PatientListQueryID)
  - Migration summary (how many subqueries migrated)
  - Total patients found in new project
  - Path to saved results file

ROUTING RULES:
  - ALWAYS run study_alignment_agent BEFORE patient_migration_agent
  - NEVER skip study_alignment_agent — you need its output to run migration
  - If study_alignment_agent returns DOES NOT MATCH → stop and report
  - If patient_migration_agent fails → report the exact error

CONSTRAINTS:
  - Do NOT hallucinate project names or patient data
  - Do NOT run migration without a validated old project from alignment
  - Do NOT include raw SQL in your final output
  - Always confirm both steps completed before returning final answer
"""
workflow = create_supervisor(
    agents=[Study_Alignment_Validator_Agent,Patient_Query_Migration_Agent],
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



