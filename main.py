import os
import sys
import json

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path=".env"):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
_load_dotenv()


def _dashscope_client():
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


def cleanup_previous_runs():
    """Removes specific generated files from previous runs."""
    files_to_delete = [
        "01_agent_clean/cleaned.csv",
        "01_agent_clean/cleaning_report.md",
        "02_agent_feature/featured.csv",
        "02_agent_feature/feature_report.md"
    ]
    
    for file_path in files_to_delete:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"Deleted: {file_path}")
        else:
            print(f"Not found, skipping: {file_path}")
            
    # Re-initialize state.json as an empty dictionary after deletion
    with open("state.json", "w") as f:
        json.dump({}, f)
    print("State initialized: state.json cleared.")


def call_llm(system, user):
    client = _dashscope_client()
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return completion.choices[0].message.content


def _load_agent(module_name: str, folder: str, filename: str):
    path = _ROOT / folder / filename
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():

    

    # Delete all previously generated files for a fresh start
    cleanup_previous_runs()
    
    # AGENT 1
    print("Running Agent 1: Data Cleaning ...")
    agent_1 = _load_agent("agent_1", "01_agent_clean", "agent_1.py")
    result_1 = agent_1.run(call_llm)
    print(f"Agent 1 done: {result_1['rows_in']:,} → {result_1['rows_out']:,} rows")
    print(f"Cleaned CSV: {result_1['cleaned_csv']}")
    print(f"Report:      {result_1['report']}")
    print()

    # AGENT 2
    print("Running Agent 2: Feature Curation ...")
    agent_2 = _load_agent("agent_2", "02_agent_feature", "agent_2.py")
    result_2 = agent_2.run(call_llm)
    print(f"Agent 2 done: {result_2['rows_in']:,} → {result_2['rows_out']:,} rows")
    print(f"New columns: {', '.join(result_2['new_columns']) or '(none)'}")
    print(f"Featured CSV: {result_2['featured_csv']}")
    print(f"Report:       {result_2['report']}")
    print()

    # AGENT 3
    print("Running Agent 3: Model Selection ...")
    agent_3 = _load_agent("agent_3", "03_agent_model", "agent_3.py")
    result_3 = agent_3.run(call_llm)
    print(f"Agent 3 done: {result_3['analysis']}")
    for model_name, metrics in result_3['comparison_table'].items():
        print(f"  * {model_name}:")
        print(f"    - R2 Score: {metrics['R2_Score']}")
        print(f"    - RMSE:     {metrics['RMSE']}")
    print("Results saved to state.json")
    print()

    # AGENT 4
    print("Running Agent 4: Validation ...")
    agent_4 = _load_agent("agent_4", "04_agent_validation", "agent_4.py")
    result_4 = agent_4.run(call_llm)
    print(f"Agent 4 done.")
    print(f"  Winner model: {result_4['winner_model']}")
    print(f"  Mean CV R-square:  {result_4['cross_validation']['mean_r2']}\n  (+/-{result_4['cross_validation']['std_r2']})")
    print(f"  Verdict:\n{result_4['verdict']}")
    print()


if __name__ == "__main__":
    main()
