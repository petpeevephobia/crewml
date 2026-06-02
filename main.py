import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
    agent_1 = _load_agent("agent_1", "01_agent_clean", "agent_1.py")
    result_1 = agent_1.run(call_llm)
    print(f"Agent 1 done: {result_1['rows_in']:,} → {result_1['rows_out']:,} rows")
    print(f"Cleaned CSV: {result_1['cleaned_csv']}")
    print(f"Report:      {result_1['report']}")
    print()

    agent_2 = _load_agent("agent_2", "02_agent_feature", "agent_2.py")
    result_2 = agent_2.run(call_llm)
    print(f"Agent 2 done: {result_2['rows_in']:,} → {result_2['rows_out']:,} rows")
    print(f"New columns: {', '.join(result_2['new_columns']) or '(none)'}")
    print(f"Featured CSV: {result_2['featured_csv']}")
    print(f"Report:       {result_2['report']}")


if __name__ == "__main__":
    main()
