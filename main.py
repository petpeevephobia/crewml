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


def _load_agent_1():
    path = _ROOT / "01_agent_clean" / "agent_1.py"
    spec = spec_from_file_location("agent_1", path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    agent_1 = _load_agent_1()
    result = agent_1.run(call_llm)
    print(f"Agent 1 done: {result['rows_in']:,} → {result['rows_out']:,} rows")
    print(f"Cleaned CSV: {result['cleaned_csv']}")
    print(f"Report:      {result['report']}")


if __name__ == "__main__":
    main()
