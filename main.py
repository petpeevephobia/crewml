import os
import sys

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def quick_start():
    content = call_llm(
        "You are a helpful assistant.",
        "Hello! Tell me a fun fact about AI.",
    )
    print(content)





def main():
    quick_start()


if __name__ == "__main__":
    main()
