import os
from openai import OpenAI

def call_llm(system, user):
    print("yay")

def quick_start():
    client = OpenAI(
  api_key=os.getenv("DASHSCOPE_API_KEY"),
  base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
  model="qwen3.6-plus",
  messages=[
    {"role": "user", "content": "Hello! Tell me a fun fact about AI."}
  ]
)





def main():
    call_llm()
    quick_start()

# openai.OpenAIError: Missing credentials. Please pass an `api_key`, `workload_identity`, `admin_api_key`, or set the `OPENAI_API_KEY` or `OPENAI_ADMIN_KEY` environment variable.
print(completion.choices[0].message.content)

if "__main__" == __name__:
    print(200)