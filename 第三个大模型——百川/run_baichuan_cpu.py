from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "/mnt/data/Baichuan2-7B-Chat"

prompts = [
    "‘中国队大胜美国队’ 和 ‘中国队大败美国队’，这两句话到底谁赢了？为什么？",
    "请用 Python 写一个一行代码的列表推导式，筛选出 list_a 中所有的偶数。注意：不要输出任何解释文字，只能输出这一行代码。",
    "如果一斤棉花和一斤铁从同一高度同时掉落，在考虑空气阻力的情况下，哪个先落地？"
]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto"
).eval()

for i, prompt in enumerate(prompts, 1):
    print(f"\n\n==== TESTING QUESTION {i} ====")
    print(f"PROMPT: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    
    response = model.chat(tokenizer, messages)
    
    print(f"OUTPUT:\n{response.strip()}\n")
