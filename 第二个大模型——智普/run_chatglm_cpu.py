from transformers import AutoTokenizer, AutoModel

model_name = "/mnt/data/chatglm3-6b"

prompts = [
    "‘中国队大胜美国队’ 和 ‘中国队大败美国队’，这两句话到底谁赢了？为什么？",
    "请用 Python 写一个一行代码的列表推导式，筛选出 list_a 中所有的偶数。注意：不要输出任何解释文字，只能输出这一行代码。",
    "如果一斤棉花和一斤铁从同一高度同时掉落，在考虑空气阻力的情况下，哪个先落地？"
]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True
).float().eval()

for i, prompt in enumerate(prompts, 1):
    print(f"\n\n==== TESTING QUESTION {i} ====")
    print(f"PROMPT: {prompt}")

    response, history = model.chat(tokenizer, prompt, history=[])
    
    print(f"OUTPUT:\n{response.strip()}\n")
