from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
# checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

messages = [
    {"role": "system", "content": "You are a helpful assistant, you speak spanish."},
    {"role": "user", "content": "Cuál es la mejor manera de cocinar cebollas?"},
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = pipe(input_text, return_full_text=False, max_new_tokens=200)
bot = response[0]["generated_text"]
if bot.startswith("assistant\n"):
    bot = bot[len("assistant\n") :].strip()

print(bot)
