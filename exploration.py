from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer and model with GPU support
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

system_prompt = """You are an expert algorithmic assistant specializing in graph problems. Keep your reasoning clear, accurate, and brief.
When the user presents a graph-related question, provide a concise, correct solution."""
prompt = """Consider the following undirected graph with 10 nodes labeled from A to J:

Nodes: {A, B, C, D, E, F, G, H, I, J} 
Edges: A - B, A - C, B - D, B - E, C - F, D - G, E - G, E - H, F - I, G - J, H - J, I - J

Determine what is the shortest path from node A to I."""

# Define your conversation history
conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

# Apply the chat template to format the input
formatted_input = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize the formatted input
input_ids = tokenizer.encode(formatted_input, return_tensors="pt").to(model.device)

# Generate a response from the model
output_ids = model.generate(
    input_ids,
    max_new_tokens=5000,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode and print the model's response
response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(response)
