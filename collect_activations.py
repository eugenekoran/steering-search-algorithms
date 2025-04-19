import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split


MAX_TOKENS = 5_000 
DEBUG = False

# Custom dataset class for graph problems
class GraphProblemDataset(Dataset):
    def __init__(self, data, tokenizer, system_prompt):
        self.data = data
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create prompt from problem text
        prompt = item["problem_text"]
        
        # Create conversation format
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        formatted_input = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        input_ids = self.tokenizer.encode(formatted_input, return_tensors="pt").squeeze(0)
        
        return {
            "input_ids": input_ids,
            "problem_id": item["id"],
            "problem_type": item["problem_type"]
        }

# Collate function to handle batching
def collate_fn(batch):
    # Get max length in the batch
    max_length = max([item["input_ids"].size(0) for item in batch])
    
    # Pad all sequences to max length
    padded_input_ids = []
    problem_ids = []
    problem_types = []
    
    for item in batch:
        input_ids = item["input_ids"]
        pad_length = max_length - input_ids.size(0)
        
        if pad_length > 0:
            # Pad with pad token
            padding = torch.ones(pad_length, dtype=input_ids.dtype) * tokenizer.pad_token_id
            padded_input = torch.cat([padding, input_ids], dim=0)
        else:
            padded_input = input_ids
            
        padded_input_ids.append(padded_input)
        problem_ids.append(item["problem_id"])
        problem_types.append(item["problem_type"])
    
    # Stack all tensors
    return {
        "input_ids": torch.stack(padded_input_ids),
        "problem_ids": problem_ids,
        "problem_types": problem_types
    }

# Load dataset from JSON file
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems from {file_path}")
    return data

# Specify the model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer and model with GPU support
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    padding_side='left'  # Set padding side to left for decoder-only models
)

# Get CoT tokens
THINK_START_TOKEN = tokenizer.convert_tokens_to_ids("<think>")
THINK_END_TOKEN = tokenizer.convert_tokens_to_ids("</think>")
print(f"<think> token ID: {THINK_START_TOKEN}")
print(f"</think> token ID: {THINK_END_TOKEN}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

# Function to create bucketed representations of layer activations
def create_bucketed_representation(hidden_states, selected_layer_indices=None, num_buckets=4):
    """
    Creates a condensed representation of hidden states by:
    1. Selecting specific layers
    2. Dividing each sequence into n equal buckets
    3. Averaging the activations within each bucket
    
    Args:
        hidden_states: List of tensors, each with shape (seq_len, model_dim)
        selected_layer_indices: List of indices for layers to include (defaults to layers 7, 14, 21, 28)
        num_buckets: Number of segments to divide the sequence into (default 4)
        
    Returns:
        Tensor of shape (num_selected_layers, num_buckets, model_dim)
    """
    # Default to layers 7, 14, 21, 28 if not specified
    if selected_layer_indices is None:
        selected_layer_indices = [7, 14, 21, 28]
    
    # Get sequence length and model dimension
    seq_len = hidden_states[0].shape[0]
    model_dim = hidden_states[0].shape[1]
    
    # Initialize the result tensor
    result = torch.zeros(len(selected_layer_indices), num_buckets, model_dim)
    
    # Process each selected layer
    for layer_idx, global_layer_idx in enumerate(selected_layer_indices):
        # Get the layer activations
        layer_activations = hidden_states[global_layer_idx]
        
        # Divide the sequence into buckets
        for bucket in range(num_buckets):
            # Calculate the start and end indices for this bucket
            start_idx = (bucket * seq_len) // num_buckets
            end_idx = ((bucket + 1) * seq_len) // num_buckets
            
            # Handle edge case for the last bucket
            if bucket == num_buckets - 1:
                end_idx = seq_len
            
            result[layer_idx, bucket] = layer_activations[start_idx:end_idx].mean(dim=0)
    
    return result

# Define system prompt
system_prompt = """You are an expert algorithmic assistant specializing in graph problems. Keep your reasoning clear, accurate, and brief.
When the user presents a graph-related question, provide a concise, correct solution."""

# Load dataset
data = load_dataset("dataset.json")

# Create dataset
full_dataset = GraphProblemDataset(data, tokenizer, system_prompt)

# Split into train and test (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
# Use the first 80% for training and the rest for testing (no randomization)
train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create data loaders
batch_size = 8  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Number of batches in train loader: {len(train_loader)}")
print(f"Number of batches in test loader: {len(test_loader)}")

# Function to map problem type to algorithm
def map_problem_to_algorithm(problem_type):
    mapping = {
        "shortest_path": "bfs",
        "path_exists": "dfs"
    }
    return mapping.get(problem_type, problem_type)

# Process data and collect activations
def process_data(data_loader, split_name, max_batches=3):
    results = []
    
    print(f"\nProcessing {split_name} data...")
    for batch_idx, batch in enumerate(data_loader):
        print(f"Processing {split_name} batch {batch_idx+1}/{len(data_loader)}")
        batch_tensor = batch["input_ids"].to(model.device)
        
        with torch.no_grad():
            # First, generate the responses
            outputs = model.generate(
                batch_tensor,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                eos_token_id=[THINK_END_TOKEN],
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Process each example in the batch
            for i, output_seq in enumerate(outputs):
                # Extract the generated tokens (excluding input)
                input_length = batch_tensor[i].size(0)
                generated_part = output_seq[input_length:].cpu()
                full_sequence = output_seq.cpu()  # Store CPU version for saving to disk
                
                # Create input dictionary directly from output_seq (which is already on the device)
                # This includes both the input prompt and the generated response
                inputs = {
                    "input_ids": output_seq.unsqueeze(0),  # Add batch dimension
                    "attention_mask": torch.ones_like(output_seq).unsqueeze(0)
                }
                
                # Extract hidden states for the full sequence (input + output)
                model_outputs = model(**inputs, output_hidden_states=True)
                
                # Get all hidden states and move to CPU for storage
                full_hidden_states = [layer[0].cpu() for layer in model_outputs.hidden_states]
                
                # Find the positions of <think> and </think> tokens in the sequence
                tokens_list = output_seq.tolist()
                think_start_pos = tokens_list.index(THINK_START_TOKEN)
                think_start_pos += 1  # Move to the position after <think>
                
                try:
                    # Position of the </think> token (the hidden state that generated this token)
                    think_end_pos = tokens_list.index(THINK_END_TOKEN)
                    was_truncated = False
                except ValueError:
                    # Output was truncated before </think> was generated
                    print(f"Log: </think> token not found in example {batch['problem_ids'][i]}")
                    # Use the end of the sequence as the ending position
                    think_end_pos = len(tokens_list)
                    was_truncated = True
                
                # Extract only the hidden states for the reasoning part
                # Make sure we have a valid range (think_end_pos could equal the sequence length in case of truncation)
                if think_start_pos < think_end_pos:
                    # Extract reasoning hidden states for each layer
                    reasoning_hidden_states = [
                        layer_states[think_start_pos:think_end_pos]
                        for layer_states in full_hidden_states
                    ]
                    
                    # Also extract the corresponding tokens
                    reasoning_tokens = output_seq[think_start_pos:think_end_pos].cpu()
                    
                    # Print debug info for the first example
                    if not i:
                        print("\nDebug info for first example:")
                        print(f"Full sequence length: {len(tokens_list)}")
                        print(f"Reasoning part: positions {think_start_pos} to {think_end_pos} (length: {think_end_pos - think_start_pos})")
                        if was_truncated:
                            print(f"Note: </think> token not found, using sequence end as the cutoff")
                        print(f"Reasoning tokens decoded: {tokenizer.decode(reasoning_tokens)}")
                else:
                    # Fallback if the reasoning part couldn't be identified properly
                    print(f"Warning: Invalid reasoning boundaries for example {batch['problem_ids'][i]} - using full sequence")
                    reasoning_hidden_states = full_hidden_states
                    reasoning_tokens = output_seq.cpu()
                
                # Map problem type to algorithm
                problem_type = batch["problem_types"][i]
                algorithm = map_problem_to_algorithm(problem_type)
                
                # Store everything we need
                results.append({
                    "problem_id": batch["problem_ids"][i],
                    "problem_type": problem_type,
                    "algorithm": algorithm,
                    "is_train": split_name == "train",
                    "output_ids": generated_part,
                    "reasoning_tokens": reasoning_tokens,  # Store the tokens corresponding to reasoning
                    "full_sequence": full_sequence,
                    "was_truncated": was_truncated,
                    "decoded_output": tokenizer.decode(generated_part, skip_special_tokens=False),
                    "decoded_reasoning": tokenizer.decode(reasoning_tokens, skip_special_tokens=False),
                    # Add the bucketed representation
                    "bucketed_activations": create_bucketed_representation(reasoning_hidden_states)
                })
        
        # For demonstration, process only a few batches
        if DEBUG and batch_idx >= max_batches - 1:
            print(f"Processed {max_batches} {split_name} batches for demonstration.")
            break
            
    return results

# Process both training and testing data
train_results = process_data(train_loader, "train")
test_results = process_data(test_loader, "test")

# Combine results
all_results = train_results + test_results
print(f"Total examples processed: {len(all_results)}")

# Save all data using torch.save
torch.save(all_results, "activations.pt")
print("Activation extraction complete. Data saved to activations.pt")


# Utility function to demonstrate how to retrieve and analyze the stored data
def analyze_stored_data(file_path="activations.pt"):
    """
    Utility function to demonstrate how to retrieve and analyze the stored data.
    
    Args:
        file_path: Path to the PyTorch file containing the stored data
    """
    print("\nDemonstrating how to retrieve and analyze the stored data:")
    
    # Load the saved data
    data = torch.load(file_path)
    
    # Print general information
    print(f"Loaded {len(data)} examples")
    
    # Count problem types and algorithms
    problem_types = {}
    algorithms = {}
    reasoning_lengths = []
    
    for item in data:
        problem_type = item["problem_type"]
        algorithm = item["algorithm"]
        
        problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
        algorithms[algorithm] = algorithms.get(algorithm, 0) + 1
        
        # Calculate reasoning length
        if "reasoning_tokens" in item:
            reasoning_lengths.append(len(item["reasoning_tokens"]))
    
    print("\nProblem type distribution:")
    for problem_type, count in problem_types.items():
        print(f"  - {problem_type}: {count}")
    
    print("\nAlgorithm distribution:")
    for algorithm, count in algorithms.items():
        print(f"  - {algorithm}: {count}")
    
    # Reasoning statistics
    if reasoning_lengths:
        avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths)
        print(f"\nReasoning statistics:")
        print(f"  - Average reasoning length: {avg_reasoning_length:.2f} tokens")
        print(f"  - Min reasoning length: {min(reasoning_lengths)} tokens")
        print(f"  - Max reasoning length: {max(reasoning_lengths)} tokens")
    
    # Count truncated examples
    truncated_count = sum(1 for item in data if item.get("was_truncated", False))
    print(f"\nTruncation statistics:")
    print(f"  - Truncated examples (missing </think>): {truncated_count} out of {len(data)} ({truncated_count/len(data)*100:.1f}%)")
    print(f"  - Complete examples: {len(data) - truncated_count} out of {len(data)} ({(len(data) - truncated_count)/len(data)*100:.1f}%)")
    
    # Analyze a few examples
    num_samples = min(3, len(data))
    print(f"\nSample of {num_samples} examples:")
    
    for i in range(num_samples):
        item = data[i]
        print(f"\nExample {i+1}:")
        print(f"Problem ID: {item['problem_id']}")
        print(f"Problem Type: {item['problem_type']} (Algorithm: {item['algorithm']})")
        print(f"Is Training Example: {item['is_train']}")
        
        # Show reasoning positions and token presence
        if "think_start_pos" in item and "think_end_pos" in item:
            print(f"Reasoning span: positions {item['think_start_pos']} to {item['think_end_pos']}")
            print(f"Reasoning length: {item['think_end_pos'] - item['think_start_pos']} tokens")
            
            # Report on token presence
            if item.get("was_truncated", False):
                print(f"Note: </think> token was not found - reasoning was truncated")
        
        # Show bucketed representation information
        if "bucketed_activations" in item:
            bucketed_activations = item["bucketed_activations"]
            print(f"\nBucketed representation:")
            print(f"  - Shape: {bucketed_activations.shape} (layers, buckets, model_dim)")
        
        # Show reasoning text
        if "decoded_reasoning" in item:
            reasoning_text = item["decoded_reasoning"]
            if len(reasoning_text) > 200:
                reasoning_text = reasoning_text[:200] + "... (truncated)"
            print(f"\nReasoning text:\n{reasoning_text}")


analyze_stored_data()
