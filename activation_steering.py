import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
import os
import json
import random
from datetime import datetime
import re
import openai
import time
from tqdm import tqdm

# Make sure all outputs are displayed
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

def load_probe_and_activations():
    """
    Load the best probe model and the activations dataset.
    
    Returns:
        probe_data: Dict containing the probe model and metadata
        activations_data: List of examples with activations
    """
    print("Loading best probe and activations...")
    probe_data = torch.load("best_probe.pt")
    activations_data = torch.load("activations.pt")
    
    print(f"Loaded probe from layer {probe_data['layer']}, bucket {probe_data['bucket_idx']}")
    print(f"Probe accuracy: {probe_data['accuracy']:.4f}")
    print(f"Loaded {len(activations_data)} examples with activations")
    
    return probe_data, activations_data

def compute_direction_vector(probe_data):
    """
    Use the binary probe weights as the direction vector for algorithm steering.
    
    Args:
        probe_data: Dict containing the probe model and metadata
        
    Returns:
        unit_direction_vector: Unit vector based on probe coefficients
    """
    # Extract the probe model
    probe_model = probe_data['model']
    
    # For binary classification, the coefficients are directly the direction vector
    direction_vector = probe_model.coef_[0]
    
    # Normalize to unit length
    norm = np.linalg.norm(direction_vector)
    unit_direction_vector = direction_vector / norm
    
    print(f"Direction vector from probe coefficients with magnitude: {norm:.4f}")
    print(f"Normalized to unit vector with length: {np.linalg.norm(unit_direction_vector):.4f}")
    
    return unit_direction_vector

def load_model_and_tokenizer():
    """Load the same model used in the original experiment."""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side='left'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    
    # Get the special tokens
    THINK_START_TOKEN = tokenizer.convert_tokens_to_ids("<think>")
    THINK_END_TOKEN = tokenizer.convert_tokens_to_ids("</think>")
    
    return model, tokenizer, THINK_START_TOKEN, THINK_END_TOKEN

def create_algorithm_steering_hook(direction_vector, scale=1.0, start_percent=0.0, end_percent=1.0):
    """
    Create a hook function that adds a scaled direction vector to activations for algorithm steering.
    
    Args:
        direction_vector: numpy array of shape (hidden_dim,)
        scale: Scaling factor for the direction vector
        start_percent: Start position as a percentage of the sequence length
        end_percent: End position as a percentage of the sequence length
    
    Returns:
        hook_fn: A hook function to register with PyTorch
    """
    # Convert direction vector to a tensor but don't specify the dtype yet
    # We'll convert it to the correct dtype inside the hook
    direction_tensor = torch.tensor(direction_vector)
    
    def steering_hook(module, input_tensors, output):
        # Process the output which could be a tuple
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Get the shape of the hidden states
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Critical: Convert direction tensor to the SAME device AND dtype as hidden_states
        # Use the to() method with both device and dtype specified
        direction = direction_tensor.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Make a copy to avoid in-place modification
        steered_hidden_states = hidden_states.clone()
        
        # Calculate start and end positions
        start_pos = int(seq_len * start_percent)
        end_pos = int(seq_len * end_percent)
        end_pos = min(end_pos, seq_len)  # Ensure we don't go beyond the sequence
        
        if start_pos >= end_pos:
            print(f"Warning: Invalid steering range {start_pos} to {end_pos} for sequence length {seq_len}")
            return output
        
        # Create a mask for the positions we want to modify
        # Explicitly convert to the same dtype as hidden_states
        position_mask = torch.zeros(seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        position_mask[start_pos:end_pos] = 1.0
        
        # Expand mask to match output dimensions [batch_size, seq_len, 1]
        mask_expanded = position_mask.view(1, seq_len, 1).expand(batch_size, -1, 1)
        
        # Add scaled direction vector only to specified positions
        # Ensure the direction vector is properly shaped and has the same dtype
        direction_expanded = direction.view(1, 1, hidden_dim)
        
        # Apply the direction with the mask
        steered_hidden_states = steered_hidden_states + scale * mask_expanded * direction_expanded
        
        # Return the steered output, maintaining the original tuple structure if needed
        if isinstance(output, tuple):
            return (steered_hidden_states,) + output[1:]
        else:
            return steered_hidden_states
    
    return steering_hook

def test_algorithm_steering(example, direction_vector, probe_data, scale=1.0, 
                           start_percent=0.0, end_percent=1.0, model=None, tokenizer=None,
                           verbose=True):
    """
    Test algorithm steering on a given example.
    
    Args:
        example: An example from the activations dataset
        direction_vector: The algorithm direction vector
        probe_data: Dict containing the probe model and metadata
        scale: Scaling factor for the direction vector (default: 1.0)
        start_percent: Start of intervention as percentage of sequence (default: 0.0)
        end_percent: End of intervention as percentage of sequence (default: 1.0)
        model: The model to use (if None, will be loaded)
        tokenizer: The tokenizer to use (if None, will be loaded)
        verbose: Whether to print detailed output (default: True)
        
    Returns:
        result_dict: Dictionary containing test results
    """
    if model is None or tokenizer is None:
        model, tokenizer, THINK_START_TOKEN, THINK_END_TOKEN = load_model_and_tokenizer()
    else:
        # Get the special tokens if model and tokenizer are provided
        THINK_START_TOKEN = tokenizer.convert_tokens_to_ids("<think>")
        THINK_END_TOKEN = tokenizer.convert_tokens_to_ids("</think>")
    
    # Extract necessary information from probe_data
    layer_idx = probe_data['layer_idx']
    actual_layer = probe_data['layer']
    
    # Extract problem details
    problem_id = example["problem_id"]
    problem_type = example["problem_type"]
    algorithm = example["algorithm"]
    
    if verbose:
        print(f"\n===== Testing Activation Steering =====")
        print(f"Problem ID: {problem_id}")
        print(f"Problem Type: {problem_type}")
        print(f"Algorithm: {algorithm}")
        print(f"Intervention: Scale={scale}, Range={start_percent*100:.1f}%-{end_percent*100:.1f}%")
    
    # Create the steering hook
    steering_hook = create_algorithm_steering_hook(
        direction_vector, 
        scale=scale,
        start_percent=start_percent,
        end_percent=end_percent
    )
    
    # Identify the target layer in the model
    # Map layer index to actual model layer
    layer_mapping = {0: 7, 1: 14, 2: 21, 3: 28}
    target_layer_index = layer_mapping[layer_idx]
    
    # Find the appropriate module to hook
    target_module = model.model.layers[target_layer_index]
    
    # Extract the original problem statement from full_sequence
    full_sequence = example["full_sequence"]
    
    # Find the position of the <think> token in the sequence
    think_pos = -1
    for i, token_id in enumerate(full_sequence):
        if token_id == THINK_START_TOKEN:
            think_pos = i
            break
    
    # Extract and decode the problem part (everything before <think>)
    if think_pos > 0:
        problem_sequence = full_sequence[:think_pos]
        problem_text = tokenizer.decode(problem_sequence, skip_special_tokens=False)
    else:
        # Fallback if <think> token not found
        problem_text = example["decoded_output"].split("<think>")[0].strip()
    
    # Get the original output directly from the decoded_reasoning key
    original_output = example["decoded_reasoning"]
    
    # Format the input for generation
    system_prompt = "You are an expert algorithmic assistant specializing in graph problems. Keep your reasoning clear, accurate, and brief. When the user presents a graph-related question, provide a concise, correct solution."
    
    # Create conversation format for the prompt
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text}
    ]
    
    # Apply chat template
    formatted_input = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    input_ids = tokenizer.encode(formatted_input, return_tensors="pt").to(model.device)
    
    # Skip generating original output - we already have it
    if verbose:
        print("\nUsing existing original output from dataset...")
    
    # Register the hook for steering
    hook_handle = target_module.register_forward_hook(steering_hook)
    
    # Generate with steering
    max_tokens = 100
    if verbose:
        print("\nGenerating steered output...")
    with torch.no_grad():
        steered_output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=[tokenizer.convert_tokens_to_ids("</think>")],
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Get the raw steered output
    steered_output = tokenizer.decode(steered_output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
    
    # Calculate the character positions for intervention markers
    # This is an approximation based on the percentage of the text
    text_length = len(steered_output)
    start_char_pos = int(text_length * start_percent)
    end_char_pos = int(text_length * end_percent)
    
    # Insert markers at the calculated positions
    marked_output = (
        steered_output[:start_char_pos] + 
        "【START_STEER】" + 
        steered_output[start_char_pos:end_char_pos] + 
        "【END_STEER】" + 
        steered_output[end_char_pos:]
    )
    
    # Remove the hook
    hook_handle.remove()
    
    if verbose:
        # Print the results
        print("\n===== ORIGINAL PROBLEM =====")
        print(problem_text)
        
        print("\n===== ORIGINAL SOLUTION =====")
        print(original_output)
        
        print("\n===== STEERED SOLUTION (with intervention markers) =====")
        print(marked_output)
        
        # Print the intervention boundaries for clarity
        print(f"\nIntervention applied at approximately characters {start_char_pos} to {end_char_pos} (out of {text_length} characters)")
    
    # Create a result dictionary
    result = {
        "problem_id": problem_id,
        "problem_type": problem_type,
        "algorithm": algorithm,
        "scale": scale,
        "start_percent": start_percent,
        "end_percent": end_percent,
        "problem_text": problem_text,
        "original_output": original_output,
        "steered_output": steered_output,
        "marked_output": marked_output,
        "intervention_start_char": start_char_pos,
        "intervention_end_char": end_char_pos
    }
    
    return result

def run_batch_tests(num_examples=5, scale=10.0, start_percent=0.0, end_percent=1.0):
    """
    Run algorithm steering tests on multiple examples.
    
    Args:
        num_examples: Number of examples to test for each algorithm (default: 5)
        scale: Scaling factor for the direction vector (default: 10.0)
        start_percent: Start of intervention as percentage of sequence (default: 0.0)
        end_percent: End of intervention as percentage of sequence (default: 1.0)
        
    Returns:
        results: List of dictionaries containing test results
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load the probe and activations
    probe_data, activations_data = load_probe_and_activations()
    
    # Compute the algorithm direction vector
    direction_vector = compute_direction_vector(probe_data)
    
    # Load model and tokenizer once
    model, tokenizer, THINK_START_TOKEN, THINK_END_TOKEN = load_model_and_tokenizer()
    
    # Separate examples by algorithm
    dfs_examples = [item for item in activations_data if item["algorithm"] == "dfs"]
    bfs_examples = [item for item in activations_data if item["algorithm"] == "bfs"]
    
    print(f"Found {len(dfs_examples)} DFS examples and {len(bfs_examples)} BFS examples")
    
    # Select random examples if we have more than requested
    if len(dfs_examples) > num_examples:
        dfs_examples = random.sample(dfs_examples, num_examples)
    if len(bfs_examples) > num_examples:
        bfs_examples = random.sample(bfs_examples, num_examples)
    
    # Prepare test configurations
    test_configs = [
        # DFS examples with positive direction (steering toward BFS)
        {"examples": dfs_examples, "direction": direction_vector, "name": "dfs_to_bfs"},
        # BFS examples with negative direction (steering toward DFS)
        {"examples": bfs_examples, "direction": -direction_vector, "name": "bfs_to_dfs"}
    ]
    
    all_results = []
    
    # Run tests for each configuration
    for config in test_configs:
        print(f"\n\n===== Running tests: {config['name']} =====")
        
        for i, example in enumerate(config["examples"]):
            print(f"\nTesting example {i+1}/{len(config['examples'])}: {example['problem_id']}")
            
            # Run the test
            result = test_algorithm_steering(
                example,
                config["direction"],
                probe_data,
                scale=scale,
                start_percent=start_percent,
                end_percent=end_percent,
                model=model,
                tokenizer=tokenizer,
                verbose=True  # Set to False to reduce output
            )
            
            # Add test configuration to result
            result["test_name"] = config["name"]
            result["test_index"] = i
            
            # Add to results list
            all_results.append(result)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/steering_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    for result in all_results:
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSaved {len(all_results)} test results to {results_file}")
    
    return all_results

def analyze_results(results):
    """
    Analyze and visualize the results of algorithm steering tests.
    
    Args:
        results: List of dictionaries containing test results
    """
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Group results by test name
    test_groups = {}
    for result in results:
        test_name = result["test_name"]
        if test_name not in test_groups:
            test_groups[test_name] = []
        test_groups[test_name].append(result)
    
    # Print summary statistics
    print("\n===== Results Summary =====")
    for test_name, group in test_groups.items():
        print(f"\nTest: {test_name}")
        print(f"Number of examples: {len(group)}")
        
    plt.figure(figsize=(10, 6))
    
    for test_name, group in test_groups.items():
        original_lengths = [len(result["original_output"]) for result in group]
        steered_lengths = [len(result["steered_output"]) for result in group]
        
        plt.scatter(original_lengths, steered_lengths, label=test_name, alpha=0.7)
    
    plt.plot([0, max(max(len(r["original_output"]) for r in results), 
                     max(len(r["steered_output"]) for r in results))], 
             [0, max(max(len(r["original_output"]) for r in results), 
                     max(len(r["steered_output"]) for r in results))], 
             'k--', alpha=0.3)
    
    plt.xlabel("Original Output Length (chars)")
    plt.ylabel("Steered Output Length (chars)")
    plt.title("Comparison of Output Lengths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("visualizations/output_length_comparison.png")
    plt.close()
    
    print("\nSaved output length comparison visualization")

def create_gpt4_validation_prompts(results_file=None, results=None):
    """
    Create validation prompts for GPT-4o to analyze if algorithm steering
    successfully steered the model's behavior.
    
    Args:
        results_file: Path to the JSON file containing test results
        results: List of dictionaries containing test results (alternative to results_file)
        
    Returns:
        prompts: List of dictionaries containing prompts for GPT-4o
    """
    # Load results if file path is provided
    if results_file and not results:
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    if not results:
        raise ValueError("Either results_file or results must be provided")
    
    # Create prompts directory if it doesn't exist
    os.makedirs('prompts', exist_ok=True)
    
    # System prompt template with explicit instructions for response format
    system_prompt = """You are an expert in analyzing language model outputs, particularly focusing on algorithmic reasoning. 
Your task is to analyze two different solutions to the same graph problem and determine:
1. Which algorithm is being used in each solution (e.g., BFS, DFS, Dijkstra's, etc.)
2. Whether there's evidence that the second solution has been steered toward a different algorithm than the first
3. Specific evidence in the text that supports your conclusion

First, think through your analysis in detail. Then, provide your final conclusion in this exact format on a new line:

CONCLUSION: [YES/PARTIAL/NO] - The second solution [was/was partially/was not] steered toward the target algorithm.

Where:
- YES means clear evidence of successful steering toward the target algorithm
- PARTIAL means some evidence of steering but mixed with elements of the original algorithm
- NO means no clear evidence of steering toward the target algorithm"""

    # User prompt template
    user_prompt_template = """I'm studying how language models approach graph problems. I have a problem and two different solutions from the same model.

Problem:
{problem_text}

Original Solution (using {original_algorithm}):
{original_output}

Modified Solution (potentially steered toward {target_algorithm}):
{steered_output}

Please analyze both solutions:
1. What algorithm is being used in each solution?
2. Is there evidence the second solution was steered toward {target_algorithm}?
3. What specific parts of the text support your conclusion?

Remember to first think through your analysis, then provide your final conclusion in the specified format on a new line."""

    prompts = []
    
    for i, result in enumerate(results):
        # Determine the target algorithm
        original_algorithm = result["algorithm"].upper()
        target_algorithm = "BFS" if original_algorithm == "DFS" else "DFS"
        
        # Preprocess the problem text to remove special tokens
        problem_text = result["problem_text"]
        
        # Remove "
        while problem_text.startswith(""):
            problem_text = problem_text[len(""):].lstrip()
        
        # Remove "
        if problem_text.endswith(""):
            problem_text = problem_text[:-len(""):].rstrip()
        
        # Truncate outputs to approximately 750 words
        def truncate_to_words(text, max_words=750):
            words = text.split()
            if len(words) <= max_words:
                return text
            return ' '.join(words[:max_words]) + "... [truncated]"
        
        original_output = truncate_to_words(result["original_output"])
        steered_output = truncate_to_words(result["steered_output"])
        
        # Format the user prompt
        user_prompt = user_prompt_template.format(
            problem_text=problem_text,
            original_algorithm=original_algorithm,
            original_output=original_output,
            target_algorithm=target_algorithm,
            steered_output=steered_output
        )
        
        # Create the prompt dictionary
        prompt = {
            "id": f"{result['test_name']}_{result['test_index']}",
            "problem_id": result["problem_id"],
            "test_name": result["test_name"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "original_algorithm": original_algorithm,
            "target_algorithm": target_algorithm,
            "scale": result["scale"]
        }
        
        prompts.append(prompt)
    
    # Save prompts to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_file = f"prompts/validation_prompts_{timestamp}.json"
    
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\nSaved {len(prompts)} validation prompts to {prompts_file}")
    
    # Also save individual prompt files for easy copying
    for prompt in prompts:
        prompt_id = prompt["id"]
        individual_file = f"prompts/prompt_{prompt_id}.txt"
        
        with open(individual_file, 'w') as f:
            f.write(f"System Prompt:\n{prompt['system_prompt']}\n\n")
            f.write(f"User Prompt:\n{prompt['user_prompt']}")
        
    print(f"Saved individual prompt files to the prompts directory")
    
    return prompts

def analyze_gpt4_responses(responses_file):
    """
    Analyze the responses from GPT-4o to determine if algorithm steering
    successfully steered the model's behavior.
    
    Args:
        responses_file: Path to the JSON file containing GPT-4o responses
        
    Returns:
        analysis: Dictionary containing analysis results
    """
    with open(responses_file, 'r') as f:
        responses = json.load(f)
    
    # Initialize counters
    total = len(responses)
    successful_steering = 0
    partial_steering = 0
    no_steering = 0
    
    # Analyze each response
    for response in responses:
        gpt4_response = response.get("gpt4_response", "")
        
        # Look for the conclusion line
        conclusion_match = re.search(r"CONCLUSION:\s*(\w+)", gpt4_response)
        if conclusion_match:
            conclusion = conclusion_match.group(1).upper()
            
            if conclusion == "YES":
                successful_steering += 1
            elif conclusion == "PARTIAL":
                partial_steering += 1
            elif conclusion == "NO":
                no_steering += 1
            else:
                print(f"Warning: Unrecognized conclusion format: {conclusion}")
        else:
            print(f"Warning: Could not find conclusion in response for {response.get('id', 'unknown')}")
            # Try to analyze the full text as a fallback
            if "clear evidence" in gpt4_response.lower() or "successfully steered" in gpt4_response.lower():
                successful_steering += 1
            elif "some evidence" in gpt4_response.lower() or "partially steered" in gpt4_response.lower():
                partial_steering += 1
            else:
                no_steering += 1
    
    # Calculate percentages
    success_percent = (successful_steering / total) * 100 if total > 0 else 0
    partial_percent = (partial_steering / total) * 100 if total > 0 else 0
    no_effect_percent = (no_steering / total) * 100 if total > 0 else 0
    
    # Create analysis dictionary
    analysis = {
        "total_examples": total,
        "successful_steering": successful_steering,
        "partial_steering": partial_steering,
        "no_steering": no_steering,
        "success_percent": success_percent,
        "partial_percent": partial_percent,
        "no_effect_percent": no_effect_percent
    }
    
    # Print summary
    print("\n===== GPT-4o Validation Analysis =====")
    print(f"Total examples analyzed: {total}")
    print(f"Successful steering: {successful_steering} ({success_percent:.1f}%)")
    print(f"Partial steering: {partial_steering} ({partial_percent:.1f}%)")
    print(f"No evidence of steering: {no_steering} ({no_effect_percent:.1f}%)")
    
    return analysis

def submit_prompts_to_openai(prompts, api_key=None, model="gpt-4o", max_retries=3, retry_delay=5):
    """
    Submit prompts to OpenAI API and collect responses.
    
    Args:
        prompts: List of dictionaries containing prompts
        api_key: OpenAI API key (if None, will look for OPENAI_API_KEY environment variable)
        model: OpenAI model to use (default: "gpt-4o")
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay between retries in seconds
        
    Returns:
        responses: List of dictionaries containing prompts and responses
    """
    # Set up OpenAI API key
    if api_key:
        openai.api_key = api_key
    elif "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        raise ValueError("OpenAI API key must be provided either as an argument or as OPENAI_API_KEY environment variable")
    
    # Initialize OpenAI client
    client = openai.OpenAI()
    
    responses = []
    
    print(f"\nSubmitting {len(prompts)} prompts to OpenAI API ({model})...")
    
    # Process each prompt
    for prompt in tqdm(prompts):
        # Prepare messages for the API
        messages = [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_prompt"]}
        ]
        
        # Initialize response
        api_response = None
        
        # Try with retries
        for attempt in range(max_retries):
            try:
                # Call the OpenAI API
                api_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,  # Use deterministic responses
                    max_tokens=1500   # Limit response length
                )
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{max_retries} for prompt {prompt['id']}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to get response for prompt {prompt['id']} after {max_retries} attempts")
        
        # If we got a response, process it
        if api_response:
            # Extract the response content
            response_content = api_response.choices[0].message.content
            
            # Create response dictionary
            response = {
                "id": prompt["id"],
                "problem_id": prompt["problem_id"],
                "test_name": prompt["test_name"],
                "original_algorithm": prompt["original_algorithm"],
                "target_algorithm": prompt["target_algorithm"],
                "scale": prompt["scale"],
                "gpt4_response": response_content
            }
            
            # Extract conclusion if present
            conclusion_match = re.search(r"CONCLUSION:\s*(\w+)", response_content)
            if conclusion_match:
                response["conclusion"] = conclusion_match.group(1).upper()
            
            # Add to responses list
            responses.append(response)
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
    
    # Save responses to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_file = f"results/gpt4_responses_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    with open(responses_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"\nSaved {len(responses)} GPT-4o responses to {responses_file}")
    
    return responses

# Update the main execution to include the OpenAI submission
if __name__ == "__main__":
    # Run batch tests with 5 examples per algorithm
    results = run_batch_tests(num_examples=5, scale=10.0)
    
    # Analyze the results
    analyze_results(results)
    
    # Create validation prompts for GPT-4o
    prompts = create_gpt4_validation_prompts(results=results)
    
    # Submit prompts to OpenAI
    responses = submit_prompts_to_openai(prompts)
    
    # Analyze the responses
    analysis = analyze_gpt4_responses(f"results/gpt4_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
