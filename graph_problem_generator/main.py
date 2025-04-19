"""
Main module for the graph problem generator.

This module provides the command-line interface for generating logical problems.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
from tqdm import tqdm
from dotenv import load_dotenv

from .graph_generator import generate_random_graph
from .problem_generator import ProblemGenerator

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate logical problems based on graph theory."
    )
    
    parser.add_argument(
        "--num-problems",
        type=int,
        default=10,
        help="Number of problems to generate (default: 10)"
    )
    
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=10,
        help="Minimum number of nodes in each graph (default: 10)"
    )
    
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=20,
        help="Maximum number of nodes in each graph (default: 20)"
    )
    
    parser.add_argument(
        "--min-path-length",
        type=int,
        default=4,
        help="Minimum path length between source and target nodes (default: 4)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.json",
        help="Path to the output file (default: dataset.json)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)"
    )
    
    return parser.parse_args()


def generate_dataset(
    num_problems: int,
    min_nodes: int,
    max_nodes: int,
    min_path_length: int,
    api_key: str = None
) -> List[Dict[str, Any]]:
    """
    Generate a dataset of logical problems.
    
    Args:
        num_problems: Number of problems to generate
        min_nodes: Minimum number of nodes in each graph
        max_nodes: Maximum number of nodes in each graph
        min_path_length: Minimum path length between source and target nodes
        api_key: OpenAI API key
        
    Returns:
        A list of problems
    """
    problem_generator = ProblemGenerator(api_key=api_key)
    dataset = []
    
    # Each graph will generate two problems (shortest path and path exists)
    num_graphs = (num_problems + 1) // 2
    
    for i in tqdm(range(num_graphs), desc="Generating problems"):
        # Generate a random graph with minimum path length
        G, source, target = generate_random_graph(
            min_nodes=min_nodes, 
            max_nodes=max_nodes,
            min_path_length=min_path_length
        )
        
        # Generate two problems for this graph
        for problem_type in ["shortest_path", "path_exists"]:
            if len(dataset) >= num_problems:
                break
                
            problem = problem_generator.generate_problem_with_answer(G, source, target, problem_type)
            problem["id"] = f"problem_{len(dataset) + 1}"
            dataset.append(problem)
    
    return dataset


def main():
    """Main entry point for the command-line interface."""
    args = parse_args()
    
    # Check if OpenAI API key is available
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key is required. Set it as an environment variable or pass it with --api-key.")
        sys.exit(1)
    
    print(f"Generating {args.num_problems} logical problems...")
    dataset = generate_dataset(
        num_problems=args.num_problems,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        min_path_length=args.min_path_length,
        api_key=api_key
    )
    
    # Save dataset to file
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main() 