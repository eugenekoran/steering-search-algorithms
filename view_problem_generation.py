#!/usr/bin/env python3
"""
Example script demonstrating how to use the graph problem generator.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt

from graph_problem_generator.graph_generator import generate_random_graph, visualize_graph
from graph_problem_generator.problem_generator import ProblemGenerator


def main():
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Generate a random graph with minimum path length
    print("Generating a random graph...")
    min_nodes = 10
    max_nodes = 15
    min_path_length = 4
    
    G, source, target = generate_random_graph(
        min_nodes=min_nodes, 
        max_nodes=max_nodes,
        min_path_length=min_path_length
    )
    
    # Visualize the graph
    print("Visualizing the graph...")
    visualize_graph(G, source=source, target=target, filename="example_graph.png")
    print("Graph visualization saved to example_graph.png")
    
    # Generate problems
    print("Generating problems...")
    problem_generator = ProblemGenerator(api_key=api_key)
    
    # Generate one problem of each type
    problems = []
    for problem_type in ["shortest_path", "path_exists"]:
        problem = problem_generator.generate_problem_with_answer(G, source, target, problem_type)
        problem["id"] = f"example_{problem_type}"
        problems.append(problem)
    
    # Save problems to file
    with open("example_problems.json", "w") as f:
        json.dump(problems, f, indent=2)
    
    # Print problems
    print("\nGenerated Problems:")
    for problem in problems:
        print(f"\nProblem ID: {problem['id']}")
        print(f"Problem Type: {problem['problem_type']}")
        print(f"Problem Text: {problem['problem_text']}")
        print(f"Answer: {problem['answer']}")
        
    # Print graph information
    print("\nGraph Information:")
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of edges: {len(G.edges())}")
    print(f"Source node: {source}")
    print(f"Target node: {target}")
    
    # Calculate and print the shortest path
    path, length = nx.single_source_dijkstra(G, source, target)
    print(f"Shortest path: {' -> '.join(path)}")
    print(f"Path length: {length}")


if __name__ == "__main__":
    main() 