"""
Problem Generator

This module provides functionality to generate logical problem text from graphs using OpenAI's API.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
from openai import OpenAI

from .graph_generator import graph_to_dict, get_shortest_path, check_path_exists


class ProblemGenerator:
    """Class for generating logical problems from graphs using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, client: Optional[Any] = None):
        """
        Initialize the problem generator.
        
        Args:
            api_key: OpenAI API key. If None, it will be read from the OPENAI_API_KEY environment variable.
            client: OpenAI client instance. If provided, api_key is ignored.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key and client is None:
            raise ValueError("OpenAI API key is required. Set it as an environment variable or pass it to the constructor.")
        
        self.client = client if client is not None else OpenAI(api_key=self.api_key)
    
    def generate_problem_text(self, G: nx.Graph, source: str, target: str, problem_type: str) -> str:
        """
        Generate a logical problem text from a graph.
        
        Args:
            G: The networkx Graph
            source: The source node
            target: The target node
            problem_type: Type of problem ('shortest_path' or 'path_exists')
            
        Returns:
            A string containing the problem text
        """
        # Convert graph to a format that can be included in the prompt
        graph_desc = self._format_graph_for_prompt(G)
        
        # Create prompt based on problem type
        if problem_type == "shortest_path":
            prompt = self._create_shortest_path_prompt(graph_desc, source, target)
        elif problem_type == "path_exists":
            prompt = self._create_path_exists_prompt(graph_desc, source, target)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a logical problem generator. Your task is to create clear and concise graph theory problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=1000
            )
            
            problem_text = response.choices[0].message.content.strip()
            return problem_text
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Return a basic problem text as fallback
            return self._create_fallback_problem(G, source, target, problem_type)
    
    def _format_graph_for_prompt(self, G: nx.Graph) -> str:
        """Format graph as a string for inclusion in the prompt."""
        nodes = list(G.nodes())
        edges = []
        
        for u, v in G.edges():
            edges.append(f"{u} -- {v}")
        
        graph_desc = "Nodes: " + ", ".join(nodes) + "\n"
        graph_desc += "Edges:\n" + "\n".join(edges)
        
        return graph_desc
    
    def _create_shortest_path_prompt(self, graph_desc: str, source: str, target: str) -> str:
        """Create a prompt for a shortest path problem."""
        return f"""
Create a graph theory problem based on the following undirected graph:

{graph_desc}

The problem should ask to find the shortest path (minimum number of edges) from node {source} to node {target}.

The problem should:
1. Use standard graph theory terminology (nodes, edges, paths, etc.)
2. Be clear and concise
3. Explicitly state that the graph is undirected
4. Ask for the shortest path from node {source} to node {target}

Return only the problem statement, without any explanations or solutions.
"""
    
    def _create_path_exists_prompt(self, graph_desc: str, source: str, target: str) -> str:
        """Create a prompt for a path existence problem."""
        return f"""
Create a graph theory problem based on the following undirected graph:

{graph_desc}

The problem should ask to determine if there exists a path from node {source} to node {target}.

The problem should:
1. Use standard graph theory terminology (nodes, edges, paths, etc.)
2. Be clear and concise
3. Explicitly state that the graph is undirected
4. Ask if there exists a path from node {source} to node {target}

Return only the problem statement, without any explanations or solutions.
"""
    
    def _create_fallback_problem(self, G: nx.Graph, source: str, target: str, problem_type: str) -> str:
        """Create a basic problem text as fallback when API call fails."""
        nodes_str = ", ".join(G.nodes())
        edges_str = ", ".join([f"{u}-{v}" for u, v in G.edges()])
        
        if problem_type == "shortest_path":
            return f"Given an undirected graph with nodes {nodes_str} and edges {edges_str}, find the shortest path from node {source} to node {target}."
        else:
            return f"Given an undirected graph with nodes {nodes_str} and edges {edges_str}, determine if there exists a path from node {source} to node {target}."
    
    def generate_problem_with_answer(self, G: nx.Graph, source: str, target: str, problem_type: str) -> Dict[str, Any]:
        """
        Generate a complete problem with text and answer.
        
        Args:
            G: The networkx Graph
            source: The source node
            target: The target node
            problem_type: Type of problem ('shortest_path' or 'path_exists')
            
        Returns:
            A dictionary containing the problem details
        """
        # Generate problem text
        problem_text = self.generate_problem_text(G, source, target, problem_type)
        
        # Compute answer
        if problem_type == "shortest_path":
            path, distance = get_shortest_path(G, source, target)
            if path:
                answer = f"The shortest path from {source} to {target} is {' -> '.join(path)} with {distance} edges."
            else:
                answer = f"There is no path from {source} to {target}."
        else:  # path_exists
            path_exists = check_path_exists(G, source, target)
            if path_exists:
                answer = f"Yes, there exists a path from {source} to {target}."
            else:
                answer = f"No, there does not exist a path from {source} to {target}."
        
        return {
            "graph": graph_to_dict(G),
            "problem_text": problem_text,
            "problem_type": problem_type,
            "source_node": source,
            "target_node": target,
            "answer": answer
        } 