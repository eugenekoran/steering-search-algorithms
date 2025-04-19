"""
Graph Generator

This module provides functionality to generate random graphs for logical problems.
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional


def generate_random_graph(min_nodes: int = 10, max_nodes: int = 20, min_path_length: int = 4) -> Tuple[nx.Graph, str, str]:
    """
    Generate a random connected graph with a random number of nodes between min_nodes and max_nodes,
    and ensure there's a path of at least min_path_length between two nodes.
    
    Args:
        min_nodes: Minimum number of nodes in the graph (default: 10)
        max_nodes: Maximum number of nodes in the graph
        min_path_length: Minimum path length between source and target nodes (default: 4)
        
    Returns:
        A tuple containing:
        - The networkx Graph object
        - The source node
        - The target node
    """
    # Ensure min_nodes is at least 10
    min_nodes = max(min_nodes, 10)
    
    # Ensure min_path_length is at least 1
    min_path_length = max(min_path_length, 1)
    
    # Ensure max_nodes is greater than min_nodes and can accommodate min_path_length
    max_nodes = max(max_nodes, min_nodes + min_path_length)
    
    # Determine number of nodes
    num_nodes = random.randint(min_nodes, max_nodes)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes with labels A, B, C, etc.
    node_labels = [chr(65 + i) for i in range(min(num_nodes, 26))]
    if num_nodes > 26:
        # If we need more than 26 nodes, use A1, A2, etc.
        node_labels.extend([f"{chr(65 + i // 26)}{i % 26 + 1}" for i in range(26, num_nodes)])
    
    G.add_nodes_from(node_labels)
    
    # First, create a path of at least min_path_length
    path_nodes = random.sample(node_labels, min_path_length + 1)
    source_node = path_nodes[0]
    target_node = path_nodes[-1]
    
    # Add edges to create the path
    for i in range(len(path_nodes) - 1):
        G.add_edge(path_nodes[i], path_nodes[i + 1])
    
    # Add random edges to ensure the graph is connected
    # First, create a random spanning tree to ensure connectivity
    nodes = list(G.nodes())
    connected_nodes = set(path_nodes)  # Start with the path nodes as connected
    unconnected_nodes = [n for n in nodes if n not in connected_nodes]
    
    while unconnected_nodes:
        node1 = random.choice(list(connected_nodes))
        node2 = random.choice(unconnected_nodes)
        G.add_edge(node1, node2)
        connected_nodes.add(node2)
        unconnected_nodes.remove(node2)
    
    # Add some additional random edges, but be careful not to create shortcuts in the path
    num_extra_edges = random.randint(0, num_nodes)
    for _ in range(num_extra_edges):
        node1 = random.choice(nodes)
        potential_nodes = [n for n in nodes if n != node1 and not G.has_edge(node1, n)]
        
        # Filter out nodes that would create a shortcut in the path
        if node1 in path_nodes:
            idx = path_nodes.index(node1)
            # Don't connect to nodes that are more than 1 step away in the path
            for i in range(len(path_nodes)):
                if abs(i - idx) > 1 and path_nodes[i] in potential_nodes:
                    potential_nodes.remove(path_nodes[i])
        
        if potential_nodes:  # Only proceed if there are potential nodes to connect to
            node2 = random.choice(potential_nodes)
            G.add_edge(node1, node2)
    
    # Verify that the shortest path length is at least min_path_length
    actual_path_length = len(nx.shortest_path(G, source=source_node, target=target_node)) - 1
    
    # If the path is too short, try again
    if actual_path_length < min_path_length:
        return generate_random_graph(min_nodes, max_nodes, min_path_length)
    
    return G, source_node, target_node


def visualize_graph(G: nx.Graph, source: Optional[str] = None, target: Optional[str] = None, filename: Optional[str] = None) -> None:
    """
    Visualize a graph and optionally save it to a file.
    
    Args:
        G: The networkx Graph to visualize
        source: The source node (will be highlighted)
        target: The target node (will be highlighted)
        filename: If provided, save the visualization to this file
    """
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_colors = ['lightblue'] * len(G.nodes())
    
    # Highlight source and target nodes if provided
    if source and source in G:
        source_idx = list(G.nodes()).index(source)
        node_colors[source_idx] = 'green'
    
    if target and target in G:
        target_idx = list(G.nodes()).index(target)
        node_colors[target_idx] = 'red'
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    
    plt.axis('off')
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_shortest_path(G: nx.Graph, source: str, target: str) -> Tuple[List[str], int]:
    """
    Find the shortest path between two nodes in a graph.
    
    Args:
        G: The networkx Graph
        source: The source node
        target: The target node
        
    Returns:
        A tuple containing the path (list of nodes) and the total distance (number of edges)
    """
    try:
        path = nx.shortest_path(G, source=source, target=target)
        distance = len(path) - 1  # Number of edges in the path
        return path, distance
    except nx.NetworkXNoPath:
        return [], -1


def check_path_exists(G: nx.Graph, source: str, target: str) -> bool:
    """
    Check if a path exists between two nodes in a graph.
    
    Args:
        G: The networkx Graph
        source: The source node
        target: The target node
        
    Returns:
        True if a path exists, False otherwise
    """
    return nx.has_path(G, source, target)


def select_problem_nodes(G: nx.Graph) -> Tuple[str, str]:
    """
    Select two nodes from the graph to use in a problem.
    
    Args:
        G: The networkx Graph
        
    Returns:
        A tuple containing the source and target nodes
    """
    nodes = list(G.nodes())
    source = random.choice(nodes)
    
    # Make sure target is different from source and there is a path
    connected_nodes = [n for n in nodes if n != source and nx.has_path(G, source, n)]
    
    if not connected_nodes:
        # If no connected nodes, just pick a different node
        remaining_nodes = [n for n in nodes if n != source]
        target = random.choice(remaining_nodes) if remaining_nodes else source
    else:
        target = random.choice(connected_nodes)
    
    return source, target


def graph_to_dict(G: nx.Graph) -> Dict[str, Any]:
    """
    Convert a networkx Graph to a dictionary representation.
    
    Args:
        G: The networkx Graph
        
    Returns:
        A dictionary representation of the graph
    """
    return {
        "nodes": list(G.nodes()),
        "edges": [
            {
                "source": u,
                "target": v
            }
            for u, v in G.edges()
        ]
    }


def dict_to_graph(graph_dict: Dict[str, Any]) -> nx.Graph:
    """
    Convert a dictionary representation to a networkx Graph.
    
    Args:
        graph_dict: Dictionary representation of a graph
        
    Returns:
        A networkx Graph
    """
    G = nx.Graph()
    G.add_nodes_from(graph_dict["nodes"])
    
    for edge in graph_dict["edges"]:
        G.add_edge(edge["source"], edge["target"])
    
    return G 