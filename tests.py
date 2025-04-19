import unittest
import networkx as nx
import os
from unittest.mock import patch, MagicMock

from graph_problem_generator.graph_generator import (
    generate_random_graph,
    get_shortest_path,
    check_path_exists,
    graph_to_dict,
    dict_to_graph
)
from graph_problem_generator.problem_generator import ProblemGenerator


class TestGraphGenerator(unittest.TestCase):
    """Test cases for the graph generator module."""
    
    def test_generate_random_graph(self):
        """Test generating a random graph."""
        min_nodes = 5
        max_nodes = 10
        G = generate_random_graph(min_nodes=min_nodes, max_nodes=max_nodes)
        
        # Check if the number of nodes is within the specified range
        self.assertTrue(min_nodes <= len(G.nodes) <= max_nodes)
        
        # Check if the graph is connected
        self.assertTrue(nx.is_connected(G))
    
    def test_get_shortest_path(self):
        """Test finding the shortest path."""
        # Create a simple graph
        G = nx.Graph()
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('A', 'C')
        
        # Test shortest path
        path, distance = get_shortest_path(G, 'A', 'C')
        self.assertEqual(path, ['A', 'C'])
        self.assertEqual(distance, 1)  # Direct connection, so distance is 1
        
        # Test no path
        G.remove_edge('B', 'C')
        G.remove_edge('A', 'C')
        path, distance = get_shortest_path(G, 'A', 'C')
        self.assertEqual(path, ['A', 'B', 'C'])
        self.assertEqual(distance, 2)  # Now it's A->B->C, so distance is 2
        
        # Test completely disconnected
        G.remove_edge('A', 'B')
        path, distance = get_shortest_path(G, 'A', 'C')
        self.assertEqual(path, [])
        self.assertEqual(distance, -1)
    
    def test_check_path_exists(self):
        """Test checking if a path exists."""
        # Create a simple graph
        G = nx.Graph()
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        
        # Test path exists
        self.assertTrue(check_path_exists(G, 'A', 'C'))
        
        # Test no path
        G.remove_edge('B', 'C')
        self.assertFalse(check_path_exists(G, 'A', 'C'))
    
    def test_graph_conversion(self):
        """Test converting between graph and dictionary."""
        # Create a simple graph
        G = nx.Graph()
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        
        # Convert to dictionary
        graph_dict = graph_to_dict(G)
        
        # Check dictionary structure
        self.assertIn('nodes', graph_dict)
        self.assertIn('edges', graph_dict)
        self.assertEqual(set(graph_dict['nodes']), {'A', 'B', 'C'})
        
        # Check edges format
        for edge in graph_dict['edges']:
            self.assertIn('source', edge)
            self.assertIn('target', edge)
            self.assertNotIn('weight', edge)
        
        # Convert back to graph
        G2 = dict_to_graph(graph_dict)
        
        # Check if the graphs are isomorphic
        self.assertTrue(nx.is_isomorphic(G, G2))


class TestProblemGenerator(unittest.TestCase):
    """Test cases for the problem generator module."""
    
    @patch('openai.OpenAI')
    def test_problem_generator_initialization(self, mock_openai):
        """Test initializing the problem generator."""
        # Setup the mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Test with API key as argument
        with patch.dict(os.environ, {}, clear=True):
            generator = ProblemGenerator(api_key="test_key")
            self.assertEqual(generator.api_key, "test_key")
            mock_openai.assert_called_with(api_key="test_key")
        
        # Test with environment variable
        mock_openai.reset_mock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
            generator = ProblemGenerator()
            self.assertEqual(generator.api_key, "env_key")
            mock_openai.assert_called_with(api_key="env_key")
        
        # Test without API key
        mock_openai.reset_mock()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                generator = ProblemGenerator()
            mock_openai.assert_not_called()
        
        # Test with client injection
        mock_openai.reset_mock()
        custom_client = MagicMock()
        generator = ProblemGenerator(client=custom_client)
        self.assertEqual(generator.client, custom_client)
        mock_openai.assert_not_called()
    
    def test_generate_problem_text(self):
        """Test generating problem text."""
        # Setup the mock client
        mock_message = MagicMock()
        mock_message.content = "Test problem text"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        mock_chat_completions = MagicMock()
        mock_chat_completions.create.return_value = mock_response
        
        mock_client = MagicMock()
        mock_client.chat.completions = mock_chat_completions
        
        # Create a simple graph
        G = nx.Graph()
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        
        # Test generating problem text
        generator = ProblemGenerator(client=mock_client)
        problem_text = generator.generate_problem_text(G, 'A', 'C', "shortest_path")
        
        self.assertEqual(problem_text, "Test problem text")
        mock_chat_completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main() 