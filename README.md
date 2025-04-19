# Steering Search Algorithms in Distill-Qwen-1.5B

Recent reasoning models trained with reinforcement learning demonstrate remarkable problem-solving capabilities, yet their internal processes remain poorly understood. This project investigates whether these models encode algorithmic strategies (specifically Depth-First Search vs. Breadth-First Search) in their activations and whether these representations can be manipulated through activation steering.

DFS and BFS provide an ideal comparison because they operate on identical data structures but employ fundamentally different exploration strategies—DFS prioritizes depth while BFS explores neighboring nodes first—representing distinct reasoning patterns.

## Key Findings

Our experiments reveal that reasoning models' activations contain linearly separable dimensions corresponding to traversal strategies:

1. A linear probe trained on the residual stream predicted the traversal algorithm with 97.5% accuracy (Layer 14), confirming the presence of algorithm-related information.

2. Activation steering interventions demonstrated these representations are causally influential: by manipulating internal activations, we successfully altered the model's traversal algorithm, though this required significant scaling (50x) of the algorithm direction.

These results don't necessarily indicate that models encode algorithms explicitly like humans. The model may instead employ a general queuing circuit configurable for different traversal patterns, with our probes detecting these configuration differences.

## Methodology

### Experiment 1: Linear Probing 

We generated 100 pairs of graph-based prompts designed to elicit either DFS or BFS approaches without explicit algorithmic hints. We then trained linear probes across multiple layers in the model's residual streams and various reasoning stages, with the most successful probe achieving 97.5% accuracy at Layer 14.

### Experiment 2: Causal Intervention 

Using PyTorch hooks, we manipulated the 'algorithm direction' at Layer 14. ChatGPT served as an independent evaluator to verify whether outputs shifted between algorithms as intended, confirming our interventions consistently altered reasoning strategies.

## Implications & Limitations

This research demonstrates we can identify and manipulate algorithmic reasoning representations within RL-trained models, opening pathways to controlled model steering, enhanced interpretability, and safer deployment.
While successful in distinguishing between DFS and BFS, scaling this approach to multiple or more subtle reasoning strategies may prove challenging due to increased activation overlap or ambiguity. Additionally, our findings don't conclusively prove the model "encodes" algorithms analogously to human understanding—we may be observing emergent patterns in how the model configures its general reasoning circuits.
