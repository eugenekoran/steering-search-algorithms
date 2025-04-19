import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

def load_data(file_path="activations.pt"):
    """
    Load the activations data and prepare it for probe training.
    
    Args:
        file_path: Path to the PyTorch file containing the activations data
        
    Returns:
        all_activations: Tensor of shape (16, n_samples, model_dim) containing all bucketed activations
        train_mask: Boolean mask indicating which samples are for training
        y: Problem types (labels) for all samples
        problem_type_to_idx: Mapping of problem types to numeric indices
        idx_to_problem_type: Mapping of numeric indices to problem types
    """
    print(f"Loading data from {file_path}...")
    data = torch.load(file_path)
    print(f"Loaded {len(data)} examples")
    
    # Extract train/test status and problem types
    train_mask = torch.tensor([item["is_train"] for item in data])
    problem_types = sorted(set(item["problem_type"] for item in data))
    problem_type_to_idx = {problem_type: idx for idx, problem_type in enumerate(problem_types)}
    idx_to_problem_type = {idx: problem_type for problem_type, idx in problem_type_to_idx.items()}
    
    # Convert problem types to numeric indices
    y = torch.tensor([problem_type_to_idx[item["problem_type"]] for item in data])
    
    print(f"Train examples: {train_mask.sum().item()}")
    print(f"Test examples: {(~train_mask).sum().item()}")
    print(f"Problem types: {problem_types}")
    
    # Extract bucketed activations
    bucketed_activations = [item["bucketed_activations"] for item in data]
    
    # Convert to a single tensor
    # Shape of each item: (4 layers, 4 buckets, model_dim)
    # We'll reshape to (16 probes, n_samples, model_dim)
    n_samples = len(data)
    n_layers = 4
    n_buckets = 4
    model_dim = bucketed_activations[0].shape[2]
    
    # Initialize the tensor for all activations
    all_activations = torch.zeros(n_layers * n_buckets, n_samples, model_dim)
    
    # Flatten the layer and bucket dimensions for each sample
    for i, activations in enumerate(bucketed_activations):
        # Reshape from (4, 4, model_dim) to (16, model_dim)
        flattened = activations.reshape(-1, model_dim)
        # Store in the all_activations tensor
        all_activations[:, i, :] = flattened
    
    print(f"Prepared activations tensor with shape: {all_activations.shape}")
    
    return all_activations, train_mask, y, problem_type_to_idx, idx_to_problem_type

def train_probes(all_activations, train_mask, y, idx_to_problem_type):
    """
    Train 16 linear probes (one for each layer-bucket combination) and evaluate them.
    
    Args:
        all_activations: Tensor of shape (16, n_samples, model_dim) containing all bucketed activations
        train_mask: Boolean mask indicating which samples are for training
        y: Problem types (labels) for all samples
        idx_to_problem_type: Mapping of numeric indices to problem types
        
    Returns:
        results: Dictionary mapping probe_idx to accuracy
        best_probe_idx: Index of the best performing probe
        trained_models: Dictionary mapping probe_idx to trained model
    """
    results = {}
    trained_models = {}
    
    n_probes, n_samples, model_dim = all_activations.shape
    
    # Convert to numpy for sklearn compatibility
    all_activations_np = all_activations.numpy()
    y_np = y.numpy()
    
    # Create train and test masks
    train_indices = torch.where(train_mask)[0].numpy()
    test_indices = torch.where(~train_mask)[0].numpy()
    
    # Map from probe_idx to (layer_idx, bucket_idx)
    def probe_to_layer_bucket(probe_idx):
        layer_idx = probe_idx // 4
        bucket_idx = probe_idx % 4
        return layer_idx, bucket_idx
    
    # Map from (layer_idx, bucket_idx) to actual layer number
    layer_mapping = {0: 7, 1: 14, 2: 21, 3: 28}
    
    # Train a probe for each of the 16 combinations
    for probe_idx in range(n_probes):
        layer_idx, bucket_idx = probe_to_layer_bucket(probe_idx)
        real_layer = layer_mapping[layer_idx]
        
        print(f"Training probe {probe_idx}: Layer {real_layer} (index {layer_idx}), Bucket {bucket_idx}...")
        
        # Extract the features for this probe across all samples
        X = all_activations_np[probe_idx]
        
        # Split into train and test
        X_train = X[train_indices]
        y_train = y_np[train_indices]
        X_test = X[test_indices]
        y_test = y_np[test_indices]
        
        # Train the probe
        probe = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
        probe.fit(X_train, y_train)
        
        # Evaluate the probe
        y_pred = probe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store the results
        results[probe_idx] = accuracy
        trained_models[probe_idx] = probe
        
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Print detailed metrics for this probe
        print("  Classification report:")
        print(classification_report(y_test, y_pred, target_names=[idx_to_problem_type[i] for i in range(len(idx_to_problem_type))]))
    
    # Find the best performing probe
    best_probe_idx = max(results, key=results.get)
    best_accuracy = results[best_probe_idx]
    best_layer_idx, best_bucket_idx = probe_to_layer_bucket(best_probe_idx)
    best_layer = layer_mapping[best_layer_idx]
    
    print(f"\nBest probe: {best_probe_idx} - Layer {best_layer} (index {best_layer_idx}), Bucket {best_bucket_idx}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    return results, best_probe_idx, trained_models

def visualize_results(results, idx_to_problem_type, trained_models, all_activations, train_mask, y):
    """
    Visualize the results of the probe training.
    
    Args:
        results: Dictionary mapping probe_idx to accuracy
        idx_to_problem_type: Mapping of numeric indices to problem types
        trained_models: Dictionary mapping probe_idx to trained model
        all_activations: Tensor of shape (16, n_samples, model_dim) containing all bucketed activations
        train_mask: Boolean mask indicating which samples are for training
        y: Problem types (labels) for all samples
    """
    # Create a directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Reshape the results into a 4x4 grid for the heatmap
    accuracies = np.zeros((4, 4))
    for probe_idx, accuracy in results.items():
        layer_idx = probe_idx // 4
        bucket_idx = probe_idx % 4
        accuracies[layer_idx, bucket_idx] = accuracy
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracies, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=["0-25%", "25-50%", "50-75%", "75-100%"],
                yticklabels=["Layer 7", "Layer 14", "Layer 21", "Layer 28"])
    plt.title("Probe Accuracy by Layer and Bucket")
    plt.xlabel("Bucket (Percentage of Sequence)")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig("visualizations/probe_accuracy_heatmap.png")
    plt.close()
    
    # Find the best performing probe
    best_probe_idx = max(results, key=results.get)
    best_model = trained_models[best_probe_idx]
    
    # Get test indices
    test_indices = torch.where(~train_mask)[0].numpy()
    
    # Get the predictions for the best probe
    X_test = all_activations[best_probe_idx, test_indices].numpy()
    y_test = y[test_indices].numpy()
    y_pred = best_model.predict(X_test)
    
    # Plot the confusion matrix for the best probe
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[idx_to_problem_type[i] for i in range(len(idx_to_problem_type))],
                yticklabels=[idx_to_problem_type[i] for i in range(len(idx_to_problem_type))])
    
    # Determine the layer and bucket of the best probe
    best_layer_idx = best_probe_idx // 4
    best_bucket_idx = best_probe_idx % 4
    layer_mapping = {0: 7, 1: 14, 2: 21, 3: 28}
    best_layer = layer_mapping[best_layer_idx]
    
    plt.title(f"Confusion Matrix for Best Probe (Layer {best_layer}, Bucket {best_bucket_idx})")
    plt.xlabel("Predicted Problem Type")
    plt.ylabel("True Problem Type")
    plt.tight_layout()
    plt.savefig("visualizations/best_probe_confusion_matrix.png")
    plt.close()
    
    # Analyze feature importance for the best probe
    if hasattr(best_model, 'coef_'):
        coef = best_model.coef_
        
        # For multiclass, we'll look at the average absolute coefficient across all classes
        if len(coef.shape) > 1 and coef.shape[0] > 1:
            feature_importance = np.mean(np.abs(coef), axis=0)
        else:
            # For binary classification
            feature_importance = np.abs(coef[0])
            
        top_k = 20  # Show top 20 features
        top_indices = np.argsort(feature_importance)[-top_k:]
        top_importances = feature_importance[top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_k), top_importances, align='center')
        plt.yticks(range(top_k), [f"Feature {i}" for i in top_indices])
        plt.title(f"Top {top_k} Feature Importances for Best Probe")
        plt.xlabel("Absolute Coefficient Value")
        plt.gca().invert_yaxis()  # Highest values at the top
        plt.tight_layout()
        plt.savefig("visualizations/best_probe_feature_importance.png")
        plt.close()

def main():
    """Main function to run the probe training and evaluation."""
    # Load the data in vectorized form
    all_activations, train_mask, y, problem_type_to_idx, idx_to_problem_type = load_data()
    
    # Train the probes
    results, best_probe_idx, trained_models = train_probes(
        all_activations, train_mask, y, idx_to_problem_type
    )
    
    # Visualize the results
    visualize_results(results, idx_to_problem_type, trained_models, all_activations, train_mask, y)
    
    # Save the best model
    best_layer_idx = best_probe_idx // 4
    best_bucket_idx = best_probe_idx % 4
    layer_mapping = {0: 7, 1: 14, 2: 21, 3: 28}
    best_layer = layer_mapping[best_layer_idx]
    
    print(f"Saving best model (Probe {best_probe_idx}: Layer {best_layer}, Bucket {best_bucket_idx})...")
    torch.save({
        'model': trained_models[best_probe_idx],
        'probe_idx': best_probe_idx,
        'layer_idx': best_layer_idx,
        'bucket_idx': best_bucket_idx,
        'layer': best_layer,
        'accuracy': results[best_probe_idx],
        'problem_type_to_idx': problem_type_to_idx,
        'idx_to_problem_type': idx_to_problem_type
    }, "best_probe.pt")
    
    print("Done!")

if __name__ == "__main__":
    main()
