# Improved script to prune a PyTorch model to reduce its size
# Handles large tensors by processing them in chunks
import torch
from pathlib import Path
import argparse
import tiktoken
from previous_chapters import GPTModel
import numpy as np

parser = argparse.ArgumentParser(description='Prune a PyTorch model to reduce its size')
parser.add_argument('--model_path', type=str, required=True, help='Path to the original model checkpoint file')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the pruned model')
parser.add_argument('--sparsity', type=float, default=0.3, help='Target sparsity level (0.0 to 1.0)')
args = parser.parse_args()

# Model configuration
GPT_CONFIG_355M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,         # Embedding dimension
    "n_heads": 16,           # Number of attention heads
    "n_layers": 24,          # Number of layers
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

print(f"Loading model from {args.model_path}")
checkpoint = torch.load(args.model_path, weights_only=True)

# Function to find threshold for large tensors by sampling
def find_threshold_by_sampling(tensor, sparsity, sample_size=1000000):
    # Flatten the tensor
    flat_tensor = tensor.abs().flatten()
    tensor_size = flat_tensor.numel()
    
    # If tensor is smaller than sample size, use the whole tensor
    if tensor_size <= sample_size:
        sorted_values, _ = torch.sort(flat_tensor)
        threshold_index = int(sparsity * tensor_size)
        return sorted_values[threshold_index]
    
    # Otherwise, sample from the tensor
    indices = torch.randperm(tensor_size)[:sample_size]
    sampled_values = flat_tensor[indices]
    sorted_values, _ = torch.sort(sampled_values)
    threshold_index = int(sparsity * sample_size)
    return sorted_values[threshold_index]

# Function to prune weights by magnitude
def prune_weights(state_dict, sparsity):
    new_state_dict = {}
    total_params = 0
    pruned_params = 0
    
    for key, weights in state_dict.items():
        if isinstance(weights, torch.Tensor) and weights.dim() > 1:  # Only prune matrices, not vectors or scalars
            try:
                # First try with torch.quantile
                threshold = torch.quantile(weights.abs().flatten(), sparsity)
            except RuntimeError:
                # If that fails, use our sampling method
                print(f"Using sampling method for large tensor: {key} (shape: {weights.shape})")
                threshold = find_threshold_by_sampling(weights, sparsity)
            
            # Create mask of weights to keep
            mask = weights.abs() > threshold
            
            # Apply the mask
            pruned_weights = weights * mask
            
            new_state_dict[key] = pruned_weights
            
            # Count parameters
            total_params += weights.numel()
            pruned_params += weights.numel() - mask.sum().item()
        else:
            new_state_dict[key] = weights
    
    print(f"Pruned {pruned_params} out of {total_params} parameters ({(pruned_params/total_params)*100:.2f}%)")
    return new_state_dict

# Prune the model weights
pruned_checkpoint = prune_weights(checkpoint, args.sparsity)

# Save the pruned model
torch.save(pruned_checkpoint, args.output_path)

# Print file sizes for comparison
original_size = Path(args.model_path).stat().st_size / (1024 * 1024)  # Size in MB
pruned_size = Path(args.output_path).stat().st_size / (1024 * 1024)  # Size in MB
print(f"Original model size: {original_size:.2f} MB")
print(f"Pruned model size: {pruned_size:.2f} MB")
print(f"Size reduction: {(1 - pruned_size/original_size) * 100:.2f}%")
