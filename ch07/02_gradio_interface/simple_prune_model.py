# Simple model pruning script that doesn't use torch.quantile
import torch
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Prune a PyTorch model to reduce its size')
parser.add_argument('--model_path', type=str, required=True, help='Path to the original model checkpoint file')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the pruned model')
parser.add_argument('--sparsity', type=float, default=0.3, help='Target sparsity level (0.0 to 1.0)')
args = parser.parse_args()

print(f"Loading model from {args.model_path}")
checkpoint = torch.load(args.model_path, weights_only=True)

# Function to prune weights using a simpler approach
def prune_weights_simple(state_dict, sparsity):
    new_state_dict = {}
    total_params = 0
    pruned_params = 0
    
    for key, weights in state_dict.items():
        if isinstance(weights, torch.Tensor) and weights.dim() > 1:  # Only prune matrices, not vectors or scalars
            # Handle each parameter group separately to save memory
            abs_weights = weights.abs().detach().cpu()
            
            # Process in batches if tensor is large
            if abs_weights.numel() > 10000000:  # 10M elements threshold
                print(f"Processing large tensor {key} in batches")
                # Create a new tensor of the same shape filled with zeros
                pruned_weights = torch.zeros_like(weights)
                
                # Process the tensor in slices along the first dimension
                batch_size = max(1, weights.shape[0] // 10)  # Process ~10 batches
                
                all_thresholds = []
                # First pass: calculate thresholds for each batch
                for i in range(0, weights.shape[0], batch_size):
                    end_idx = min(i + batch_size, weights.shape[0])
                    batch = abs_weights[i:end_idx].flatten()
                    sorted_batch, _ = torch.sort(batch)
                    threshold_idx = int(len(batch) * sparsity)
                    if threshold_idx < len(sorted_batch):
                        all_thresholds.append(sorted_batch[threshold_idx].item())
                
                # Use the median threshold across all batches
                threshold = np.median(all_thresholds)
                
                # Second pass: apply the threshold to each batch
                for i in range(0, weights.shape[0], batch_size):
                    end_idx = min(i + batch_size, weights.shape[0])
                    batch = weights[i:end_idx]
                    mask = (abs_weights[i:end_idx] > threshold)
                    pruned_weights[i:end_idx] = batch * mask
                
                # Count parameters
                mask = (abs_weights > threshold)
                kept_params = mask.sum().item()
            else:
                # For smaller tensors, we can process them directly
                sorted_abs_weights, _ = torch.sort(abs_weights.flatten())
                threshold_idx = int(sorted_abs_weights.numel() * sparsity)
                threshold = sorted_abs_weights[threshold_idx].item() if threshold_idx < sorted_abs_weights.numel() else 0
                
                # Create mask and apply
                mask = (abs_weights > threshold)
                pruned_weights = weights * mask
                kept_params = mask.sum().item()
            
            new_state_dict[key] = pruned_weights
            
            # Count parameters
            total_params += weights.numel()
            pruned_params += weights.numel() - kept_params
        else:
            new_state_dict[key] = weights
    
    print(f"Pruned {pruned_params} out of {total_params} parameters ({(pruned_params/total_params)*100:.2f}%)")
    return new_state_dict

# Prune the model weights
pruned_checkpoint = prune_weights_simple(checkpoint, args.sparsity)

# Save the pruned model
torch.save(pruned_checkpoint, args.output_path)

# Print file sizes for comparison
original_size = Path(args.model_path).stat().st_size / (1024 * 1024)  # Size in MB
pruned_size = Path(args.output_path).stat().st_size / (1024 * 1024)  # Size in MB
print(f"Original model size: {original_size:.2f} MB")
print(f"Pruned model size: {pruned_size:.2f} MB")
print(f"Size reduction: {(1 - pruned_size/original_size) * 100:.2f}%")
