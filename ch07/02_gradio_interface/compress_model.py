# Script to compress a PyTorch model before uploading to Hugging Face
import torch
from pathlib import Path
import argparse
import tiktoken
from previous_chapters import GPTModel

parser = argparse.ArgumentParser(description='Compress a PyTorch model before uploading to Hugging Face')
parser.add_argument('--model_path', type=str, required=True, help='Path to the original model checkpoint file')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the compressed model')
parser.add_argument('--quantize', action='store_true', help='Apply quantization to reduce model size')
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

# Create the model
model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(checkpoint)

if args.quantize:
    print("Applying quantization to reduce model size...")
    # Quantize the model to int8
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    # Save the quantized model
    torch.save(model_quantized.state_dict(), args.output_path)
    print(f"Quantized model saved to {args.output_path}")
else:
    # Apply simple compression without quantization
    print("Compressing model...")
    torch.save(model.state_dict(), args.output_path, _use_new_zipfile_serialization=True)
    print(f"Compressed model saved to {args.output_path}")

# Print file sizes for comparison
original_size = Path(args.model_path).stat().st_size / (1024 * 1024)  # Size in MB
compressed_size = Path(args.output_path).stat().st_size / (1024 * 1024)  # Size in MB
print(f"Original model size: {original_size:.2f} MB")
print(f"Compressed model size: {compressed_size:.2f} MB")
print(f"Size reduction: {(1 - compressed_size/original_size) * 100:.2f}%")
