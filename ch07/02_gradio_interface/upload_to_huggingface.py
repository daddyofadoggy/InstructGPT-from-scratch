# Script to upload fine-tuned GPT model to Hugging Face Hub
import os
from pathlib import Path
import shutil
import torch
from huggingface_hub import HfApi, upload_folder
import tiktoken
import argparse

# Import your model class
from llms_from_scratch.ch04 import GPTModel

# Parse command line arguments
parser = argparse.ArgumentParser(description='Upload fine-tuned GPT model to Hugging Face Hub')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file')
parser.add_argument('--repo_name', type=str, required=True, help='Name for the Hugging Face repository (e.g., username/model-name)')
parser.add_argument('--token', type=str, help='Hugging Face API token. If not provided, will use the HUGGING_FACE_HUB_TOKEN environment variable')
args = parser.parse_args()

# Set up the model configuration
GPT_CONFIG_355M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,         # Embedding dimension
    "n_heads": 16,           # Number of attention heads
    "n_layers": 24,          # Number of layers
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

# Create a temporary directory to store the model files
temp_dir = Path("./hf_model_upload")
temp_dir.mkdir(exist_ok=True)

# Load the model checkpoint
print(f"Loading model from {args.model_path}")
checkpoint = torch.load(args.model_path, weights_only=True)
model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(checkpoint)

# Save the model in the temporary directory
model_path = temp_dir / "gpt2-medium355M-sft.pth"
torch.save(model.state_dict(), model_path)

# Create a README.md file with information about your model
with open(temp_dir / "README.md", "w") as f:
    f.write(f"""# Fine-tuned GPT Model

This is a fine-tuned GPT model based on the architecture described in the book "Build a Large Language Model From Scratch" by Sebastian Raschka.

## Model Details

- **Architecture**: GPT-2 Medium (355M parameters)
- **Training**: Instruction fine-tuned
- **Context Length**: 1024 tokens
- **Use Case**: Conversational AI

## Usage

```python
from pathlib import Path
import torch
import tiktoken
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text

# Set up model configuration
GPT_CONFIG_355M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,         # Embedding dimension
    "n_heads": 16,           # Number of attention heads
    "n_layers": 24,          # Number of layers
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("gpt2-medium355M-sft.pth", weights_only=True)
model = GPTModel(GPT_CONFIG_355M)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Generate text
prompt = "Tell me about deep learning"
token_ids = generate(
    model=model,
    idx=text_to_token_ids(prompt, tokenizer).to(device),
    max_new_tokens=100,
    context_size=GPT_CONFIG_355M["context_length"],
    eos_id=50256
)
text = token_ids_to_text(token_ids, tokenizer)
print(text)
```

## Requirements

- llms_from_scratch package (https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg)
- tiktoken
- torch

## License

This model is released under the Apache License 2.0.
""")

# Create a requirements.txt file
with open(temp_dir / "requirements.txt", "w") as f:
    f.write("""torch>=2.0.0
tiktoken
gradio>=4.0.0
git+https://github.com/rasbt/LLMs-from-scratch.git#subdirectory=pkg
""")

# Copy the Gradio app
shutil.copy("app_gradio.py", temp_dir / "app.py")

# Create a Hugging Face Space configuration file
os.makedirs(temp_dir / ".github" / "workflows", exist_ok=True)
with open(temp_dir / ".github" / "workflows" / "sync-to-space.yml", "w") as f:
    f.write("""name: Sync to Hugging Face Space

on:
  push:
    branches: [main]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          SPACE_ID: ${{ secrets.SPACE_ID }}
        run: git push https://HF_TOKEN:$HF_TOKEN@huggingface.co/spaces/$SPACE_ID main
""")

# Upload the model to Hugging Face Hub
print(f"Uploading model to Hugging Face Hub repository: {args.repo_name}")
api = HfApi()

# Use the provided token or get from environment variable
token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise ValueError("No Hugging Face token provided. Please provide it via --token or set the HUGGING_FACE_HUB_TOKEN environment variable.")

# Create the repository if it doesn't exist
try:
    api.create_repo(
        repo_id=args.repo_name,
        token=token,
        private=False,
        repo_type="model",
        exist_ok=True
    )
except Exception as e:
    print(f"Warning: {e}")
    print("Continuing with upload...")

# Upload the model files
upload_folder(
    folder_path=str(temp_dir),
    repo_id=args.repo_name,
    token=token
)

print(f"Successfully uploaded model to https://huggingface.co/{args.repo_name}")

# Create a Hugging Face Space for the demo
space_name = f"{args.repo_name.split('/')[0]}/{args.repo_name.split('/')[-1]}-space"
print(f"Creating Hugging Face Space: {space_name}")

try:
    api.create_repo(
        repo_id=space_name,
        token=token,
        private=False,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True
    )
except Exception as e:
    print(f"Warning: {e}")
    print("Continuing with upload...")

# Upload the app to the Space
upload_folder(
    folder_path=str(temp_dir),
    repo_id=space_name,
    token=token
)

print(f"Successfully created Gradio Space at https://huggingface.co/spaces/{space_name}")
print("\nSetup complete!")
print(f"Model repository: https://huggingface.co/{args.repo_name}")
print(f"Gradio interface: https://huggingface.co/spaces/{space_name}")
print("\nTo run the Gradio interface locally:")
print("python app_gradio.py")
