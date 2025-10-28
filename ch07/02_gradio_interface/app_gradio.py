# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import gradio as gr

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
#from llms_from_scratch.ch04 import GPTModel
# from llms_from_scratch.ch05 import (
#     generate,
#     text_to_token_ids,
#     token_ids_to_text,
# )

from previous_chapters import GPTModel

from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.
    """

    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    # For local development
    model_path = Path("gpt2-small124M-sft.pth")
    
    # For Hugging Face deployment
    hf_model_path = Path("gpt2-small124M-sft.pth")
    
    # Try loading from the Hugging Face model path first, then fall back to local
    if hf_model_path.exists():
        model_path = hf_model_path
    elif not model_path.exists():
        print(
            f"Could not find the model file. Please run the chapter 7 code "
            "to generate the gpt2-medium355M-sft.pth file or upload it to this directory."
        )
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # Set to evaluation mode

    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Load model and tokenizer
tokenizer, model, model_config = get_model_and_tokenizer()


def generate_response(message, max_new_tokens=100):
    """Generate a response using the fine-tuned GPT model"""
    torch.manual_seed(123)
    
    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message}
    """
    
    with torch.no_grad():  # Ensure no gradients are computed during inference
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=max_new_tokens,
            context_size=model_config["context_length"],
            eos_id=50256
        )

    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)
    
    return response


# Create a Gradio chat interface
def respond(message, history):
    bot_message = generate_response(message)
    return bot_message


# Define the chat interface
chat_interface = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Ask me something...", container=False, scale=7),
    title="Fine-tuned GPT Model Chat",
    description="Chat with a fine-tuned GPT model from 'Build a Large Language Model From Scratch' by Sebastian Raschka",
    theme="soft",
    examples=[
        "What is the capital of France?",
        "What is the opposite of 'wet'?",
        "Write a short poem about AI",
        "Explain the concept of attention in neural networks"
    ],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

# Launch the interface
if __name__ == "__main__":
    chat_interface.launch(share=True)