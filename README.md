# Build a Large Language Model (From Scratch) - Chapter Summary Report

## Overview

This repository implements a complete GPT-like Large Language Model from scratch using PyTorch, covering the entire development lifecycle from basic text processing to instruction-following capabilities. The implementation mirrors the approach used in creating foundational models like ChatGPT, but in an educational scale that can run on a laptop.

---

## Chapter-by-Chapter Summary

### Chapter 1: Understanding Large Language Models

**Content**: Introduction and overview chapter with no code implementation. This chapter provides conceptual foundation for understanding how Large Language Models work and sets the stage for the hands-on implementation in subsequent chapters.

**Key Topics**:
- Overview of Large Language Models
- High-level architecture concepts
- LLM development lifecycle

---

### Chapter 2: Working with Text Data

![Data Preparation Overview](ch02/01_main-chapter-code/ch02_compressed/01.webp)

**Summary**: Chapter 2 establishes the fundamental data preparation pipeline needed to get text data ready for LLM training. It walks through tokenizing text (breaking it into smaller units like words and punctuation), converting tokens into numerical token IDs, and creating data loaders that generate training batches. The chapter implements a simple tokenizer from scratch before introducing OpenAI's BytePair Encoding (BPE) tokenizer via tiktoken. Finally, it demonstrates creating embeddings (token embeddings and positional embeddings) that convert token IDs into continuous vector representations suitable for neural network training.

**Key Concepts**:
- **Tokenization**: Breaking text into smaller units (words, punctuation, subwords)
- **Vocabulary building**: Creating mappings between tokens and integer IDs
- **BytePair Encoding (BPE)**: Subword tokenization algorithm handling out-of-vocabulary words
- **Token embeddings**: Converting token IDs into dense vector representations
- **Positional embeddings**: Encoding positional information for each token
- **Special tokens**: `<|endoftext|>` for boundaries, `<|unk|>` for unknown words
- **Data loading**: PyTorch DataLoader with sliding window approach

**Notable Implementations**:
- `SimpleTokenizerV1` and `SimpleTokenizerV2`: Custom tokenizer classes
- `GPTDatasetV1`: Custom PyTorch Dataset with sliding window batching
- `create_dataloader_v1`: Factory function for data loaders
- Embedding layer demonstration with `nn.Embedding`

![Tokenization Process](ch02/01_main-chapter-code/ch02_compressed/04.webp)

![Input Embeddings Workflow](ch02/01_main-chapter-code/ch02_compressed/19.webp)

**Code Location**: `ch02/01_main-chapter-code/ch02.ipynb`

---

### Chapter 3: Coding Attention Mechanisms

![Attention Mechanisms Overview](ch03/01_main-chapter-code/ch03_compressed/01.webp)

**Summary**: Chapter 3 implements the attention mechanism, the core component enabling LLMs to process and understand contextual relationships in text. It starts with a simplified self-attention mechanism without trainable weights, then builds up to full scaled dot-product attention with query, key, and value projections. The chapter introduces causal attention masking (preventing tokens from attending to future positions) and dropout for regularization. Finally, it extends single-head attention to multi-head attention, allowing the model to learn different representation subspaces simultaneously.

**Key Concepts**:
- **Self-attention mechanism**: Computing context-aware representations by attending to all positions
- **Attention scores (ω)**: Unnormalized compatibility scores between query and key vectors
- **Attention weights**: Normalized attention scores (via softmax) summing to 1
- **Query, Key, Value (QKV) matrices**: Trainable weight matrices for projecting inputs
- **Scaled dot-product attention**: Dividing scores by √d_k for training stability
- **Causal attention mask**: Preventing future token access during autoregressive generation
- **Dropout in attention**: Randomly masking attention weights during training
- **Multi-head attention**: Running multiple attention mechanisms in parallel

**Notable Implementations**:
- Simple self-attention (no trainable weights)
- `SelfAttention_v1` and `SelfAttention_v2`: Classes with trainable W_query, W_key, W_value
- `CausalAttention`: Complete implementation with dropout and causal masking
- `MultiHeadAttention`: Efficient implementation using weight splitting

![Self-Attention Process](ch03/01_main-chapter-code/ch03_compressed/12.webp)

![Causal Attention Mask](ch03/01_main-chapter-code/ch03_compressed/19.webp)

![Multi-Head Attention](ch03/01_main-chapter-code/ch03_compressed/25.webp)

**Code Location**: `ch03/01_main-chapter-code/ch03.ipynb`

---

### Chapter 4: Implementing a GPT Model from Scratch

![GPT Architecture Overview](ch04/01_main-chapter-code/ch04_compressed/02.webp)

**Summary**: Chapter 4 assembles all previous components into a complete GPT architecture capable of generating text. It implements layer normalization for training stability, the GELU activation function, feed-forward networks, and shortcut/residual connections to prevent vanishing gradients. The chapter constructs transformer blocks by combining multi-head attention with feed-forward layers, then stacks 12 of these blocks to create the full 124M parameter GPT-2 model. Finally, it implements a greedy decoding text generation function that produces one token at a time.

**Key Concepts**:
- **Layer Normalization (LayerNorm)**: Centering activations around mean 0 with variance 1
- **GELU activation function**: Gaussian Error Linear Unit, smooth alternative to ReLU
- **Feed-forward networks**: Two-layer MLPs with expansion factor of 4
- **Shortcut/residual connections**: Adding layer input to output to prevent vanishing gradients
- **Transformer blocks**: Combining attention, feed-forward, normalization, and residual connections
- **GPT architecture**: Token embeddings → positional embeddings → 12 transformer blocks → output head
- **Model configuration**: vocab_size=50257, context_length=1024, emb_dim=768, n_heads=12, n_layers=12
- **Greedy decoding**: Selecting highest probability token at each generation step

**Notable Implementations**:
- `LayerNorm`: Custom implementation with trainable scale and shift parameters
- `GELU`: Implementing the approximation formula for GELU activation
- `FeedForward`: 768 → 3072 → 768 dimension transformation with GELU
- `TransformerBlock`: Combining MultiHeadAttention, FeedForward, LayerNorm, and residual connections
- `GPTModel`: Complete architecture with 12 transformer blocks
- `generate_text_simple`: Iterative greedy decoding function

![Complete GPT Architecture](ch04/01_main-chapter-code/ch04_compressed/11.webp)

![Transformer Block Structure](ch04/01_main-chapter-code/ch04_compressed/13.webp)

![Text Generation Process](ch04/01_main-chapter-code/ch04_compressed/16.webp)

**Code Location**: `ch04/01_main-chapter-code/ch04.ipynb`

---

### Chapter 5: Pretraining on Unlabeled Data

![Chapter Overview](ch05/01_main-chapter-code/ch05_compressed/chapter-overview.webp)

**Summary**: Chapter 5 implements the complete training loop for pretraining an LLM on unlabeled text data using next-token prediction. It introduces cross-entropy loss and perplexity as evaluation metrics, then trains a GPT model on a short story dataset. The chapter covers temperature scaling and top-k sampling as decoding strategies to control randomness in text generation. Finally, it demonstrates how to load pretrained GPT-2 weights from OpenAI, converting them from TensorFlow format into the custom PyTorch implementation, enabling use of models from 124M to 1558M parameters.

**Key Concepts**:
- **Cross-entropy loss**: Negative log-likelihood of correct predictions, used as training objective
- **Perplexity**: Exponential of cross-entropy loss, interpretable as effective vocabulary uncertainty
- **Training/validation/test split**: 70/10/20 ratio for model evaluation
- **Training loop**: Forward pass → loss calculation → backpropagation → parameter update
- **Temperature scaling**: Dividing logits by temperature to control output distribution sharpness
- **Top-k sampling**: Restricting sampling to k most likely tokens
- **Model checkpointing**: Saving model state_dict and optimizer state
- **Weight loading**: Converting and loading OpenAI's pretrained GPT-2 weights
- **AdamW optimizer**: Adaptive learning rate with weight decay regularization

**Notable Implementations**:
- `calc_loss_batch`: Computing cross-entropy loss for a single batch
- `calc_loss_loader`: Averaging loss over multiple batches
- `train_model_simple`: Main training loop with periodic evaluation
- `generate` (with temperature and top-k): Enhanced text generation with sampling strategies
- `download_and_load_gpt2`: Downloading pretrained weights from OpenAI
- `load_weights_into_gpt`: Transferring TensorFlow weights to PyTorch model

![GPT Processing Pipeline](ch05/01_main-chapter-code/ch05_compressed/gpt-process.webp)

![Top-K Sampling](ch05/01_main-chapter-code/ch05_compressed/topk.webp)

![GPT-2 Model Sizes](ch05/01_main-chapter-code/ch05_compressed/gpt-sizes.webp)

**Code Location**: `ch05/01_main-chapter-code/ch05.ipynb`

---

### Chapter 6: Finetuning for Text Classification

![Chapter Overview](ch06/01_main-chapter-code/ch06_compressed/chapter-overview.webp)

**Summary**: Chapter 6 demonstrates classification finetuning by adapting a pretrained GPT model for spam detection. It prepares the SMS Spam Collection dataset, balancing classes through undersampling and creating appropriate data loaders with padding. The chapter modifies the GPT architecture by replacing the 50,257-dimensional output head with a 2-class classification head while freezing most parameters except the last transformer block, final LayerNorm, and output layer. After training for 5 epochs with AdamW optimizer, the model achieves ~97% accuracy on the spam classification task, demonstrating effective transfer learning.

**Key Concepts**:
- **Classification finetuning**: Adapting pretrained LLM for specific classification tasks
- **Instruction finetuning vs classification finetuning**: Generalist vs specialist model training
- **Dataset preparation**: Downloading, balancing (undersampling), and splitting data
- **Class balancing**: Addressing imbalanced datasets (4825 ham, 747 spam) through sampling
- **Padding for variable-length sequences**: Using `<|endoftext|>` token for padding
- **Selective parameter freezing**: Training only last transformer block, LayerNorm, and output head
- **Last token classification**: Using final token's representation (contains full context)
- **Binary cross-entropy loss**: Training objective for 2-class classification

**Notable Implementations**:
- `download_and_unzip_spam_data`: Downloading and extracting SMS Spam dataset
- `create_balanced_dataset`: Undersampling majority class for balance
- `SpamDataset`: Custom Dataset with tokenization and padding
- `calc_accuracy_loader`: Computing classification accuracy
- Model modification: Replacing output head and unfreezing specific layers
- `train_classifier_simple`: Training loop tracking both loss and accuracy
- `classify_review`: Inference function for classifying new text

![Classification vs Instruction Finetuning](ch06/01_main-chapter-code/ch06_compressed/spam-non-spam.webp)

![Trainable Layers](ch06/01_main-chapter-code/ch06_compressed/trainable.webp)

![Training Loop Structure](ch06/01_main-chapter-code/ch06_compressed/training-loop.webp)

**Code Location**: `ch06/01_main-chapter-code/ch06.ipynb`

---

### Chapter 7: Finetuning to Follow Instructions

![Chapter Overview](ch07/01_main-chapter-code/ch07_compressed/overview.webp)

**Summary**: Chapter 7 teaches a pretrained LLM to follow natural language instructions through supervised instruction finetuning. It prepares an instruction dataset with input-output pairs formatted in Alpaca-style prompts (instruction, optional input, and response). The chapter implements custom batching to handle variable-length instruction-response pairs, using masking to compute loss only on response portions. After finetuning the 355M parameter GPT-2 model on the instruction dataset, it demonstrates evaluation using another LLM (Llama 3 via Ollama) to automatically score the quality of generated responses.

**Key Concepts**:
- **Instruction finetuning**: Training model to follow natural language instructions
- **Supervised instruction finetuning**: Using explicit input-output instruction pairs
- **Alpaca-style prompt formatting**: Structured format with "Below is an instruction..." template
- **Response masking**: Computing loss only on response tokens, not instruction tokens
- **Custom collate function**: Handling variable-length sequences with padding and masking
- **Target shifting**: Aligning targets to predict next token at each position
- **LLM-based evaluation**: Using larger model (Llama 3 8B) to score responses
- **Ollama integration**: Running local LLMs for automated evaluation
- **Multi-stage data processing**: Format → Tokenize → Pad → Mask → Batch

**Notable Implementations**:
- `format_input`: Creating Alpaca-style formatted prompts
- `InstructionDataset`: Custom Dataset handling instruction-response pairs
- `custom_collate_fn`: Advanced batching with padding, masking, and device placement
- Target mask creation: Setting instruction portion to -100 (ignored in loss)
- Loading 355M GPT-2 model: Using larger model for better instruction following
- `train_model_simple` (adapted): Modified training loop for instruction data
- `query_model` (Ollama): Interfacing with local Llama 3 for evaluation
- Automated scoring: Using LLM to assign numeric scores (0-100)

![Instruction Following Example](ch07/01_main-chapter-code/ch07_compressed/instruction-following.webp)

![Prompt Formatting Styles](ch07/01_main-chapter-code/ch07_compressed/prompt-style.webp)

![Detailed Batching Process](ch07/01_main-chapter-code/ch07_compressed/detailed-batching.webp)

**Code Location**: `ch07/01_main-chapter-code/ch07.ipynb`

---

## One-Paragraph Summary

This repository provides a comprehensive, hands-on implementation of building a Large Language Model from scratch, starting with fundamental text processing and progressing to advanced instruction-following capabilities. Chapter 2 establishes the data pipeline foundation with tokenization using BytePair Encoding and embedding layers, while Chapter 3 implements the crucial attention mechanism that enables contextual understanding. Chapter 4 assembles these components into a complete GPT architecture with layer normalization, GELU activations, feed-forward networks, and residual connections, creating a functional 124M parameter model. Chapter 5 demonstrates the pretraining process using next-token prediction on unlabeled data and shows how to load OpenAI's pretrained GPT-2 weights (up to 1558M parameters), introducing temperature scaling and top-k sampling for controlled text generation. Chapter 6 adapts the pretrained model for spam classification through selective parameter freezing and finetuning, achieving ~97% accuracy, while Chapter 7 implements instruction finetuning using Alpaca-style prompts with response masking, training a 355M parameter model to follow natural language instructions and introducing automated evaluation using Llama 3 via Ollama. Together, these chapters create a complete educational journey through the LLM development lifecycle, from basic text processing to producing helpful assistant-like models capable of understanding and executing complex instructions.

---

## Architecture Diagram

The mental model below summarizes the LLM development lifecycle covered in this repository:

![LLM Mental Model](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)

---

## Key Technical Achievements

1. **Complete GPT Implementation**: Full transformer architecture with multi-head attention, layer normalization, and residual connections
2. **From-Scratch Components**: Custom tokenizer, attention mechanisms, and training loops
3. **Weight Transfer**: Successfully loading OpenAI GPT-2 pretrained weights (124M-1558M parameters)
4. **Two Finetuning Approaches**: Classification (spam detection) and instruction-following
5. **Advanced Sampling**: Temperature scaling and top-k sampling for controlled generation
6. **Automated Evaluation**: LLM-based evaluation using local Ollama models
7. **Laptop-Scale Training**: All code designed to run on conventional laptops without specialized hardware

---

## Repository Structure

```
LLMs-from-scratch-main/
├── ch01/          # Understanding LLMs (no code)
├── ch02/          # Working with Text Data
├── ch03/          # Coding Attention Mechanisms
├── ch04/          # Implementing GPT Model
├── ch05/          # Pretraining on Unlabeled Data
├── ch06/          # Finetuning for Classification
├── ch07/          # Finetuning to Follow Instructions
├── appendix-A/    # Introduction to PyTorch
├── appendix-D/    # Training Loop Enhancements
└── appendix-E/    # Parameter-efficient Finetuning with LoRA
```

---

## Additional Resources

- **Book**: *Build a Large Language Model (From Scratch)* by Sebastian Raschka
- **Publisher**: Manning Publications
- **ISBN**: 9781633437166
- **GitHub**: https://github.com/rasbt/LLMs-from-scratch
- **Video Course**: 17+ hour companion video course available

---

*Report generated: 2025-10-26*
