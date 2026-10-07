# My_GPT — Building a GPT-Style Language Model from Scratch

A hands-on implementation of a small GPT-style language model built with **Python and PyTorch** to understand the internal mechanics of modern transformer-based language models.

This project focuses on implementing the core architecture step by step rather than relying on a pretrained model or high-level LLM framework.

---

## Project Goals

The goal of this project is to understand how a GPT-style decoder-only transformer works by implementing the major components directly in PyTorch:

- Tokenization and vocabulary construction
- Training batches and sequence handling
- Token embeddings
- Positional embeddings
- Query, Key, and Value projections
- Scaled dot-product self-attention
- Causal masking
- Multi-head attention
- Feed-forward neural networks
- Transformer blocks
- Residual connections
- Layer normalization
- GPT model assembly
- Cross-entropy training objective
- AdamW optimization
- Validation / evaluation
- Autoregressive text generation

---

## Why I Built This

Using an LLM through an API is useful, but it does not by itself explain what happens inside the model.

I built this project to move from **using language models** to understanding the architecture behind them:

```text
Tokens
  ↓
Token Embeddings + Positional Embeddings
  ↓
Multi-Head Causal Self-Attention
  ↓
Feed-Forward Network
  ↓
Transformer Blocks
  ↓
Linear Output Head
  ↓
Next-Token Probabilities
  ↓
Autoregressive Text Generation
```

---

## Repository Structure

```text
my_GPT/
│
├── My_GPT_Step_by_Step.ipynb
├── my_gpt.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── gpt_outputs/
│   ├── experiments.csv
│   ├── generated_text.txt
│   └── mini_gpt.pt
│
├── assets/
│   └── architecture.md
│
└── scripts/
    └── inspect_outputs.py
```

### `My_GPT_Step_by_Step.ipynb`

Educational notebook that walks through the architecture component by component.

### `my_gpt.ipynb`

More compact implementation bringing the components together into a trainable GPT-style model.

### `gpt_outputs/`

Contains experiment logs, generated text, and a saved model checkpoint.

---

## Core Architecture

A decoder-only GPT-style transformer predicts the next token using only the tokens that appeared before it.

The causal mask prevents each token from attending to future tokens.

Conceptually:

```text
Input Tokens
    ↓
Embedding Layer
    ↓
Positional Embedding
    ↓
┌──────────────────────────────┐
│ Transformer Block            │
│                              │
│ LayerNorm                    │
│     ↓                        │
│ Multi-Head Causal Attention  │
│     ↓                        │
│ Residual Connection          │
│     ↓                        │
│ LayerNorm                    │
│     ↓                        │
│ Feed-Forward Network         │
│     ↓                        │
│ Residual Connection          │
└──────────────────────────────┘
    ↓
Repeated Transformer Blocks
    ↓
Final LayerNorm
    ↓
Linear Language Model Head
    ↓
Token Logits
    ↓
Softmax / Sampling
    ↓
Generated Text
```

---

## Self-Attention

For each token embedding, the model creates:

- **Query (Q)** — what the token is looking for
- **Key (K)** — what information the token offers
- **Value (V)** — the actual information passed forward

Attention is computed approximately as:

```text
Attention(Q, K, V) =
softmax(QKᵀ / √d) · V
```

A **causal mask** is applied before the softmax so a token cannot see future tokens.

---

## Multi-Head Attention

Instead of computing attention once, GPT uses multiple attention heads.

Each head can learn different relationships in the text.

```text
Input
  ↓
Head 1 ─┐
Head 2 ─┤
Head 3 ─┤ → Concatenate → Linear Projection
Head 4 ─┘
```

---

## Training Objective

The model learns by predicting the next token.

For an input such as:

```text
The cat sat on the
```

the target may be:

```text
cat sat on the mat
```

The loss is calculated using **cross-entropy loss**, and the model parameters are updated with **AdamW**.

---

## Text Generation

After training, the model generates text autoregressively:

1. Provide a prompt
2. Run the model
3. Obtain next-token logits
4. Convert logits into probabilities
5. Sample or select a token
6. Append the new token
7. Repeat

Example workflow:

```text
Prompt
  ↓
GPT Model
  ↓
Next-token probabilities
  ↓
Sample token
  ↓
Append token
  ↓
Repeat
```

Generated samples from the project are stored in:

```text
gpt_outputs/generated_text.txt
```

---

## Experiments

Training / model experiments are stored in:

```text
gpt_outputs/experiments.csv
```

This file can be used to compare different configurations such as:

- embedding dimensions
- number of attention heads
- number of transformer layers
- context length
- learning rate
- training steps
- loss values
- generation behavior

Run:

```bash
python scripts/inspect_outputs.py
```

to inspect the current experiment log and generated-text file.

---

## Saved Model

The repository contains a saved model checkpoint:

```text
gpt_outputs/mini_gpt.pt
```

This represents the trained state of the small GPT model produced during experimentation.

> This is an educational small-scale model and is not intended to compete with production LLMs.

---

## Technologies

- Python
- PyTorch
- Neural Networks
- Transformer Architecture
- Self-Attention
- Multi-Head Attention
- Natural Language Processing
- Language Modeling
- Jupyter Notebook
- Git / GitHub

---

## What This Project Demonstrates

This project demonstrates practical understanding of:

- how transformer-based language models process sequences
- how attention works internally
- why causal masking is needed in GPT
- how multiple attention heads operate in parallel
- how transformer blocks are assembled
- how language models are trained using next-token prediction
- how autoregressive generation works
- how PyTorch modules, tensors, optimization, and GPU execution fit together

---

## Limitations

This is a learning-focused implementation.

The model:

- is trained on a relatively small dataset
- has far fewer parameters than production LLMs
- is not intended for factual or production use
- may generate repetitive or incoherent text
- exists primarily to demonstrate transformer engineering fundamentals

---

## Next Improvements

Planned improvements include:

- cleaner configuration management
- reproducible training scripts outside the notebook
- better experiment tracking
- validation-loss visualization
- top-k and temperature-based generation
- checkpoint loading / inference script
- larger training corpus
- improved tokenizer
- model architecture visualization
- performance benchmarking

---

## Portfolio Context

This project is part of my AI Engineering portfolio, focused on:

**LLMs • Generative AI • RAG • AI Agents • NLP • Deep Learning**

It complements higher-level LLM application development by demonstrating how the core transformer architecture works internally.
