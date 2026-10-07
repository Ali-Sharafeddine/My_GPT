# My_GPT Architecture

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Token Embedding + Positional Embedding
 ↓
Transformer Block × N
 │
 ├── LayerNorm
 ├── Multi-Head Causal Self-Attention
 ├── Residual Connection
 ├── LayerNorm
 ├── Feed-Forward Network
 └── Residual Connection
 ↓
Final LayerNorm
 ↓
Linear LM Head
 ↓
Vocabulary Logits
 ↓
Next-Token Sampling
 ↓
Generated Text
```

## Attention Path

```text
Input embeddings
     │
     ├── Linear → Query
     ├── Linear → Key
     └── Linear → Value
              │
              ↓
        Q × Kᵀ / √d
              ↓
         Causal Mask
              ↓
           Softmax
              ↓
        Attention Weights
              ↓
          × Values
              ↓
        Attention Output
```
