# My_GPT

A small educational GPT-style character-level language model built from scratch with PyTorch.

## Project workflow

- Checked Python, PyTorch, and GPU availability.
- Selected CUDA when available, otherwise CPU.
- Prepared a practice text corpus.
- Built a vocabulary of unique characters.
- Implemented character encoding and decoding.
- Split the data into training and validation sets.
- Created batches with next-character targets.
- Added trainable token and position embeddings.
- Implemented queries, keys, and values for self-attention.
- Applied causal masking to block future characters.
- Combined multiple attention heads.
- Built transformer blocks with feed-forward layers, normalization, and residual connections.
- Assembled the complete GPT-style model.
- Produced scores for possible next characters.
- Calculated cross-entropy loss.
- Trained the model using backpropagation and AdamW.
- Tracked training and validation loss.
- Retained the checkpoint with the lowest measured validation loss.
- Generated text through repeated next-character sampling.
- Compared attention heads, transformer layers, and embedding sizes.
- Saved experiment results, generated text, and model weights locally.
- Reloaded the saved model and verified matching predictions.

## How to run

1. Open `My_GPT_Step_by_Step.ipynb` in VS Code.
2. Select a Python environment with PyTorch installed.
3. Run the cells in order with `QUICK_RUN = True`.
4. Set `QUICK_RUN = False` for longer training.

## Limitations

This is a learning project using a small, repetitive practice corpus.
Its results do not demonstrate general conversational ability.
Model weights and virtual environments are excluded from Git.