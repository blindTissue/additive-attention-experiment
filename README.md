# Additive Attention Experiment

This experiment explores two design choices from the "Attention is All You Need" paper:

1. **Positional Embeddings**: Sinusoidal vs. Learnable
2. **Attention Mechanism**: Additive (Bahdanau-style) vs. Dot-Product (Scaled)

## Background

The original Transformer paper discussed both sinusoidal and learned positional embeddings, ultimately using sinusoidal for their ability to (presumably) extrapolate to longer sequences. For attention mechanisms, the paper compared additive attention (used in earlier seq2seq models) with dot-product attention.

## Experiment Goals

1. **Verify if sinusoidal and learnable positional embeddings achieve similar performance** on a character-level language modeling task
2. **Test if additive attention is feasible in modern Transformers** with memory-efficient implementation

## Implementation

- **Model**: GPT-style decoder-only Transformer, adapted from Andrej Karpathy's [minGPT](https://github.com/karpathy/ng-video-lecture/tree/master) 
- **Dataset**: Tiny Shakespeare, also from [minGPT](https://github.com/karpathy/ng-video-lecture/tree/master)
- **Architecture**: 6 layers, 6 heads, 384 embedding dimensions
- **Training**: 5000 steps, batch size 64, context length 256

### Modified Additive Attention

**Important Note**: This implementation uses a **modified** version of additive attention that is memory-efficient but mathematically different from the standard Bahdanau attention.

- **Standard additive attention**: `w(tanh(q + k))` - requires materializing a `(B, T, T, head_size)` tensor (~64x more memory than dot-product)
- **This implementation**: `w(tanh(q)) + w(tanh(k))` - only requires `(B, T, T)` tensors

Since `tanh(q + k) ≠ tanh(q) + tanh(k)`, this is a different attention mechanism. It can be viewed as a separable approximation of additive attention, where query and key contributions are computed independently and then combined additively.

## Results

Training produces individual loss plots for each configuration:
- `loss_plot_sinusoidal=True_additive=False.png`
- `loss_plot_sinusoidal=True_additive=True.png`
- `loss_plot_sinusoidal=False_additive=False.png`
- `loss_plot_sinusoidal=False_additive=True.png`

## Usage

```bash
# Train with different configurations
python main.py --use_sinusoidal_pos_emb true --use_additive_attention false
python main.py --use_sinusoidal_pos_emb true --use_additive_attention true
python main.py --use_sinusoidal_pos_emb false --use_additive_attention false
python main.py --use_sinusoidal_pos_emb false --use_additive_attention true
```

## Dependencies

- Python >=3.11
- PyTorch
- matplotlib

```bash
pip install torch matplotlib
```