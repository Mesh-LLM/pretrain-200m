# Pretrain 200M LLM on Goose Session Data

From-scratch pretrain of a 200M parameter Llama-style transformer on FineWeb-Edu + goose session data.

## Architecture
- d_model: 768, n_layers: 12, n_heads: 12, head_dim: 64
- Vocab: 50,257 (GPT-2 BPE), max_seq_len: 1024
- Tied embeddings, RoPE, SwiGLU, RMSNorm

## Data
- 93% FineWeb-Edu (downloaded directly on the node)
- 7% goose session data (included in this repo as `goose_all_combined.txt.gz`)

## Run
```bash
# Decompress goose data
gunzip -k goose_all_combined.txt.gz

# Tokenize (downloads FineWeb-Edu + tokenizes everything)
python3 pretrain_200m.py --tokenize

# Train
python3 pretrain_200m.py
```
