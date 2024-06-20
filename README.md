# onnx_embedding_models
utilities for loading and running text embeddings with onnx

## CHANGELOG

0.0.14 - 2024-06-19
- support for loading from any huggingface repo with from_pretrained (provided it has a tokenizer and some onnx model in it)
- added mxbai-embed-large to the model registry
- embedding model `.encode` method now returns numpy arrays by default instead of lists
