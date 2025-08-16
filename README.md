# onnx_embedding_models
utilities for loading and running text embeddings with onnx

## UPDATE: 8-16-2025

Sentence Transformers now appears to have [robust support](https://sbert.net/docs/sentence_transformer/usage/efficiency.html#onnx) for running arbitrary embedding models with ONNX backend. As such, I would recommend using Sentence Transformers for embeddings inference with ONNX. This package has not been maintained and has some bugs affecting accuracy.

## CHANGELOG

0.0.14 - 2024-06-19
- support for loading from any huggingface repo with from_pretrained (provided it has a tokenizer and some onnx model in it)
- added mxbai-embed-large to the model registry
- embedding model `.encode` method now returns numpy arrays by default instead of lists
