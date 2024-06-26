registry = {
    # 3 layers, 384-dim
    "bge-micro": {
        "repo": "TaylorAI/bge-micro-v2",
        "path_in_repo": "onnx/model_quantized.onnx",
        "max_length": 512,
        "pooling_strategy": "mean",
        "ndim": 384,
    },
    # 6 layers, 384-dim
    "gte-tiny": {
        "repo": "TaylorAI/gte-tiny",
        "path_in_repo": "onnx/model_quantized.onnx",
        "max_length": 512,
        "pooling_strategy": "mean",
        "ndim": 384,
    },
    "minilm-l6": {
        "repo": "Xenova/all-MiniLM-L6-v2",
        "path_in_repo": "onnx/model_quantized.onnx",
        "max_length": 512,
        "pooling_strategy": "mean",
        "ndim": 384,
    },
    "snowflake-xs": {
        "repo": "Snowflake/snowflake-arctic-embed-xs",
        "path_in_repo": "onnx/model_quantized.onnx",
        "max_length": 512,
        "pooling_strategy": "cls",
        "normalize": True,
        "ndim": 384,
    },
    # 12 layers, 384-dim
    "minilm-l12": {
        "repo": "Xenova/all-MiniLM-L12-v2",
        "path_in_repo": "onnx/model_quantized.onnx",
        "max_length": 512,
        "pooling_strategy": "mean",
        "ndim": 384,
    },
    "bge-small": {
        "repo": "neuralmagic/bge-small-en-v1.5-quant",
        "path_in_repo": "model.onnx",
        "max_length": 512,
        "pooling_strategy": "first",
        "ndim": 384,
    },
    "multilingual-e5-small": {
        "repo": "intfloat/multilingual-e5-small",
        "path_in_repo": "onnx/model.onnx",
        "max_length": 512,
        "pooling_strategy": "mean",
        "ndim": 384,
    },
    # 12 layers, 768-dim
    "bge-base": {
        "repo": "neuralmagic/bge-base-en-v1.5-quant",
        "path_in_repo": "model.onnx",
        "max_length": 512,
        "pooling_strategy": "first",
        "ndim": 768,
    },
    # 24 layers, 1024-dim
    "bge-large": {
        "repo": "neuralmagic/bge-large-en-v1.5-quant",
        "path_in_repo": "model.onnx",
        "max_length": 512,
        "pooling_strategy": "first",
        "ndim": 1024,
    },
    "mixedbread-large": {
        "repo": 'mixedbread-ai/mxbai-embed-large-v1',
        "path_in_repo": "onnx/model_quantized.onnx",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024
    },
    ### SPARSE MODELS ###
    "distilbert-splade": {
        "repo": "andersonbcdefg/distilbert-splade-onnx",
        "path_in_repo": "model.onnx",
        "max_length": 512,
        "pooling_strategy": "max"
    }
}
