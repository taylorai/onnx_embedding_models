import os
import shutil
import tempfile
from typing import Literal, Optional

import numpy as np

from .registry import registry


class EmbeddingModel:
    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: str,
        max_length: int,
        pooling_strategy: Literal["mean", "first", "cls"],
        normalize: bool,
        intra_op_num_threads: int = 0,
        thread_spinning: bool = True,
    ):
        """
        This assumes the model file is already downloaded.
        Use classmethods / utilities to load from registry or Hub.
        Recommended to also pre-download tokenizer to avoid cold start.
        """
        import onnxruntime as ort
        from transformers import AutoTokenizer
        sess_options = ort.SessionOptions()
        # options to make it faster
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.add_session_config_entry(
            "session.intra_op.allow_spinning",
            "1" if thread_spinning else "0",
        )

        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.session = ort.InferenceSession(onnx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    @classmethod
    def from_registry(
        cls,
        model_id: str,
        normalize: bool = True,
        destination: Optional[str] = None,
        intra_op_num_threads: int = 0,
        thread_spinning: bool = True,
    ):
        """
        Downloads a model from the pre-selected registry and returns an instance.
        """
        destination = tempfile.mkdtemp() if destination is None else destination
        cls.download_from_registry(model_id, destination)
        return cls(
            os.path.join(destination, "model.onnx"),
            destination,
            max_length=registry[model_id]["max_length"],
            pooling_strategy=registry[model_id]["pooling_strategy"],
            normalize=normalize,
            intra_op_num_threads=intra_op_num_threads,
            thread_spinning=thread_spinning,
        )


    @staticmethod
    def download_from_registry(model_id: str, destination: str):
        """
        Downloads a model from the pre-selected registry.
        """
        import huggingface_hub
        from transformers import AutoTokenizer
        os.makedirs(destination, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_file = huggingface_hub.hf_hub_download(
                repo_id=registry[model_id]["repo"],
                filename=registry[model_id]["path_in_repo"],
                local_dir=tmpdir,
                local_dir_use_symlinks=False,  # we want the actual onnx files there
            )
            shutil.move(onnx_file, os.path.join(destination, "model.onnx"))
        tokenizer = AutoTokenizer.from_pretrained(registry[model_id]["repo"])
        tokenizer.save_pretrained(destination)

    @staticmethod
    def download_from_hub(
        model_repo: str, model_path_in_repo: str, tokenizer_repo: str, destination: str
    ):
        """
        Downloads a model from a custom Hub repo, for those not in the registry.
        """
        import huggingface_hub
        from transformers import AutoTokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_file = huggingface_hub.hf_hub_download(
                repo_id=model_repo,
                filename=model_path_in_repo,
                local_dir=tmpdir,
                local_dir_use_symlinks=False,  # we want the actual onnx files there
            )
            shutil.move(onnx_file, os.path.join(destination, "model.onnx"))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
        tokenizer.save_pretrained(destination)

    def _pool(
        self,
        last_hidden_state: np.ndarray,  # B, L, D
        pooler_output: Optional[np.ndarray] = None,  # B, D
        mask: Optional[np.ndarray] = None,  # B, L
    ):
        # hiddens: B, L, D; mask: B, L
        if mask is None:
            mask = np.ones(last_hidden_state.shape[:2])
        if self.pooling_strategy == "mean":
            pooled = np.sum(
                last_hidden_state * np.expand_dims(mask, -1), axis=1
            ) / np.sum(mask, axis=-1, keepdims=True)
        elif self.pooling_strategy == "first":
            pooled = last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "cls":
            if pooler_output is None:
                # use first token w/ no pooling linear layer
                pooled = last_hidden_state[:, 0, :]
            else:
                pooled = pooler_output
        else:
            raise NotImplementedError(
                f"pooling strategy {self.pooling_strategy} not implemented"
            )
        if self.normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)

        return pooled

    def embed(
        self, 
        text: str,
        return_numpy: bool = False,
    ):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="np",
        )
        outputs = self.session.run(None, {k: v for k, v in inputs.items()})
        if len(outputs) == 2:
            hidden_states, pooler_output = outputs
        else:
            hidden_states, pooler_output = outputs[0], None

        emb = self._pool(hidden_states, pooler_output).flatten()
        if return_numpy:
            return emb
        return emb.tolist()

    def embed_batch(
        self,
        texts: list[str],
        return_numpy: bool = False,
    ):
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        ) # dont return tensors, this adds unnecessary padding
        output_embs = []
        for i in range(len(texts)):
            outputs = self.session.run(None, {k: np.array(v[i]).reshape(1, -1) for k, v in inputs.items()})
            if len(outputs) == 2:
                hidden_states, pooler_output = outputs
            else:
                hidden_states, pooler_output = outputs[0], None

            emb = self._pool(hidden_states, pooler_output).flatten()
            output_embs.append(emb.tolist())

        if return_numpy:
            return np.array(output_embs)
        
        return output_embs
    
    def encode(self, texts: list[str], return_numpy=False):
        return self.embed_batch(texts, return_numpy=return_numpy)
    
    def __call__(self, texts: list[str], return_numpy=False):
        if isinstance(texts, str):
            return self.embed(texts, return_numpy=return_numpy)
        elif isinstance(texts, list):
            return self.embed_batch(texts, return_numpy=return_numpy)

        
ONNXEmbeddingModel = EmbeddingModel