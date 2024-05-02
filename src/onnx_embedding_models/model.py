import os
import abc
import tqdm
import shutil
import tempfile
from typing import Literal, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from .registry import registry

class EmbeddingModelBase(abc.ABC):
    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: str,
        max_length: int,
        pooling_strategy: Literal["mean", "first", "cls", "max", None],
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
        if destination is not None:
            os.makedirs(destination, exist_ok=True)
            if not os.path.exists(os.path.join(destination, "model.onnx")):
                cls.download_from_registry(model_id, destination)
                instance = cls(
                    os.path.join(destination, "model.onnx"),
                    destination,
                    max_length=registry[model_id]["max_length"],
                    pooling_strategy=registry[model_id]["pooling_strategy"],
                    normalize=normalize,
                    intra_op_num_threads=intra_op_num_threads,
                    thread_spinning=thread_spinning,
                )
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                cls.download_from_registry(model_id, tmpdir)
                instance = cls(
                    os.path.join(tmpdir, "model.onnx"),
                    tmpdir,
                    max_length=registry[model_id]["max_length"],
                    pooling_strategy=registry[model_id]["pooling_strategy"],
                    normalize=normalize,
                    intra_op_num_threads=intra_op_num_threads,
                    thread_spinning=thread_spinning,
                )
        return instance


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

    @staticmethod
    def dict_slice(d, idx):
        """
        Slice a dictionary of jagged arrays, returning a dictionary of singleton numpy arrays.
        """
        return {k: np.array(v[idx]).reshape(1, -1) for k, v in d.items()}
    
    def _forward_one(
        self,
        inputs: dict[str, np.ndarray]
    ):
        outputs = self.session.run(None, inputs)
        if len(outputs) == 2:
            hidden_states, pooler_output = outputs
        else:
            hidden_states, pooler_output = outputs[0], None
        return hidden_states, pooler_output

    def _forward_batch(
        self,
        inputs: dict[str, list[list[int]]],
        show_progress: bool = True,
        chunk_size: Optional[int] = None,
    ):
        hidden_states_list = []
        pooler_output_list = []
        num_inputs = len(inputs["input_ids"])
        for i in tqdm.tqdm(range(num_inputs), disable=not show_progress):
            # hidden_states, pooler_output = self._forward_one(self.dict_slice(inputs, i))
            # hidden_states_list.append(hidden_states)
            # pooler_output_list.append(pooler_output)
            if chunk_size is not None:
                input_ids = inputs["input_ids"][i]
                attention_mask = inputs["attention_mask"][i]
                token_type_ids = inputs["token_type_ids"][i] if "token_type_ids" in inputs else None
                input_ids_chunks = [
                    input_ids[i : i + chunk_size] for i in range(0, len(input_ids), chunk_size)
                ]
                attention_mask_chunks = [
                    attention_mask[i : i + chunk_size]
                    for i in range(0, len(attention_mask), chunk_size)
                ]
                if token_type_ids is not None:
                    token_type_ids_chunks = [
                        token_type_ids[i : i + chunk_size]
                        for i in range(0, len(token_type_ids), chunk_size)
                    ]
                else:
                    token_type_ids_chunks = [None] * len(input_ids_chunks)

                hidden_states_chunks = []
                for chunk_ids, chunk_mask, chunk_ttids in zip(input_ids_chunks, attention_mask_chunks, token_type_ids_chunks):
                    chunk_inputs = {
                        "input_ids": np.array([chunk_ids]),
                        "attention_mask": np.array([chunk_mask]),
                    }
                    if chunk_ttids is not None:
                        chunk_inputs["token_type_ids"] = np.array([chunk_ttids])
                    hidden_states_chunk, pooler_output_chunk = self._forward_one(chunk_inputs)
                    hidden_states_chunks.append(hidden_states_chunk) # B, chunk_size, D

                hidden_states = np.concatenate(hidden_states_chunks, axis=1)
                pooler_output = None # can't aggregate pooler_output for chunked
            else:
                hidden_states, pooler_output = self._forward_one(self.dict_slice(inputs, i))
            
            hidden_states_list.append(hidden_states)
            pooler_output_list.append(pooler_output)

        return hidden_states_list, pooler_output_list

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
        if self.pooling_strategy == "max":
            pooled = np.max(
                last_hidden_state + (1 - np.expand_dims(mask, -1)) * -1e6, axis=1
            )
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

    def encode(
        self,
        texts: list[str],
        return_numpy=False,
        show_progress=True,
        **kwargs,
    ):
        pass
    
    def __call__(self, texts: list[str], return_numpy=False, show_progress=True, **kwargs):
        if isinstance(texts, str):
            return self.encode([texts], return_numpy=return_numpy, show_progress=show_progress, **kwargs)[0]
        elif isinstance(texts, list):
            return self.encode(texts, return_numpy=return_numpy, show_progress=show_progress, **kwargs)

class EmbeddingModel(EmbeddingModelBase):
    def encode(
        self,
        texts: list[str],
        return_numpy: bool = False,
        show_progress: bool = True,
    ):
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        ) # dont return tensors, this adds unnecessary padding
        hidden_states_list, pooler_output_list = self._forward_batch(inputs, show_progress=show_progress)
        output_embs = [
            self._pool(hidden_states, pooler_output).flatten()
            for hidden_states, pooler_output in zip(hidden_states_list, pooler_output_list)
        ]
        if return_numpy:
            return np.array(output_embs)
        return output_embs

ONNXEmbeddingModel = EmbeddingModel

class SpladeModel(EmbeddingModelBase):

    @staticmethod
    def _create_sparse_embedding(
        activations: np.ndarray,
        max_dims: int,
    ):
        B, V = activations.shape
        topk_indices = np.argsort(activations, axis=-1)[:, -max_dims:] # B, max_dims
        sparse_embeddings = np.zeros((B, V), dtype=np.float32)
        for i in range(B):
            sparse_embeddings[i, topk_indices[i]] = activations[i, topk_indices[i]]

        return sparse_embeddings
    
    def encode(
        self,
        texts: list[str],
        return_numpy: bool = True,
        show_progress: bool = True,
        max_dims: Union[int, str] = "auto",
        return_sparse: bool = False,
        chunk_size: Optional[int] = None,
    ):
        if return_numpy and return_sparse:
            raise ValueError("Can't return both numpy and sparse embeddings")
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        ) # dont return tensors, this adds unnecessary padding
        hidden_states_list, _ = self._forward_batch(inputs, show_progress=show_progress, chunk_size=chunk_size)
        # relu + max pool
        sparse_activations = np.array([
            self._pool(np.maximum(hidden_states, 0)).flatten()
            for hidden_states in hidden_states_list
        ]) # B, V

        # topk
        max_dims = self.max_length if max_dims == "auto" else max_dims
        sparse_embs = self._create_sparse_embedding(sparse_activations, max_dims)

        # Select top-k activations
        if return_numpy:
            return sparse_embs
        elif return_sparse:
            return csr_matrix(sparse_embs)
        return sparse_embs.tolist()