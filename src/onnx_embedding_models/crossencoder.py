import os
from typing import Literal
from tqdm.auto import trange
from .utils import dict_slice
import numpy as np

class CrossEncoder:
    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: str,
        max_length: int,
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
        self.session = ort.InferenceSession(onnx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def rerank(self, texts: list[str], candidate_labels: list[list[str]]):
        """
        Rerank the candidate labels for each text.
        """
        results = []
        # allow for candidate labels to be (label, score) tuples - in that case, extract the labels
        if not isinstance(candidate_labels[0][0], str):
            candidate_labels = [[tup[0] for tup in lab_list] for lab_list in candidate_labels]

        # flatten samples while keeping track of the original text index
        samples = []
        for i, (text, labels) in enumerate(zip(texts, candidate_labels)):
            samples.extend([
                (i, text, label) for label in labels
            ])

        # tokenize everything at once
        tokenized = self.tokenizer.batch_encode_plus(
            [(text, label) for _, text, label in samples],
            add_special_tokens=True, 
            max_length=512, 
            # no padding
            padding=False,
            truncation='only_first', # truncate only query, not the label
            return_tensors=None
        )
        # process one at a time (best for cpu)
        all_logits = []
        for i in trange(0, len(samples)):
            batch = dict_slice(tokenized, i)
            logits = self.session.run(None, batch)[0][0] # list of length 2
            all_logits.append(logits)
        
        all_logits = np.array(all_logits)

        # numerically stable softmax
        all_probs = np.exp(all_logits - all_logits.max(axis=1, keepdims=True))
        all_probs /= all_probs.sum(axis=1, keepdims=True)

        # zip with original index and group
        # group by index
        grouped = {}
        assert len(samples) == len(all_probs), "Mismatch in number of samples and output probabilities"
        for orig_sample, probs in zip(samples, all_probs):
            i, text, label = orig_sample
            if i not in grouped:
                grouped[i] = []
            grouped[i].append((label, probs[1]))
        
        results = [
            sorted(grouped[i], key=lambda x: x[1], reverse=True) for i in range(len(texts))
        ]
        
        return results

    # def predict_scores(self, texts: list[str], candidate_labels: Optional[list[list[str]]] = None):
    #     """
    #     Predict the relevance scores for each text and label pair. If provided, will only 
    #     rerank the candidate labels.
    #     """
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     tokenizer = AutoTokenizer.from_pretrained(backbones[self.base_model])
    #     model = self.model
    #     model.eval()
    #     with torch.no_grad():
    #         batch = prepare_batch(texts, candidate_labels, tokenizer, device)
    #         scores = model(**batch).logits
    #     return scores