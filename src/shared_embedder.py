import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

DEFAULT_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
PROJECT_DIR = os.environ.get("WASTEWISE_PROJECT_PATH", DEFAULT_PROJECT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "Models", "sentence_transformer")

_onnx_embedder = None


class ONNXSentenceEmbedder:
    """Zero-PyTorch, low-memory SentenceTransformer inference using ONNX Runtime."""

    def __init__(self, model_dir: str = MODEL_DIR):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Sentence Transformer directory not found: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
        if not os.path.isfile(onnx_path):
            onnx_path = os.path.join(model_dir, "model.onnx")

        if not os.path.isfile(onnx_path):
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            ort_model = ORTModelForFeatureExtraction.from_pretrained(model_dir, export=True)
            ort_model.save_pretrained(model_dir)
            onnx_path = os.path.join(model_dir, "model.onnx")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        encoded = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=256, return_tensors="np"
        )
        ort_inputs = {k: v for k, v in encoded.items() if k in self.input_names}
        token_embeddings = self.session.run(None, ort_inputs)[0]

        # Mean pooling matching SentenceTransformer
        mask = np.expand_dims(encoded["attention_mask"], -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask, axis=1)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings


def get_shared_embedder() -> ONNXSentenceEmbedder:
    global _onnx_embedder
    if _onnx_embedder is None:
        _onnx_embedder = ONNXSentenceEmbedder()
    return _onnx_embedder


class SharedLlamaIndexEmbedding(BaseEmbedding):
    """LlamaIndex wrapper around the single shared ONNX embedder."""

    _embedder: ONNXSentenceEmbedder = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embedder = get_shared_embedder()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embedder.encode([query], normalize_embeddings=True)[0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embedder.encode([text], normalize_embeddings=True)[0].tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)