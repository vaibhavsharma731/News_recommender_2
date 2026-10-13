"""One shared copy of each neural model per process."""
import threading
import config

_lock = threading.Lock()
_cache = {}


def _get(key, factory):
    with _lock:
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


def get_embedder(model_name=None):
    model_name = model_name or config.EMBEDDING_MODEL

    def build():
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = config.EMBED_MAX_SEQ_LEN
        return model

    return _get(("embed", model_name), build)


def get_cross_encoder(model_name=None):
    model_name = model_name or config.RERANKER_MODEL

    def build():
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name, max_length=config.RERANK_MAX_LEN)

    return _get(("rerank", model_name), build)