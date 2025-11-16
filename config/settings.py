# config/settings.py

# ===== Chunking =====
CHUNK_MAX_TOKENS = 1000     # 每個 chunk 的最大字元/token 長度

# ===== Retrieval =====
SEARCH_TOPK = 50           # 相似度搜尋取前 N 個
RERANK_TOPK = 10           # reranker 最終留下前 N 個（最後返回的結果數）


DEFAULT_USE_RERANK = False
