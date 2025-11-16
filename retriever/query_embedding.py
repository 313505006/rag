# retriever/query_embedding.py
"""
Query Embedding 模組：
把擴展後的多個 query 轉成向量。
為了一致性，沿用 FileEmbedder。
"""

from typing import List
import numpy as np
from .file_embedding import FileEmbedder

def embed_query(queries: List[str], embedder: FileEmbedder | None = None) -> np.ndarray:
    """
    輸入：多個 query 字串
    輸出：np.ndarray (num_queries, dim)
    """
    if embedder is None:
        embedder = FileEmbedder()
    return embedder.encode(queries)
