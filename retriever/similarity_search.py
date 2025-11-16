# retriever/similarity_search.py
"""
Similarity Search 模組：
包一層，方便之後想換成其他 vector DB (Milvus / pgvector 等)。
"""

from typing import List, Dict, Tuple
import numpy as np

from .vector_store import VectorStore

def similarity_search(query_vecs: np.ndarray,
                      vector_store: VectorStore,
                      top_k: int = 10
                      ) -> List[List[Dict]]:
    """
    將 VectorStore.search() 的結果整理成 list[dict] 格式。

    輸入：
        query_vecs: (num_queries, dim)
    輸出：
        results: List[ List[ {"score": float, **metadata} ] ]
    """
    raw = vector_store.search(query_vecs, top_k=top_k)
    all_results: List[List[Dict]] = []
    for q_res in raw:
        one_list = []
        for score, meta in q_res:
            item = dict(meta)
            item["score"] = score
            one_list.append(item)
        all_results.append(one_list)
    return all_results

