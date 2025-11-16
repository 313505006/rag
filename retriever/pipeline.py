# retriever/pipeline.py
"""
RetrieverPipeline：
整合所有模組，對應你架構圖中的 Retriever 區塊。

- index_files(): 用於離線建立索引（Files → Abstracting → Embedding → 向量 DB）
- retrieve():    線上查詢流程（Query → Expand → Embedding → Similarity Search → Reranking）
"""
"""
RetrieverPipeline：
新增兩種模式：
1. use_rerank = True  →  Query → Embedding → Similarity Search → Reranker
2. use_rerank = False →  Query → Embedding → Similarity Search（直接結果）
"""

from typing import List, Dict
from config.settings import SEARCH_TOPK, RERANK_TOPK, DEFAULT_USE_RERANK

from .file_abstractor import abstract_files
from .file_embedding import embed_files, FileEmbedder
from .query_expand import expand_query
from .query_embedding import embed_query
from .vector_store import VectorStore
from .similarity_search import similarity_search
from .reranker import rerank_results, Reranker


class RetrieverPipeline:
    def __init__(self,
                 vector_db_path: str,
                 embedder: FileEmbedder | None = None,
                 reranker: Reranker | None = None):

        self.vector_store = VectorStore(vector_db_path)
        self.embedder = embedder or FileEmbedder()
        self.reranker = reranker or Reranker()

    # ---------------------------------------------------------
    #  單次建立索引（preprocess 時用）
    # ---------------------------------------------------------
    def index_files(self, files: List[Dict], max_chars: int = 2000):
        abstracts = abstract_files(files, max_chars=max_chars)
        embeddings, metadatas = embed_files(abstracts, embedder=self.embedder)
        self.vector_store.add_embeddings(embeddings, metadatas)

    # ---------------------------------------------------------
    #  查詢（主功能：use_rerank 控制是否啟用重排序）
    # ---------------------------------------------------------
    def retrieve(self,
                 query: str,
                 top_k: int = None,
                 use_rerank: bool = DEFAULT_USE_RERANK) -> List[Dict]:
        """
        use_rerank=True  →  similarity search → rerank
        use_rerank=False →  similarity search（直接回傳結果）
        """

        final_top_k = top_k if top_k is not None else RERANK_TOPK

        # 1. Query Expand
        expanded_queries = expand_query(query)
        if not expanded_queries:
            return []

        # 2. Encoding Query
        q_vecs = embed_query(expanded_queries, embedder=self.embedder)

        # 3. Similarity Search
        candidates = similarity_search(
            q_vecs,
            self.vector_store,
            top_k=SEARCH_TOPK
        )

        # 目前只用第一組 query
        candidates = candidates[0]

        # ---------------------------------------------------------
        #  不使用 Reranker：直接依 similarity 排序後回傳
        # ---------------------------------------------------------
        if not use_rerank:
            print("⚡ 使用快速模式：不執行 Reranker（依 similarity 排序）")
            ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
            return ranked[:final_top_k]

        # ---------------------------------------------------------
        #  使用 Reranker：Cross-Encoder scoring → Sort
        # ---------------------------------------------------------
        print("🧠 使用精準模式：啟用 Reranker 重新排序")

        reranked_per_query = rerank_results(
            query,
            [candidates],
            reranker=self.reranker,
            top_k=final_top_k
        )

        return reranked_per_query[0] if reranked_per_query else []