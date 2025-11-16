# retriever/vector_store.py
"""
Vector Store 模組：
使用 FAISS 作為向量資料庫，負責：
- 新增文件向量
- 儲存 / 載入 index
- 依 query 向量做 kNN 搜尋
"""

from typing import List, Dict, Tuple, Optional
import os
import json

import numpy as np
import faiss

class VectorStore:
    def __init__(self, index_path: str):
        """
        index_path: 存放向量索引的檔案路徑（例如 data/vector_db/index.faiss）
        會另外在同資料夾存一份 meta.json 紀錄文件 metadata。
        """
        self.index_path = index_path
        self.meta_path = index_path + ".meta.json"
        self.index: Optional[faiss.Index] = None
        self.metadatas: List[Dict] = []

        self._load_if_exists()

    # ---------- Index 讀寫 ----------

    def _load_if_exists(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    # ---------- 新增向量 ----------

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]):
        """
        embeddings: (N, dim) float32
        metadatas:  list[dict] 長度 N
        """
        embeddings = embeddings.astype("float32")
        n, dim = embeddings.shape

        if self.index is None:
            # 使用內積相似度，可搭配向量先做 L2 normalize
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.metadatas.extend(metadatas)
        self._save()

    # ---------- 搜尋 ----------

    def search(self, query_vecs: np.ndarray, top_k: int = 10
               ) -> List[List[Tuple[float, Dict]]]:
        """
        依多個 query 向量做搜尋。

        回傳：List (num_queries)，
              每個元素是 list[(score, metadata)] (長度 top_k)
        """
        if self.index is None or len(self.metadatas) == 0:
            return [[] for _ in range(len(query_vecs))]

        query_vecs = query_vecs.astype("float32")
        scores, indices = self.index.search(query_vecs, top_k)

        all_results: List[List[Tuple[float, Dict]]] = []
        for q_idx in range(len(query_vecs)):
            q_scores = scores[q_idx]
            q_indices = indices[q_idx]
            results = []
            for s, idx in zip(q_scores, q_indices):
                if idx < 0 or idx >= len(self.metadatas):
                    continue
                meta = self.metadatas[idx]
                results.append((float(s), meta))
            all_results.append(results)
        return all_results
