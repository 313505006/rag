# retriever/file_embedding.py
"""
Files Embedding 模組：
將摘要後的文件轉成向量表示，供後續向量搜尋使用。
這裡使用 HuggingFace 的 sentence-transformers / embedding 模型。
"""

from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# === 這裡換成你的 Qwen3-Embedding 模型名稱 ===
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # TODO: 改成實際可用的名稱

class FileEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        將多個文字編碼成 numpy 向量 (num_texts, dim)
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**enc)
            # 這裡示意使用 CLS pooling，你可依模型官方建議修改
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0]  # CLS token

            emb = emb.cpu().numpy()
            all_embeddings.append(emb)

        return np.vstack(all_embeddings)

def embed_files(abstracts: List[Dict], embedder: FileEmbedder | None = None
                ) -> Tuple[np.ndarray, List[Dict]]:
    """
    將摘要文件轉成向量。

    輸入：
        abstracts: list[{"id": str, "abstract": str, ...}]
    輸出：
        embeddings: np.ndarray (N, dim)
        metadatas:  list[dict]（每筆對應的 meta，包含 id / 原始路徑 / 摘要等）
    """
    if embedder is None:
        embedder = FileEmbedder()

    texts = [f["abstract"] for f in abstracts]
    embeddings = embedder.encode(texts)
    metadatas = abstracts  # 這裡直接沿用，裡面已包含 id/abstract/path 等資訊
    return embeddings, metadatas
