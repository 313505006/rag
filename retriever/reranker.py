# retriever/reranker.py
"""
Files Reranking 模組：
使用 Cross-Encoder / Reranker (例如 Qwen3-Reranker) 重新排序候選文件，
讓與 query 最相關的文件排在前面。
"""

from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 若沒有 pad_token，自動補上（Qwen reranker 沒 pad_token）
        if self.tokenizer.pad_token is None:
            print("⚠️  Reranker tokenizer 沒有 pad_token，自動設定為 eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, docs: List[str]):
        """
        逐筆 scoring：避免 batch > 1 的 padding 問題。
        並且 Qwen3-Reranker 輸出 logits=[neg, pos]，要取 pos 作為最佳相關分數。
        """
        scores = []

        for doc in docs:
            enc = self.tokenizer(
                query,
                doc,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.model.device)

            out = self.model(**enc)

            # logits shape: [1, 2]
            #    logits[0][1] = positive relevance score
            logit = out.logits[0][1].item()  # <--- 修正重點在這一行！

            scores.append(float(logit))

        return scores


def rerank_results(query: str,
                   candidates_per_query: List[List[Dict]],
                   reranker: Reranker | None = None,
                   top_k: int = 10
                   ) -> List[List[Dict]]:
    """
    對 similarity_search 回傳的候選文件進行 rerank。

    輸入：
        query: 原始 query（這裡假設只有一個 query，若未來支援多個可再改）
        candidates_per_query: List[ List[{"score": float, "abstract": str, ...}] ]
                - 目前 pipeline 會只用第一個 query 的候選，所以取 index 0。
    輸出：
        reranked: List[ List[dict] ]
    """
    if reranker is None:
        reranker = Reranker()

    if not candidates_per_query:
        return []

    candidates = candidates_per_query[0]
    if not candidates:
        return [[]]

    docs = [c.get("abstract") or c.get("text") for c in candidates]
    scores = reranker.score(query, docs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = s

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return [reranked[:top_k]]
