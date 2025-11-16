# run_retriever_example.py
"""
查詢範例：
先執行 preprocess_data.py 產生：
- chunks / 摘要
- embeddings
- FAISS 向量資料庫

然後用此程式做查詢。
"""
from config.settings import RERANK_TOPK
from retriever import RetrieverPipeline

VECTOR_DB_PATH = "data/vector_db/index.faiss"

def main():
    print("=== 載入 Retriever Pipeline ===")
    pipeline = RetrieverPipeline(vector_db_path=VECTOR_DB_PATH)

    # 查詢示例
    query = "我想要知道特殊教育法修正日期"

    print(f"\n=== 查詢：{query} ===")
    results = pipeline.retrieve(query, top_k=RERANK_TOPK,use_rerank=True)
    # results = pipeline.retrieve(query, top_k=RERANK_TOPK,use_rerank=False)

    if not results:
        print("找不到相關文件。")
        return

    print("\n=== 查詢結果 (Top K) ===\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Top {i} ---")
        print("✔ 文件 ID:", doc["id"])
        print("✔ 相似度分數:", doc.get("score"))
        print("✔ 重排序分數:", doc.get("rerank_score"))
        print("✔ 摘要/內容（前 200 字）:")
        print(doc["abstract"][:200], "...\n")

if __name__ == "__main__":
    main()
