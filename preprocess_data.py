# preprocess_data.py
"""
完整資料前處理流程（兩種模式）：

Mode A (--mode summarize):
1. PDF / TXT 載入
2. 清洗
3. Chunking
4. LLM 摘要（可跳過）
5. embedding
6. 儲存到 vector DB

Mode B (--mode no_summarize):
1. PDF / TXT 載入
2. 清洗
3. Chunking
4. 不摘要（直接用 chunk 原文）
5. embedding
6. 儲存到 vector DB
"""

import os
import argparse
from retriever import RetrieverPipeline
from retriever.file_loader import load_text_files
from utils.pdf_reader import read_pdf
from utils.preprocess import clean_text
from utils.chunk import chunk_by_sentences
from retriever.file_abstractor_llm import abstract_chunks
from utils.storage import save_json, load_json
from config.settings import CHUNK_MAX_TOKENS


SUMMARY_PATH = "data/processed/abstracts.json"


# ==========================================================
# 載入所有文件
# ==========================================================
def load_all_documents(folder):
    docs = []

    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)

        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

        elif fname.endswith(".pdf"):
            text = read_pdf(fpath)

        else:
            continue

        docs.append({
            "id": fname,
            "text": text,
            "path": fpath
        })

    return docs


# ==========================================================
# 主流程
# ==========================================================
def process_and_index(folder="data/raw_files",
                      vector_db="data/vector_db/index.faiss",
                      mode="summarize"):
    """
    mode:
    - "summarize": 使用 LLM 摘要
    - "no_summarize": 不做摘要
    """

    print("==== 讀取文件中 ====")
    docs = load_all_documents(folder)
    print(f"共載入 {len(docs)} 份教材")

    # 建立 retriever pipeline
    pipeline = RetrieverPipeline(vector_db_path=vector_db)

    all_chunks = []
    metas = []

    print("==== 清洗 + Chunking ====")
    for doc in docs:
        clean = clean_text(doc["text"])
        chunks = chunk_by_sentences(clean, max_tokens=CHUNK_MAX_TOKENS)

        for idx, c in enumerate(chunks):
            all_chunks.append(c)
            metas.append({
                "id": doc["id"],
                "chunk_id": idx,
                "text": c
            })

    print(f"共切成 {len(all_chunks)} 個 chunks")


    # ==========================================================
    # 模式 A：Summarize
    # ==========================================================
    if mode == "summarize":
        print("==== 模式：LLM 摘要 ====")

        # 若已有摘要紀錄 → 跳過
        old_summary = load_json(SUMMARY_PATH)
        if old_summary is not None and len(old_summary) == len(all_chunks):
            print("==== 發現已存在摘要，跳過 LLM 摘要 ====")
            summaries = old_summary
        else:
            print("==== 執行 LLM 摘要（Files Abstracting）====")
            summaries = abstract_chunks(all_chunks)
            print(f"==== 摘要完成，寫入 {SUMMARY_PATH} ====")
            save_json(SUMMARY_PATH, summaries)

        # 建立 final_docs
        final_docs = []
        for meta, summary in zip(metas, summaries):
            final_docs.append({
                "id": f"{meta['id']}_chunk{meta['chunk_id']}",
                "text": meta["text"],        # 原文 chunk
                "abstract": summary          # 摘要
            })

    # ==========================================================
    # 模式 B：No Summarize
    # ==========================================================
    else:
        print("==== 模式：不做摘要（直接使用 chunk 原始內容）====")

        final_docs = []
        for meta in metas:
            final_docs.append({
                "id": f"{meta['id']}_chunk{meta['chunk_id']}",
                "text": meta["text"],
                "abstract": meta["text"],     # 用原文取代摘要
            })


    # ==========================================================
    # 寫入向量資料庫
    # ==========================================================
    print("==== 寫入 Vector DB（embedding + indexing） ====")
    pipeline.index_files(final_docs)
    print("完成！")


# ==========================================================
# CLI 入口
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["summarize", "no_summarize"],
                        default="summarize",
                        help="choose processing mode")
    args = parser.parse_args()

    process_and_index(mode=args.mode)
