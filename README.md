# 📘 LLM-final-project — Retrieval-Augmented Generation System

本專案實作一套完整的 **Retrieval-Augmented Generation (RAG)** 系統，
包含文件前處理、Embedding、向量資料庫、語意檢索、Reranking 與 LLM 回答生成。

系統支援：

* **PDF / TXT / Docx 載入**
* **Chunking**
* **LLM 摘要（可選）**
* **Embedding（Qwen3-Embedding）**
* **Reranking（Qwen3-Reranker）**
* **FAISS 向量資料庫**
* **FastAPI 查詢 API**

---

## 📂 專案架構

```
rag_system/
│
├── config/
│   ├── settings.py              # 全域設定（模型、路徑、chunk size 等）
│
├── data/
│   ├── raw_files/               # 原始教材（pdf, docx, txt）
│   ├── abstracts/               # 摘要後的文件
│   ├── embeddings/              # 向量（optional）
│   └── vector_db/               # FAISS 索引
│
├── retriever/
│   ├── file_loader.py           # 載入 PDF/TXT/Docx
│   ├── file_abstractor.py       # LLM 摘要
│   ├── file_embedding.py        # 文件 embedding
│   ├── query_expand.py          # 查詢擴展
│   ├── query_embedding.py       # Query embedding
│   ├── vector_store.py          # FAISS 存取
│   ├── similarity_search.py     # 相似度檢索
│   ├── reranker.py              # 使用 Qwen3-Reranker
│   └── pipeline.py              # Retriever Pipeline 主程式
│
├── generator/
│   ├── llm_generator.py         # 回答生成模組
│
├── api/
│   ├── main.py                  # FastAPI Server
│   └── schemas.py               # API schema
│
├── utils/
│   ├── logger.py                # Log 工具
│   ├── preprocess.py            # 文本清洗工具
│   └── tokenizer.py             # chunk 切分工具
│
├── tests/
│   └── test_retriever.py        # 單元測試
│
└── run.py                       # 本地測試入口
```

---

## ⚙️ 安裝環境

### 1. 安裝 PyTorch（支援 CUDA 12.8 與 RTX 5090）

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. 安裝其他依賴

```bash
pip install -r requirements.txt
```

---

## 🔧 模型需求

本系統預設使用：

| 模組                   | 模型名稱                               | 來源          |
| -------------------- | ---------------------------------- | ----------- |
| 文件 / Query Embedding | `Qwen/Qwen3-Embedding-0.6B` 或 `8B` | HuggingFace |
| 文件重排序 (Reranking)    | `Qwen/Qwen3-Reranker-4B`           | HuggingFace |
| 回答生成                 | 任意 LLM（如 Qwen2-7B, Qwen2.5-7B）     | HuggingFace |

> 💡 這些模型只會下載一次，之後都從本地 cache 載入，不會重複下載。

---

## 📚 資料前處理流程（兩種模式）

### **Mode A：有摘要（--mode summarize）**

```
1. 載入 PDF/TXT/docx
2. 文本清洗
3. Chunk 切分
4. 使用 LLM 摘要（可大量壓縮教材）
5. Qwen3-Embedding 嵌入
6. 儲存向量到 FAISS
```
```bash
python preprocess_data.py --mode summarize
```


### **Mode B：無摘要（--mode no_summarize）**

```
1. 載入 PDF/TXT/docx
2. 文本清洗
3. Chunk 切分
4. 直接使用原文 chunk
5. Qwen3-Embedding 嵌入
6. 儲存向量到 FAISS
```
```bash
python preprocess_data.py --mode no_summarize
```


---

## 🔍 執行檢索（Retriever 測試）

```bash
python run_retriever_example.py
```

流程包含：

```
1. 載入 raw_files/ 教材
2. 摘要（可選）
3. 文件 embedding
4. 存入 FAISS vector DB
5. 查詢 → Query Expansion
6. 相似度搜尋（FAISS）
7. Qwen3-Reranker 重排序
8. 回傳最終結果
```

