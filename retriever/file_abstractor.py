# retriever/file_abstractor.py
"""
Files Abstracting 模組：
將冗長教材壓縮成摘要，減少向量長度，同時保留重點內容。

這裡先給一個「簡化版實作」：
- 如果長度 > max_chars，取前 max_chars 字 + "..."
- 你可以之後改成呼叫 LLM 進行真正的摘要。
"""

from typing import List, Dict

def simple_truncate(text: str, max_chars: int = 2000) -> str:
    """簡單截斷文字（示意用，可之後改成 LLM 摘要）"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n......(內容略)"

def abstract_files(files: List[Dict], max_chars: int = 2000) -> List[Dict]:
    """
    輸入：list[{"id": str, "text": str, ...}]
    輸出：list[{"id": str, "abstract": str, ...}]
    """
    results = []
    for f in files:
        abstract = simple_truncate(f["text"], max_chars=max_chars)
        item = dict(f)
        item["abstract"] = abstract
        results.append(item)
    return results

