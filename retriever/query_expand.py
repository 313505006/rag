# retriever/query_expand.py
"""
Query Expand 模組：
對使用者輸入的 query 進行語義擴展，提升召回率。

這裡先給簡易版本：
- 回傳 [原始 query]，等你之後接 LLM 再擴展成多個 query。
"""

from typing import List

def expand_query(query: str) -> List[str]:
    """
    簡單版本：目前只回傳 [原始查詢]。
    未來可以：
      - 呼叫 LLM 產生同義問句
      - 拆解成多個子問題
    """
    query = query.strip()
    if not query:
        return []
    return [query]
