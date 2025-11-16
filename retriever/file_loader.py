# retriever/file_loader.py
"""
載入原始教材 / 文件的工具。
目前示範從資料夾讀取 .txt 檔，你可依需求改成 PDF / DOCX 解析。
"""

import os
from typing import List, Dict

def load_text_files(folder: str) -> List[Dict]:
    """
    從資料夾讀取所有 .txt 檔案，回傳 list[{"id": str, "text": str, "path": str}]
    """
    files = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(folder, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        files.append({
            "id": fname,
            "text": text,
            "path": fpath
        })
    return files
