# utils/preprocess.py

import re

def clean_text(text: str) -> str:
    """
    清洗教科書/教材常見雜訊
    """
    text = text.replace("\u3000", " ")  # 全形空白
    text = re.sub(r"\s+", " ", text)    # 多重空白
    text = text.strip()
    return text


def split_into_sentences(text: str) -> list:
    """
    將文件拆成句子列表，用於 chunking
    """
    text = clean_text(text)
    sentences = re.split(r"(?<=[。！？\?])", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
