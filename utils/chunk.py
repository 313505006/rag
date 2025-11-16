# utils/chunk.py

from typing import List
from .preprocess import split_into_sentences

from config.settings import CHUNK_MAX_TOKENS

def chunk_by_sentences(text: str, max_tokens: int = CHUNK_MAX_TOKENS):

    """
    將長文件切成多段 chunk，每段 ~300 tokens（可自行調整）
    使用句子累積，避免切斷語意。
    """
    sentences = split_into_sentences(text)

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) > max_tokens:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_fixed(text: str, size: int = 512) -> List[str]:
    """
    固定字元 chunk（保留給不想用句子拆的場景）
    """
    text = text.strip()
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return chunks
