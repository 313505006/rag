# retriever/file_abstractor_llm.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # 自行換模型

class LLMAbstractor:
    def __init__(self, model_name: str = LLM_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    @torch.no_grad()
    def summarize(self, text: str) -> str:
        prompt = f"請將以下教材內容濃縮成重點摘要（越清楚越好）：\n{text}\n\n摘要："

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ---------------------------------------------------------
        # 🔥 從「摘要：」後面開始截取（去掉 prompt）
        # ---------------------------------------------------------
        if "摘要：" in decoded:
            summary = decoded.split("摘要：", 1)[1].strip()
        else:
            # 如果模型沒有照格式輸出 fallback 到全文
            summary = decoded.strip()

        return summary



def abstract_chunks(chunks: list) -> list:
    """
    對每個 chunk 做 LLM 摘要
    """
    abs_model = LLMAbstractor()
    results = []

    for c in tqdm(chunks, desc="摘要中"):
        summary = abs_model.summarize(c)
        results.append(summary)

    return results
