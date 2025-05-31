import os
import json
import faiss
import numpy as np
from typing import List, Dict
from openai import OpenAI



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise ValueError("❗ 請設置 OPENAI_API_KEY 環境變數")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

class SplitFAISSRetriever:
    def __init__(self):
        self.violation_indexes = {}     # 每類別的違規索引
        self.violation_texts = {}
        self.violation_ids = {}
        self.appropriate_indexes = {}   # 每類別的適當索引
        self.appropriate_texts = {}
        self.appropriate_ids = {}

    def build_index(self, category: str, cases: List[Dict], case_type: str):
        if not cases:
            print(f"⚠️ {category} 類別沒有 {case_type} 案例")
            return
        embeddings, texts, ids = [], [], []
        for case in cases:
            text = case.get("content") or case.get("prohibited_terms")
            if isinstance(text, list):
                text = " ".join(text)
            embedding = get_embedding(text)
            embeddings.append(embedding)
            texts.append(text)
            ids.append(f"{category}_{case.get('id')}")
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        if case_type == "violation":
            self.violation_indexes[category] = index
            self.violation_texts[category] = texts
            self.violation_ids[category] = ids
        else:
            self.appropriate_indexes[category] = index
            self.appropriate_texts[category] = texts
            self.appropriate_ids[category] = ids
        print(f"✅ 已建立 {category} {case_type} 索引，共 {len(ids)} 條")

    def search(self, category: str, query: str, case_type: str, top_k=20) -> List[Dict]:
        index = self.violation_indexes if case_type == "violation" else self.appropriate_indexes
        texts = self.violation_texts if case_type == "violation" else self.appropriate_texts
        ids = self.violation_ids if case_type == "violation" else self.appropriate_ids
        if category not in index:
            print(f"❌ 查無 {category} {case_type} 索引")
            return []

        max_k = len(ids[category])  # 案例總數
        top_k = min(top_k, max_k)  # 調整 top_k
        query_embedding = np.array([get_embedding(query)]).astype("float32")
        D, I = index[category].search(query_embedding, top_k)
        return [{"id": ids[category][i], "text": texts[category][i]} for i in I[0]]

if __name__ == "__main__":
    from LawLoader import StructuredLawLoader  # 請替換成實際的載入模組
    # 載入分類資料
    category_structure = {
        "化粧品": {
            "law": ["化粧品衛生管理法.json", "食品、化粧品、藥物、醫療器材相關法規彙編.json"],
            "violation": ["化妝品涉及影響生理機能或改變身體結構之詞句.json"]
        },
        "藥物": {
            "law": ["食品、化粧品、藥物、醫療器材相關法規彙編.json"],
            "violation": [
                "13項保健功效及不適當功效延申例句之參考.json",
                "中藥成藥不適當共通性廣告詞句.json",
                "中醫藥司之中藥成藥效能、適應症語意解析及中藥廣告違規態樣釋例彙編.json",
                "衛生福利部暨臺北市政府衛生局健康食品廣告例句83.json",
                "衛生福利部暨臺北市政府衛生局食品廣告例句209.json"
            ],
            "approprite": [
                "食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-可用詞句.json",
                "食品藥品健康食品得使用詞句補充案例.json"
            ]
        },
        "食品": {
            "law": [
                "食品、化粧品、藥物、醫療器材相關法規彙編.json",
                "食品安全衛生管理法.json"
            ],
            "violation": [
                "13項保健功效及不適當功效延申例句之參考.json",
                "衛生福利部暨臺北市政府衛生局健康食品廣告例句83.json",
                "衛生福利部暨臺北市政府衛生局食品廣告例句209.json"
            ],
            "approprite": [
                "食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-可用詞句.json",
                "食品藥品健康食品得使用詞句補充案例.json"
            ]
        },
        "醫療器材": {
            "law": ["食品、化粧品、藥物、醫療器材相關法規彙編.json"]
        }
    }

    loader = StructuredLawLoader('./laws')
    retriever = SplitFAISSRetriever()

    for category in category_structure.keys():
        retriever.build_index(category, loader.get_violations(category), case_type="violation")
        retriever.build_index(category, loader.get_appropriates(category), case_type="appropriate")

    # 測試查詢
    query_text = "完全無副作用，效果極佳"
    category = "食品"
    print("\n🔎 違規案例查詢結果：")
    for res in retriever.search(category, query_text, "violation"):
        print(res)
        print(f"ID: {res['id']}, Text: {res['text']}")

    print("\n🔎 適當案例查詢結果：")
    for res in retriever.search(category, query_text, "appropriate"):
        print(f"ID: {res['id']}, Text: {res['text']}")
