from __future__ import annotations
import os, sys, re, ast, csv
from typing import Dict, List, get_type_hints
from openai import OpenAI
from utils.QueryLoader import QueryLoader
from utils.Retriever import SplitFAISSRetriever
from utils.LawLoader import StructuredLawLoader
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_rag_prompt(query: str, category: str, violation_cases: List[str], appropriate_cases: List[str]) -> str:
    
    appropriate_text = "\n\t\t-".join(appropriate_cases) if appropriate_cases else "無"
    violation_text = "\n\t\t-".join(violation_cases) if violation_cases else "無"
    return f"""
    你是一位台灣法規專家，請根據以下廣告文字、分類、檢索到的違規案例與適當案例，以及法規條款，判斷該廣告和案例判斷廣告用詞是否涉及誇大療效及違法。請回應時輸出「違法機率(%)」說明理由

    ### 合規性判斷
    - **無罪判定原則**：不捏造或過度解讀廣告文字，**從寬認定合法性**，但如果是**藥物**宣稱**科學實證**、**專家推薦****國外研究**一類用語，則提高違法可能性認定
    認證**。
        - **比對允許使用的廣告用詞或適當案例**
            - 完整補充營養」「調整體質」「促進新陳代謝」「幫助入睡」「保護消化道全機能」「改變細菌叢生態」
            「排便有感」「在嚴謹的營養均衡與熱量控制，以及適當的運動條件下，適量攝取本產品有助於不易形成體脂
            肪」這些文字出現時不視為有違法風險
            - 「能完整補充人體營養」、「調整體質」、「提升生理機能」、「調節生理機能」、「促進新陳代謝」、「幫
            助入睡」、「調整體質」、「青春美麗」、「排便超有感」、「給你排便順暢新體驗」、「維持正常的排便習
            慣」、「排便順暢」、「促進新陳代謝」、「調整體質」、「改變細菌叢生態」、調節生理機能」、「保護消化
            道全機能」、「提升吸收滋養消化機能」"這些文字出現時不視為有違法風險
            - {appropriate_text}
    - **比對不允許使用的廣告用語或違規案例：**
        - ["一步登天"、"立刻見效"、"永久根治"]
        - {violation_text}

    ## 分析步驟
    1. 比對檢索到的違規案例與適當案例，找出與廣告文字相似或雷同之處。
    2. 對廣告內容進行分析，檢視是否存在療效誇大、誤導性宣稱，或包含未經核可之成分。
        - 僅以廣告文字本身判斷，不引申上下文、隱含意義或誇大解讀。只要文字中出現與違規案例相符或類似的語句，即屬明顯違規，否則根據字面含義從寬認定。
    3. 綜合評估違法風險, 依據法律及案例進行風險評估。 給出違法機率(0~100%), 作為最終判斷依據。

    ### 回應格式範例
    違法機率: 0~100%
    
    廣告文字：
    {query}

    """



def llm_select_category(query: str, law_definitions: dict) -> str:
    # Construct the prompt with law titles and their content
    law_entries = "\n".join([f"- {title}: {content}" for title, content in law_definitions.items()])

    system_prompt = f"""
你是一位台灣法規專家。根據以下廣告文字和法規資料，請從下列文件中選擇最適用的文件，用以判斷廣告的合法性。請只回答最適合的文件名稱（一個），不要包含多個選擇或其他文字。
廣告文字：{query}

法規列表及內容：
{law_entries}


"""
    print("PROMPT",system_prompt)

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "請選擇文件名稱。"}
        ]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    queryloader = QueryLoader('./final_project_query.csv', './cleaned_legal_definitions.json')
    law_definitions = queryloader.get_law_definitions()
    loader = StructuredLawLoader('./laws')
    retriever = SplitFAISSRetriever()
    for cat in loader.category_structure.keys():
        retriever.build_index(cat, loader.get_violations(cat), "violation")
        retriever.build_index(cat, loader.get_appropriates(cat), "appropriate")

    results = []
    for item in queryloader.get_queries():
        query_id = item['id']
        query_text = item['query']
        
        # 🔸 先用 LLM 選擇分類
        category_title = llm_select_category(query_text, law_definitions)
        print(f"✅ Query {query_id}: 分類 {category_title}")

        # 🔸 根據分類檢索案例
        violation_cases = [f"ID:{c['id']}, {c['text']}" for c in retriever.search(category_title, query_text, "violation", top_k=3)]
        appropriate_cases = [f"ID:{c['id']}, {c['text']}" for c in retriever.search(category_title, query_text, "appropriate", top_k=3)]

        # 🔸 生成 prompt，呼叫 LLM 判斷違法機率
        prompt = generate_rag_prompt(query_text, category_title, violation_cases, appropriate_cases)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一位台灣法規專家。"},
                {"role": "user", "content": prompt}
            ]
        )
        result_text = response.choices[0].message.content.strip()
        match = re.search(r"違法機率[:：]\s*(\d+)", result_text)
        if match:
            risk = int(match.group(1))
            result_binary = 0 if risk >= 75 else 1
        else:
            print("⚠️ 未找到違法機率，預設合法(1)")
            result_binary = 1
        results.append({"ID": query_id, "Answer": result_binary})
        print(f"✅ Query {query_id}: 結果 {result_binary} \n{result_text}")

    with open("final_result.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Answer"])
        writer.writeheader()
        writer.writerows(results)

