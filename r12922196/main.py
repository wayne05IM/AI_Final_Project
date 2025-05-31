from __future__ import annotations
import os, sys, re, ast, csv
from typing import Dict, List, get_type_hints
import random
from openai import OpenAI
from utils.QueryLoader import QueryLoader

model_type = "gpt-4o-mini"
# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}


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

    results = []
    for item in queryloader.get_queries():
        query_id = item['id']
        query_text = item['query']
        selected_title = llm_select_category(query_text, law_definitions)
        print(f"ID: {query_id}, Selected Law Title: {selected_title}")
        results.append({"ID": query_id, "SelectedTitle": selected_title})

    # 儲存結果為 CSV（只包含ID和SelectedTitle）
    output_csv = "query_selected_titles.csv"
    with open(output_csv, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "SelectedTitle"])
        writer.writeheader()
        writer.writerows(results)

    print(f"結果已儲存至 {output_csv}")

