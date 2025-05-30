from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from pathlib import Path
import os, json, glob
import time
import pandas as pd
from pathlib import Path
from openai._exceptions import RateLimitError

# 設定 OpenAI API Key
os.environ["OPENAI_API_KEY"] = "[API-KEY]"
input_df = pd.read_csv("final_project_query.csv")

# 放 json 的資料夾
base_dir = "./laws_and_examples/"
# 所有 JSON 法規檔案路徑
json_files = [
    "13項保健功效及不適當功效延申例句之參考.json",
    "中醫藥司之中藥成藥效能、適應症語意解析及中藥廣告違規態樣釋例彙編.json",
    "中藥成藥不適當共通性廣告詞句.json",
    "化妝品涉及影響生理機能或改變身體結構之詞句.json",
    "化粧品衛生管理法.json",
    "食品、化粧品、藥物、醫療器材相關法規彙編.json",
    "食品安全衛生管理法.json",
    "食品藥品健康食品得使用詞句補充案例.json",
    "食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-可用詞句.json",
    "衛生福利部暨臺北市政府衛生局食品廣告例句209.json",
    "衛生福利部暨臺北市政府衛生局健康食品廣告例句83.json"
]

# 處理法規檔案並轉為 Document 格式
all_docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for json_file in json_files:
    with open(Path(base_dir) / json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 遞迴處理檔案
        def flatten_json(prefix, obj):
            texts = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    texts.extend(flatten_json(f"{prefix}-{k}" if prefix else k, v))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    texts.extend(flatten_json(f"{prefix}[{i}]", item))
            elif isinstance(obj, str):
                texts.append((prefix, obj))
            return texts

        entries = flatten_json("", data)
        for idx, content in entries:
            doc = Document(page_content=content, metadata={"source": os.path.basename(json_file), "id": idx})
            all_docs.append(doc)

split_docs = text_splitter.split_documents(all_docs)

# embedding
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# RAG Chain
llm = ChatOpenAI(model_name="gpt-4")

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
你是一位熟悉台灣《藥事法》、《食品安全衛生管理法》、《化粧品衛生管理法》等法規的法規審查員，請根據下列「相關法規內容」判斷問題中的廣告詞是否合法。

如果有可能違法，請只回覆 1。如果不違法，請只回覆 0。


=== 相關法規內容 ===
{context}

=== 廣告詞 ===
{question}
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

results = []

output_path = "task2_output.csv"

# 檢查是否已有輸出檔，沒有就寫入標題
if not Path(output_path).exists():
    pd.DataFrame(columns=["ID", "Answer"]).to_csv(output_path, index=False, encoding="utf-8-sig")


# 避免跑到一半跳掉，每次處理一個 query 就儲存到檔案
for idx, row in input_df.iterrows():
    ad_id = row["ID"]
    query = row["Question"]

    # 跳過已經寫入的
    output_df = pd.read_csv(output_path)
    if ad_id in output_df["ID"].values:
        print(f"跳過已完成 ID: {ad_id}")
        continue

    print(f"正在處理 ID: {ad_id}")

    # 加入重試機制
    while True:
        try:
            response = qa_chain({"query": query})
            result_text = response["result"]
            print(f"Response: {result_text}")
            # 寫入一筆結果
            pd.DataFrame([{
                "ID": ad_id,
                "Answer": result_text
            }]).to_csv(output_path, mode='a', header=False, index=False, encoding="utf-8-sig")

            print(f"已完成 ID: {ad_id}")
            break

        except RateLimitError as e:
            print(f"RateLimitError，等待 10 秒後重試：{e}")
            time.sleep(10)
        except Exception as e:
            print(f"其他錯誤（ID: {ad_id}）：{e}")
            break
    
