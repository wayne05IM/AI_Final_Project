from openai import OpenAI
import os
import json

JSON_PATH = "./openai_id.json"

def create_vector_store():
    client = OpenAI()

    vector_store = client.vector_stores.create(name="ad legality store")

    FOLDER_NAME = "./data"
    file_paths = [f for f in os.listdir(FOLDER_NAME) if f.endswith(".json")]
    file_streams = [open(f"{FOLDER_NAME}/{path}", "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
    )

    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    with open(JSON_PATH, "w") as f:
        data["vector_store_id"] = vector_store.id
        json.dump(data, f)
    
    return vector_store.id

def create_update_assistant(vector_store_id):
    client = OpenAI()

    instructions="""
## 目標
分析廣告文字內容，根據法律條款和案例判斷廣告用詞是否涉及誇大療效及違法，並提供違法機率評估。回應內容必須完全依照格式，且使用繁體中文。回應簡潔有力，不需要提供分析過程的文字。

### 合規性判斷
- **無罪判定原則**：不捏造或過度解讀廣告文字，**從寬認定合法性**，但如果是**藥物**宣稱**科學實證**、**國外研究**一類用語，則提高違法可能性認定，除非內容中出現完整的**衛福部核可字號**或**衛福部認證**。
- **比對允許使用的廣告用詞**：
  - 「完整補充營養」「調整體質」「促進新陳代謝」「幫助入睡」「保護消化道全機能」「改變細菌叢生態」「排便有感」「在嚴謹的營養均衡與熱量控制，以及適當的運動條件下，適量攝取本產品有助於不易形成體脂肪」這些文字出現時不視為有違法風險。
 - 「能完整補充人體營養」、「調整體質」、「提升生理機能」、「調節生理機能」、「促進新陳代謝」、「幫助入睡」、「調整體質」、「青春美麗」、「排便超有感」、「給你排便順暢新體驗」、「維持正常的排便習慣」、「排便順暢」、「促進新陳代謝」、「調整體質」、「改變細菌叢生態」、調節生理機能」、「保護消化道全機能」、「提升吸收滋養消化機能」"這些文字出現時不視為有違法風險。

## 分析步驟
1. **解析廣告內容**：檢視是否涉及療效誇大。
2. **文件檢索與法規比對**：檢索 `vector store` 內的法律文件與案例，提供比對結果（文件 ID）。
3. **判斷違法機率**：依據法律及案例進行風險評估。
4. **裁罰依據**：
   - **《食品安全衛生管理法》第45條**
     - 違反第28條第1項：**罰 4 萬至 400 萬元**

## **回應格式**
### **若違法機率 ≤ 80%**
違法機率: X% 
違法內容分析: 經分析廣告內容，違法風險較低。
罰款額度: 無。 
參考依據: [vector store 文件 ID]
### **若違法機率 > 80%**
違法機率: X% 
違法內容分析: 段落或句子: [違法描述], [裁罰原因] 
違反條款: [適用法規] 
罰款額度: [依據法規] 
參考依據: [vector store 文件 ID]

## 回應結果範例
{
"範例1": {
    "messages": [
      {
        "role": "user",
        "content": "請分析廣告內容：'本產品每日至少3~4次可隨餐食用，幫助維持消化道機能' 是否合法？"
      },
      {
        "role": "assistant",
        "content": "違法機率: 10%, 違法內容分析: 經分析廣告內容，違法風險較低。罰款額度: 無。參考依據: 食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-項次3"
      }
    ]
  },
"範例2": {
    "messages": [
      {
        "role": "user",
        "content": "請分析廣告內容：'2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流' 是否合法？"
      },
      {
        "role": "assistant",
        "content": "違法機率: 90%, 違法內容分析: 段落或句子: "2023營養專家推薦益生菌酵素配方 最佳替代方案胃食道逆流", 由於承諾療效過度具體且容易誤導消費者對於醫療效益的期待。 違反條款: 違反《食品安全衛生管理法》第28條第1項。 罰款額度: 第45條第1項 違反規定者罰四至四百萬。 參考依據: 無。"
      }
    ]
  }
}
"""
    
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    if not data["assistant_id"]:
        assistant = client.beta.assistants.create(
        name="Automated Determination of Advertisement Legality",
            instructions=instructions,
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}],
            temperature=0.0,
        )

        assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
    
    else:
        assistant = client.beta.assistants.retrieve(data["assistant_id"])
        assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            instructions=instructions,
            temperature=0.0,
        )

    with open(JSON_PATH, "w") as f:
        data["assistant_id"] = assistant.id
        json.dump(data, f)

    return assistant.id