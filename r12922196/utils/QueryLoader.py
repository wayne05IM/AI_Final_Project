import json
import csv
from typing import List, Dict

class QueryLoader:
    def __init__(self, query_file: str, law_definition_file: str):
        self.query_file = query_file
        self.law_definition_file = law_definition_file
        self.queries: List[Dict[str, str]] = []
        self.law_definition: Dict[str, str] = {}

        self.load_queries()
        self.load_law_definitions()

    def load_queries(self):
        if self.query_file.endswith(".json"):
            with open(self.query_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.queries = data
                elif isinstance(data, dict):
                    self.queries = [{"id": k, "query": v} for k, v in data.items()]
        elif self.query_file.endswith(".csv"):
            with open(self.query_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 注意欄位名稱：ID 和 Question
                    self.queries.append({
                        "id": row["ID"].strip(),
                        "query": row["Question"].strip()
                    })
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")

    def load_law_definitions(self):
        with open(self.law_definition_file, 'r', encoding='utf-8') as f:
            self.law_definition = json.load(f)

    def get_queries(self) -> List[Dict[str, str]]:
        return self.queries

    def get_law_definitions(self) -> Dict[str, str]:
        return self.law_definition

    def summary(self) -> Dict[str, int]:
        return {
            "total_queries": len(self.queries),
            "total_law_sections": len(self.law_definition)
        }

if __name__ == "__main__":
    loader = QueryLoader('./final_project_query.csv', './cleaned_legal_definitions.json')

    print(loader.summary())
    for q in loader.get_queries()[92:93]:  # 顯示前3筆
        print(f"ID: {q['id']}, Query: {q['query'][:50]}...")
