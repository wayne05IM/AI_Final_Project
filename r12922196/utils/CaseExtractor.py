import json
from typing import List, Dict, Union

class ViolationCaseExtractor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cases: List[Dict[str, Union[str, List[str]]]] = []
        self.extract_cases()

    def extract_cases(self):
        with open(self.filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            title = data.get("title", self.filepath)

            # 中藥成藥不適當共通性廣告詞句
            if "inappropriate_tcm_ads" in data:
                for cat in data["inappropriate_tcm_ads"].get("violation_categories", []):
                    cat_id = cat.get("id", "")
                    category = cat.get("category", "")
                    terms = cat.get("prohibited_terms", [])
                    terms_str = json.dumps(terms, ensure_ascii=False)
                    self.cases.append({
                        "id": cat_id,
                        "category": category,
                        "prohibited_terms": terms_str,
                        "source": title
                    })

            # 中醫藥司之中藥成藥效能、適應症語意解析及中藥廣告違規態樣釋例彙編
            if "categories" in data:
                for cat in data["categories"]:
                    category = cat.get("category", "")
                    for example in cat.get("examples", []):
                        self.cases.append({
                            "id": example.get("ID", ""),
                            "category": category,
                            "content": example.get("ad_content", ""),
                            "source": title
                        })

            # 化妝品涉及影響生理機能或改變身體結構之詞句
            if "prohibited_physiological_claims" in data:
                for cat in data["prohibited_physiological_claims"].get("categories", []):
                    cat_id = cat.get("id", "")
                    category = cat.get("category", "")
                    terms = cat.get("prohibited_terms", [])
                    terms_str = json.dumps(terms, ensure_ascii=False)
                    self.cases.append({
                        "id": cat_id,
                        "category": category,
                        "prohibited_terms": terms_str,
                        "source": title
                    })

            # 健康食品違規例句
            if "health_food_violations" in data:
                for case in data["health_food_violations"].get("cases", []):
                    self.cases.append({
                        "id": case.get("ID", ""),
                        "category": case.get("violation_type", ""),
                        "content": case.get("ad_content", ""),
                        "source": title
                    })

            # 食品廣告違規例句
            if "advertisement_violation_cases" in data:
                for case in data["advertisement_violation_cases"].get("cases", []):
                    self.cases.append({
                        "id": case.get("id", ""),
                        "category": case.get("violation_type", ""),
                        "content": case.get("ad_content", ""),
                        "source": title
                    })

    def get_cases(self) -> List[Dict[str, Union[str, List[str]]]]:
        return self.cases


class AppropriateCaseExtractor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cases: List[Dict[str, str]] = []
        self.extract_cases()

    def extract_cases(self):
        with open(self.filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            title = data.get("title", self.filepath)

            # Case 1: 食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則
            if "categories" in data:
                for cat in data["categories"]:
                    if "cases" in cat:
                        for case in cat["cases"]:
                            self.cases.append({
                                "id": case.get("ID", ""),
                                "category": cat.get("category", ""),
                                "content": case.get("ad_content", ""),
                                "source": title
                            })
                    if "subcategories" in cat:
                        for subcat in cat["subcategories"]:
                            subcategory = subcat.get("subcategory", "")
                            for case in subcat.get("cases", []):
                                self.cases.append({
                                    "id": case.get("ID", ""),
                                    "category": f"{cat.get('category', '')}/{subcategory}",
                                    "content": case.get("ad_content", ""),
                                    "source": title
                                })
            
            # Case 2: 食品藥品健康食品得使用詞句補充案例
            if "cases" in data:
                for case in data["cases"]:
                    self.cases.append({
                        "id": case.get("ID", ""),
                        "category": "補充案例",
                        "content": case.get("content", ""),
                        "source": title
                    })

    def get_cases(self) -> List[Dict[str, str]]:
        return self.cases

if __name__ == "__main__":
    # 測試檔案路徑
    files = [
        './laws/食品及相關產品標示宣傳廣告涉及不實誇張易生誤解或醫療效能認定準則-可用詞句.json',
        './laws/食品藥品健康食品得使用詞句補充案例.json'
    ]

    all_cases = []
    for file in files:
        extractor = AppropriateCaseExtractor(file)
        all_cases.extend(extractor.get_cases())

    # 顯示結果
    for case in all_cases:
        print(json.dumps(case, ensure_ascii=False, indent=2))

    # 可選：儲存成 JSON 檔
    # with open("extracted_appropriate_cases.json", "w", encoding="utf-8") as f:
    #     json.dump(all_cases, f, ensure_ascii=False, indent=2)

    files = [
        './laws/中藥成藥不適當共通性廣告詞句.json',
        './laws/中醫藥司之中藥成藥效能、適應症語意解析及中藥廣告違規態樣釋例彙編.json',
        './laws/化妝品涉及影響生理機能或改變身體結構之詞句.json',
        './laws/衛生福利部暨臺北市政府衛生局健康食品廣告例句83.json',
        './laws/衛生福利部暨臺北市政府衛生局食品廣告例句209.json'
    ]

    all_cases = []
    for file in files:
        extractor = ViolationCaseExtractor(file)
        all_cases.extend(extractor.get_cases())

    # 顯示結果
    for case in all_cases:
        print(json.dumps(case, ensure_ascii=False, indent=2))
