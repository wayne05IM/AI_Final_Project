import os
import json
from typing import List, Dict, Any, Union
from utils.CaseExtractor import ViolationCaseExtractor, AppropriateCaseExtractor


category_structure = {
        "化粧品": {
            "law": ["化粧品衛生管理法.json", "食品、化粧品、藥物、醫療器材相關法規彙編.json"],
            "violation": ["13項保健功效及不適當功效延申例句之參考.json",
                          "化妝品涉及影響生理機能或改變身體結構之詞句.json"]
        },
        "藥物": {
            "law": ["食品、化粧品、藥物、醫療器材相關法規彙編.json"],
            "violation": [
                "13項保健功效及不適當功效延申例句之參考.json",
                # "中藥成藥不適當共通性廣告詞句.json",
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

class FileWithJSON:
    def __init__(self, filepath: str, filename: str, json_data: Dict[str, Any]):
        self.filepath = filepath
        self.filename = filename
        self.json_data = json_data
    
    def to_dict(self):
        return {
            "filepath": self.filepath,
            "filename": self.filename,
            "json_data": self.json_data
        }

class StructuredLawLoader:
    def __init__(self, directory: str, extension: str = ".json"):
        self.directory = directory
        self.extension = extension
        self.category_structure = category_structure
        self.laws: Dict[str, List[FileWithJSON]] = {cat: [] for cat in category_structure}
        self.violations: Dict[str, List[Dict[str, Union[str, List[str]]]]] = {cat: [] for cat in category_structure}
        self.appropriates: Dict[str, List[Dict[str, str]]] = {cat: [] for cat in category_structure}
        self.load_files()

    def load_files(self):
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(self.extension):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r', encoding='utf-8-sig') as f:
                            data = json.load(f)
                            doc = FileWithJSON(full_path, file, data)
                            self._assign_to_categories(doc, full_path)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

    def _assign_to_categories(self, doc: FileWithJSON, full_path: str):
        for category, paths in self.category_structure.items():
            # 法規
            if "law" in paths and doc.filename in paths["law"]:
                self.laws[category].append(doc)
            # 違規
            if "violation" in paths and doc.filename in (paths["violation"] if isinstance(paths["violation"], list) else [paths["violation"]]):
                extractor = ViolationCaseExtractor(full_path)
                self.violations[category].extend(extractor.get_cases())
            # 適當
            if "approprite" in paths and doc.filename in (paths["approprite"] if isinstance(paths["approprite"], list) else [paths["approprite"]]):
                extractor = AppropriateCaseExtractor(full_path)
                self.appropriates[category].extend(extractor.get_cases())

    def get_laws(self, category: str) -> List[FileWithJSON]:
        return self.laws.get(category, [])

    def get_violations(self, category: str) -> List[Dict[str, Union[str, List[str]]]]:
        return self.violations.get(category, [])

    def get_appropriates(self, category: str) -> List[Dict[str, str]]:
        return self.appropriates.get(category, [])

    def summary(self) -> Dict[str, Any]:
        return {
            cat: {
                "law_count": len(self.laws[cat]),
                "violation_case_count": len(self.violations[cat]),
                "appropriate_case_count": len(self.appropriates[cat])
            } for cat in self.category_structure
        }

if __name__ == "__main__":

    loader = StructuredLawLoader('./laws')

    print(loader.summary())

    print("\n化粧品相關法條:")
    for doc in loader.get_laws("化粧品"):
        print(doc.to_dict())

    print("\n化粧品違規案例:")
    for case in loader.get_violations("化粧品"):
        print(case) 

    print("\n食品適當案例:")
    for case in loader.get_appropriates("食品"):
        print(case)
