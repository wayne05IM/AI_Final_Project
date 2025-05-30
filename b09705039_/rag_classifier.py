import os
import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Data class for storing analysis results"""
    id: str
    question: str
    violation_probability: int
    classification: int
    llm_output: str
    processing_time: float
    tokens_used: int
    cost: float
    error: Optional[str] = None

class ImprovedRAGSystem:
    def __init__(self, 
                 documents_directory: str,
                 api_key: str,
                 model_name: str = "gpt-4o-mini",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 retriever_k: int = 5):
        """
        Initialize the improved RAG system
        
        Args:
            documents_directory: Path to directory containing JSON legal documents
            api_key: OpenAI API key
            model_name: LLM model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            retriever_k: Number of documents to retrieve
        """
        self.documents_directory = documents_directory
        self.api_key = api_key
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k
        
        # Set API key
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize components
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Legal keywords for enhanced filtering
        self.legal_keywords = {
            'allowed_terms': [
                '完整補充營養', '調整體質', '促進新陳代謝', '幫助入睡',
                '保護消化道全機能', '改變細菌叢生態', '排便有感',
                '能完整補充人體營養', '提升生理機能', '調節生理機能',
                '青春美麗', '排便超有感', '給你排便順暢新體驗',
                '維持正常的排便習慣', '排便順暢', '提升吸收滋養消化機能'
            ],
            'high_risk_terms': [
                '科學實證', '國外研究', '臨床證實', '醫學認證',
                '治療', '療效', '根治', '痊癒', '藥效'
            ]
        }

    def json_to_enhanced_text(self, data: dict, source_file: str) -> str:
        """
        Enhanced JSON to text conversion with better structure
        """
        lines = []
        
        # Add source information
        lines.append(f"來源文件: {Path(source_file).stem}")
        
        # Process different types of legal documents
        if '法規名稱' in data or '條文' in data:
            lines.append("文件類型: 法規條文")
        elif '案例' in data or '裁罰' in data:
            lines.append("文件類型: 裁罰案例")
        
        # Enhanced content extraction
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"  項目{i+1}:")
                        for sub_key, sub_value in item.items():
                            lines.append(f"    {sub_key}: {sub_value}")
                    else:
                        lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)

    def load_and_process_documents(self) -> List[Document]:
        """
        Load and process JSON documents with enhanced error handling
        """
        docs = []
        failed_files = []
        
        json_files = list(Path(self.documents_directory).glob("*.json"))
        logger.info(f"發現 {len(json_files)} 個 JSON 文件")
        
        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    content = self.json_to_enhanced_text(data, str(filepath))
                    
                    # Add enhanced metadata
                    metadata = {
                        "source": str(filepath),
                        "filename": filepath.name,
                        "doc_type": self._identify_document_type(data),
                        "content_length": len(content)
                    }
                    
                    docs.append(Document(page_content=content, metadata=metadata))
                    
            except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
                logger.warning(f"無法處理文件 {filepath}: {e}")
                failed_files.append(str(filepath))
        
        logger.info(f"成功載入 {len(docs)} 個文件，失敗 {len(failed_files)} 個文件")
        if failed_files:
            logger.warning(f"失敗文件: {failed_files}")
            
        return docs

    def _identify_document_type(self, data: dict) -> str:
        """Identify the type of legal document"""
        if any(key in data for key in ['法規名稱', '條文', '法條']):
            return 'regulation'
        elif any(key in data for key in ['案例', '裁罰', '違規']):
            return 'case'
        elif any(key in data for key in ['判決', '法院']):
            return 'judgment'
        else:
            return 'other'

    def create_enhanced_vectorstore(self, force_rebuild: bool = False):
        """
        Create enhanced vectorstore with hybrid retrieval
        """
        vectorstore_path = "enhanced_faiss_store"
        
        if not force_rebuild and Path(vectorstore_path).exists():
            logger.info("載入現有向量資料庫...")
            try:
                self.vectorstore = FAISS.load_local(
                    vectorstore_path, 
                    self.embedding, 
                    allow_dangerous_deserialization=True
                )
                logger.info(f"成功載入向量資料庫，包含 {self.vectorstore.index.ntotal} 個向量")
                return
            except Exception as e:
                logger.warning(f"載入向量資料庫失敗: {e}，將重新建立")
        
        # Load and process documents
        raw_docs = self.load_and_process_documents()
        if not raw_docs:
            raise ValueError("未找到任何有效的法規文件")
        
        # Enhanced text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
            keep_separator=True
        )
        
        docs = splitter.split_documents(raw_docs)
        logger.info(f"文件切分完成，共 {len(docs)} 個片段")
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(docs, self.embedding)
        self.vectorstore.save_local(vectorstore_path)
        logger.info(f"向量資料庫建立完成，已儲存至 {vectorstore_path}")

    def setup_hybrid_retriever(self):
        """
        Setup hybrid retriever combining vector similarity and BM25
        """
        if not self.vectorstore:
            raise ValueError("向量資料庫尚未建立")
        
        # Vector similarity retriever
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retriever_k}
        )
        
        # BM25 retriever for keyword matching
        all_docs = []
        for i in range(self.vectorstore.index.ntotal):
            doc_id = self.vectorstore.index_to_docstore_id[i]
            doc = self.vectorstore.docstore.search(doc_id)
            all_docs.append(doc)
        
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = self.retriever_k
        
        # Combine retrievers
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Favor vector similarity slightly
        )
        
        logger.info("混合檢索器設定完成")

    def create_enhanced_prompt(self) -> PromptTemplate:
        """
        Create enhanced prompt template with better instructions
        """
        return PromptTemplate(
            input_variables=["summaries", "question"],
            template="""
## 任務目標
您是專業的法規分析專家，需要分析廣告內容是否違反《食品安全衛生管理法》等相關法規。
請根據提供的法規文件和案例，對廣告內容進行客觀、準確的合規性評估。

## 分析原則
### 1. 從寬認定原則
- 採用「無罪推定」原則，不過度解讀廣告文字
- 除非有明確違法證據，否則傾向認定為合法
- 特別注意：藥物宣稱「科學實證」、「國外研究」等用語需提高警覺

### 2. 合法用詞參考
以下用詞視為合法範圍：
- 營養補充類：「完整補充營養」、「能完整補充人體營養」
- 生理調節類：「調整體質」、「提升生理機能」、「調節生理機能」、「促進新陳代謝」
- 功能描述類：「幫助入睡」、「保護消化道全機能」、「提升吸收滋養消化機能」
- 美容保健類：「青春美麗」、「改變細菌叢生態」
- 排便相關：「排便有感」、「排便超有感」、「排便順暢」、「維持正常的排便習慣」

### 3. 高風險用詞識別
以下用詞需特別注意違法風險：
- 療效宣稱：「治療」、「療效」、「根治」、「痊癒」
- 醫學宣稱：「臨床證實」、「醫學認證」、「科學實證」、「國外研究」
- 誇大效果：涉及疾病治療、醫療效果的描述

## 分析步驟
1. **內容解析**：仔細閱讀廣告內容，識別關鍵用詞
2. **法規比對**：與提供的法規文件進行比對
3. **案例參考**：參考相似的裁罰案例
4. **風險評估**：綜合判斷違法機率（0-100%）

## 回應格式要求
### 低風險情況（違法機率 ≤ 80%）：
```
違法機率: X%
違法內容分析: 經分析廣告內容，違法風險較低。[簡述原因]
罰款額度: 無。
參考依據: [引用的法規或案例文件]
```

### 高風險情況（違法機率 > 80%）：
```
違法機率: X%
違法內容分析: 「[具體違法文字]」涉及[違法類型]，違反相關法規。
違反條款: 食品安全衛生管理法第XX條
罰款額度: 新臺幣4萬元以上400萬元以下罰鍰。
參考依據: [引用的法規或案例文件]
```

## 待分析廣告內容
{question}

## 參考法規文件
{summaries}

請基於以上資訊進行專業分析，確保回應格式正確且使用繁體中文。
"""
        )

    def setup_rag_chain(self):
        """
        Setup the RAG chain with enhanced components
        """
        if not self.retriever:
            raise ValueError("檢索器尚未設定")
        
        # Create enhanced prompt
        custom_prompt = self.create_enhanced_prompt()
        
        # Setup QA chain
        qa_chain = load_qa_with_sources_chain(
            self.llm, 
            chain_type="stuff", 
            prompt=custom_prompt
        )
        
        # Setup RAG chain
        self.rag_chain = RetrievalQA(
            retriever=self.retriever,
            combine_documents_chain=qa_chain,
            return_source_documents=True
        )
        
        logger.info("RAG 鏈設定完成")

    def enhanced_parse_result(self, llm_output: str) -> Tuple[int, int]:
        """
        Enhanced parsing of LLM output with better error handling
        """
        try:
            # Extract violation probability
            prob_match = re.search(r"違法機率[:：]\s*(\d+)", llm_output)
            if prob_match:
                probability = int(prob_match.group(1))
                classification = 1 if probability <= 80 else 0
                return probability, classification
            else:
                # Fallback: look for keywords indicating high/low risk
                high_risk_indicators = ['違法', '違反', '裁罰', '罰鍰', '禁止']
                low_risk_indicators = ['合法', '允許', '無風險', '符合規定']
                
                content_lower = llm_output.lower()
                high_risk_count = sum(1 for indicator in high_risk_indicators if indicator in content_lower)
                low_risk_count = sum(1 for indicator in low_risk_indicators if indicator in content_lower)
                
                if high_risk_count > low_risk_count:
                    return 85, 0  # High risk
                else:
                    return 30, 1  # Low risk
                    
        except Exception as e:
            logger.warning(f"解析結果時發生錯誤: {e}")
            return -1, -1

    def analyze_single_question(self, question_id: str, question: str) -> AnalysisResult:
        """
        Analyze a single question with comprehensive error handling
        """
        start_time = time.time()
        
        try:
            with get_openai_callback() as cb:
                response = self.rag_chain.invoke({"query": question})
                llm_answer = response["result"]
                
                # Parse results
                probability, classification = self.enhanced_parse_result(llm_output=llm_answer)
                
                processing_time = time.time() - start_time
                
                return AnalysisResult(
                    id=question_id,
                    question=question,
                    violation_probability=probability,
                    classification=classification,
                    llm_output=llm_answer,
                    processing_time=processing_time,
                    tokens_used=cb.total_tokens,
                    cost=cb.total_cost
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"處理問題 {question_id} 時發生錯誤: {str(e)}"
            logger.error(error_msg)
            
            return AnalysisResult(
                id=question_id,
                question=question,
                violation_probability=-1,
                classification=-1,
                llm_output="處理錯誤",
                processing_time=processing_time,
                tokens_used=0,
                cost=0.0,
                error=error_msg
            )

    def batch_analyze(self, 
                     csv_file_path: str, 
                     max_workers: int = 3,
                     output_dir: str = ".") -> List[AnalysisResult]:
        """
        Batch analyze questions with parallel processing and comprehensive reporting
        """
        # Load questions
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"載入 {len(df)} 個問題進行分析")
        except Exception as e:
            logger.error(f"載入CSV文件失敗: {e}")
            return []
        
        results = []
        total_cost = 0.0
        total_tokens = 0
        
        # Process with thread pool for I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(self.analyze_single_question, str(row['ID']), row['Question']): row 
                for _, row in df.iterrows()
            }
            
            # Collect results
            for future in as_completed(future_to_question):
                result = future.result()
                results.append(result)
                
                if result.error is None:
                    total_cost += result.cost
                    total_tokens += result.tokens_used
                    logger.info(f"✅ {result.id} 完成 - 違法機率: {result.violation_probability}% - 分類: {result.classification}")
                else:
                    logger.error(f"❌ {result.id} 失敗 - {result.error}")
        
        # Sort results by ID
        results.sort(key=lambda x: int(x.id) if x.id.isdigit() else 0)
        
        # Generate comprehensive reports
        self.generate_analysis_report(results, output_dir)
        
        # Print summary
        successful_results = [r for r in results if r.error is None]
        logger.info(f"""
=== 批次分析完成摘要 ===
總問題數: {len(results)}
成功分析: {len(successful_results)}
失敗數量: {len(results) - len(successful_results)}
總花費: ${total_cost:.4f}
總token使用: {total_tokens}
平均處理時間: {sum(r.processing_time for r in results) / len(results):.2f}秒
        """)
        
        return results

    def save_results(self, results: List[AnalysisResult], output_file: str):
        """
        Save analysis results to CSV
        """
        df_results = pd.DataFrame([
            {
                "ID": result.id,
                "Question": result.question,
                "Violation_Probability": result.violation_probability,
                "Classification": result.classification,
                "LLM_Output": result.llm_output,
                "Processing_Time": result.processing_time,
                "Tokens_Used": result.tokens_used,
                "Cost": result.cost,
                "Error": result.error
            }
            for result in results
        ])
        
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"結果已儲存至 {output_file}")

    def save_classification_only(self, results: List[AnalysisResult], output_file: str):
        """
        Save only classification results in the required format
        """
        # 建立 DataFrame，只保留分類結果（Answer）
        output_df = pd.DataFrame([
            {"Classification": result.classification}
            for result in results
        ])
        
        # 重新命名欄位
        output_df.rename(columns={"Classification": "Answer"}, inplace=True)
        
        # 加上連續的 ID，從 0 開始
        output_df.reset_index(inplace=True)
        output_df.rename(columns={"index": "ID"}, inplace=True)
        
        # 儲存成 CSV 檔案
        output_df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"分類結果已儲存至 {output_file}")
        
        # 顯示分類統計
        classification_counts = output_df['Answer'].value_counts()
        logger.info(f"分類統計: \n{classification_counts}")
        
        return output_df

    def generate_analysis_report(self, results: List[AnalysisResult], output_dir: str = "."):
        """
        Generate comprehensive analysis report with multiple output formats
        """
        import os
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. 完整詳細結果
        detailed_file = output_dir / "detailed_analysis_results.csv"
        self.save_results(results, str(detailed_file))
        
        # 2. 僅分類結果（符合提交格式）
        classification_file = output_dir / "classification_results.csv"
        classification_df = self.save_classification_only(results, str(classification_file))
        
        # 3. 統計摘要
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        summary_stats = {
            "總問題數": len(results),
            "成功分析數": len(successful_results),
            "失敗數": len(failed_results),
            "成功率": f"{len(successful_results)/len(results)*100:.1f}%" if results else "0%",
            "合法判定數 (Classification=1)": len([r for r in successful_results if r.classification == 1]),
            "違法判定數 (Classification=0)": len([r for r in successful_results if r.classification == 0]),
            "無法判定數 (Classification=-1)": len([r for r in results if r.classification == -1]),
            "總處理費用": f"${sum(r.cost for r in successful_results):.4f}",
            "總Token使用": sum(r.tokens_used for r in successful_results),
            "平均處理時間": f"{sum(r.processing_time for r in results)/len(results):.2f}秒" if results else "0秒"
        }
        
        # 儲存統計摘要
        summary_file = output_dir / "analysis_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)
        
        # 4. 錯誤報告（如果有的話）
        if failed_results:
            error_df = pd.DataFrame([
                {
                    "ID": result.id,
                    "Question": result.question[:100] + "..." if len(result.question) > 100 else result.question,
                    "Error": result.error
                }
                for result in failed_results
            ])
            error_file = output_dir / "error_report.csv"
            error_df.to_csv(error_file, index=False, encoding='utf-8-sig')
            logger.warning(f"錯誤報告已儲存至 {error_file}")
        
        # 5. 違法機率分布分析
        if successful_results:
            prob_analysis = []
            for result in successful_results:
                if result.violation_probability >= 0:
                    prob_analysis.append({
                        "ID": result.id,
                        "Violation_Probability": result.violation_probability,
                        "Classification": result.classification,
                        "Risk_Level": self._categorize_risk_level(result.violation_probability)
                    })
            
            if prob_analysis:
                prob_df = pd.DataFrame(prob_analysis)
                prob_file = output_dir / "probability_analysis.csv"
                prob_df.to_csv(prob_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"""
=== 分析報告生成完成 ===
輸出目錄: {output_dir}
- 詳細結果: {detailed_file.name}
- 分類結果: {classification_file.name}
- 統計摘要: {summary_file.name}
{'- 錯誤報告: error_report.csv' if failed_results else ''}
{'- 機率分析: probability_analysis.csv' if successful_results else ''}
        """)
        
        # 印出統計摘要
        print("\n=== 分析統計摘要 ===")
        for key, value in summary_stats.items():
            print(f"{key}: {value}")
        
        return classification_df

    def _categorize_risk_level(self, probability: int) -> str:
        """Categorize risk level based on violation probability"""
        if probability < 0:
            return "無法判定"
        elif probability <= 30:
            return "低風險"
        elif probability <= 60:
            return "中低風險"
        elif probability <= 80:
            return "中高風險"
        else:
            return "高風險"

    def initialize_system(self, force_rebuild_vectorstore: bool = False):
        """
        Initialize the complete RAG system
        """
        logger.info("開始初始化RAG系統...")
        
        # Step 1: Create vectorstore
        self.create_enhanced_vectorstore(force_rebuild=force_rebuild_vectorstore)
        
        # Step 2: Setup retriever
        self.setup_hybrid_retriever()
        
        # Step 3: Setup RAG chain
        self.setup_rag_chain()
        
        logger.info("RAG系統初始化完成！")

# Usage example
def main():
    # Configuration
    DOCUMENTS_DIR = "./法規及案例 Vector Stores/"
    API_KEY = "Your API Key Here"  # Replace with your actual OpenAI API key
    CSV_FILE = "./final_project_query.csv"
    OUTPUT_DIR = "./"
    
    # Initialize system
    rag_system = ImprovedRAGSystem(
        documents_directory=DOCUMENTS_DIR,
        api_key=API_KEY,
        model_name="gpt-4o-mini",
        chunk_size=800,
        chunk_overlap=100,
        retriever_k=5
    )
    
    # Initialize the system
    rag_system.initialize_system(force_rebuild_vectorstore=False)
    
    # Perform batch analysis with comprehensive reporting
    results = rag_system.batch_analyze(
        csv_file_path=CSV_FILE,
        max_workers=3,
        output_dir=OUTPUT_DIR
    )
    
    return results

# Alternative: Quick classification-only extraction
def extract_classification_results(results: List[AnalysisResult], output_path: str):
    """
    Quick function to extract only classification results in the required format
    This matches your exact requirements
    """
    # 建立 DataFrame，只保留分類結果（Answer）
    output_df = pd.DataFrame([{"Classification": result.classification} for result in results])
    output_df.rename(columns={"Classification": "Answer"}, inplace=True)
    
    # 加上連續的 ID，從 0 開始
    output_df.reset_index(inplace=True)
    output_df.rename(columns={"index": "ID"}, inplace=True)
    
    # 儲存成 CSV 檔案
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"分類結果已儲存至: {output_path}")
    
    # 顯示統計
    print(f"分類統計:")
    print(output_df['Answer'].value_counts())
    
    return output_df

if __name__ == "__main__":
    # Run full analysis
    results = main()
    
    # Extract classification results in your specific format
    classification_output_path = "./classification_results.csv"
    classification_df = extract_classification_results(results, classification_output_path)