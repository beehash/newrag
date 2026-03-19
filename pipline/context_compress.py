import re
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from config import EMBEDDING_CONFIG, RERANK_CONFIG

from pipline.cos_similarity import CosineSimilarityCalculator
from pipline.rerank import Reranker

class ContextCompressor:
    """
    上下文压缩类，将文档切分为句子并保留与查询相关的句子
    """
    
    def __init__(self, model_name=EMBEDDING_CONFIG["model_name"], reranker_model=RERANK_CONFIG["model_name"]):
        """
        初始化上下文压缩器
        
        Args:
            model_name: 用于生成嵌入的模型名称
            reranker_model: 用于重排序的模型名称
        """
        self.similarity_calculator = CosineSimilarityCalculator(model_name)
        self.reranker = Reranker(model_name=reranker_model)
    
    def split_into_sentences(self, text):
        """
        将文本切分为句子列表
        
        Args:
            text: 文本
            
        Returns:
            list: 句子列表
        """
        # 使用正则表达式切分句子
        # 匹配句号、问号、感叹号等结束符
        sentences = re.split(r'[。！？.!?]', text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def compress_with_sentence_filter(self, query, document, top_k=3):
        """
        使用句子过滤方式压缩文档，保留与查询相关的句子
        
        Args:
            query: 查询文本
            document: 文档，包含title和content
            top_k: 保留的句子数量
            
        Returns:
            dict: 压缩后的文档
        """
        # 切分句子
        sentences = self.split_into_sentences(document.get("content", ""))
        
        if not sentences:
            return document
        
        # 构建句子文档列表
        sentence_docs = []
        for i, sentence in enumerate(sentences):
            sentence_docs.append({
                "title": "",
                "content": sentence,
                "index": i
            })
        
        # 使用bge-reranker-large模型对句子进行排序
        reranked_sentences = self.reranker.rerank(query, sentence_docs, top_k=top_k + 1)
        
        # 提取排名靠前的句子索引
        top_indices = [s.get("index", -1) for s in reranked_sentences]
        
        # 添加相邻句子（+-1）
        all_indices = set()
        for idx in top_indices:
            all_indices.add(idx)
            if idx > 0:
                all_indices.add(idx - 1)
            if idx < len(sentences) - 1:
                all_indices.add(idx + 1)
        
        # 按原始顺序排序
        sorted_indices = sorted(list(all_indices))
        
        # 提取句子并重新组合
        selected_sentences = [sentences[idx] for idx in sorted_indices]
        compressed_content = "。".join(selected_sentences) + "。"
        
        # 返回压缩后的文档
        compressed_document = document.copy()
        compressed_document["content"] = compressed_content
        compressed_document["original_content_length"] = len(document.get("content", ""))
        compressed_document["compressed_content_length"] = len(compressed_content)
        
        return compressed_document
    
    def compress_with_cos_similarity(self, query, document, top_k=3):
        """
        使用余弦相似度方式压缩文档，为每个chunk重新计算相似度
        
        Args:
            query: 查询文本
            document: 文档，包含title和content
            top_k: 保留的句子数量
            
        Returns:
            dict: 压缩后的文档
        """
        # 获取原始文档内容
        original_content = document.get("content", "")
        
        if not original_content:
            return document
        
        # 直接计算整个chunk与查询的余弦相似度
        chunk_similarity = self.similarity_calculator.calculate(query, original_content)
        
        # 返回压缩后的文档（这里不进行句子级别的压缩，只更新相似度）
        compressed_document = document.copy()
        compressed_document["similarity"] = chunk_similarity
        
        return compressed_document
    
    def compress(self, query, document, top_k=3, method="cos_similarity"):
        """
        压缩文档，保留与查询相关的句子
        
        Args:
            query: 查询文本
            document: 文档，包含title和content
            top_k: 保留的句子数量
            method: 压缩方法，可选值："sentence_filter" 或 "cos_similarity"
            
        Returns:
            dict: 压缩后的文档
        """
        if method == "sentence_filter":
            return self.compress_with_sentence_filter(query, document, top_k)
        else:
            return self.compress_with_cos_similarity(query, document, top_k)
