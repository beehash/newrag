"""
BM25 稀疏向量生成模块
使用 BM25 算法生成稀疏向量用于检索
"""

import jieba
import math
from collections import Counter
from typing import List, Dict, Tuple


class BM25SparseEmbedding:
    """
    BM25 稀疏向量生成器
    基于 BM25 算法将文本转换为稀疏向量表示
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 稀疏向量生成器
        
        Args:
            k1: BM25 参数，控制词频饱和度，默认 1.5
            b: BM25 参数，控制文档长度归一化，默认 0.75
        """
        self.k1 = k1
        self.b = b
        self.doc_freq = {}  # 文档频率
        self.idf = {}  # 逆文档频率
        self.avg_doc_len = 0  # 平均文档长度
        self.total_docs = 0  # 文档总数
        self.vocab = {}  # 词汇表
        self.vocab_size = 0
        
    def _tokenize(self, text: str) -> List[str]:
        """
        分词处理
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        return list(jieba.cut(text))
    
    def fit(self, documents: List[str]):
        """
        训练 BM25 模型，计算 IDF 和文档统计信息
        
        Args:
            documents: 文档列表
        """
        self.total_docs = len(documents)
        if self.total_docs == 0:
            return
        
        # 分词并统计
        tokenized_docs = []
        total_len = 0
        
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            total_len += len(tokens)
            
            # 统计文档频率（每个词在多少文档中出现）
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        
        # 计算平均文档长度
        self.avg_doc_len = total_len / self.total_docs if self.total_docs > 0 else 0
        
        # 计算 IDF
        for token, freq in self.doc_freq.items():
            # 使用 BM25 的 IDF 公式
            idf = math.log((self.total_docs - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[token] = idf
        
        # 构建词汇表
        self.vocab = {token: idx for idx, token in enumerate(self.doc_freq.keys())}
        self.vocab_size = len(self.vocab)
        
        print(f"BM25 模型训练完成，词汇表大小: {self.vocab_size}, 文档数: {self.total_docs}")
    
    def encode(self, text: str) -> Dict[int, float]:
        """
        将文本编码为稀疏向量
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[int, float]: 稀疏向量，格式为 {词索引: BM25权重}
        """
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        
        # 统计词频
        token_counts = Counter(tokens)
        
        # 计算 BM25 权重
        sparse_vector = {}
        for token, freq in token_counts.items():
            if token in self.vocab:
                token_idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                
                # BM25 公式
                tf = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)) if self.avg_doc_len > 0 else freq
                score = idf * tf
                
                if score > 0:
                    sparse_vector[token_idx] = score
        
        return sparse_vector
    
    def encode_queries(self, queries: List[str]) -> List[Dict[int, float]]:
        """
        批量编码查询
        
        Args:
            queries: 查询列表
            
        Returns:
            List[Dict[int, float]]: 稀疏向量列表
        """
        return [self.encode(q) for q in queries]
    
    def get_vocab_size(self) -> int:
        """
        获取词汇表大小
        
        Returns:
            int: 词汇表大小
        """
        return self.vocab_size


class BM25SparseEmbeddingService:
    """
    BM25 稀疏向量服务
    单例模式，用于全局共享 BM25 模型
    """
    
    _instance = None
    _bm25_model = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._bm25_model = None
    
    def fit(self, documents: List[str]):
        """
        训练 BM25 模型
        
        Args:
            documents: 文档列表
        """
        self._bm25_model = BM25SparseEmbedding()
        self._bm25_model.fit(documents)
    
    def encode(self, text: str) -> Dict[int, float]:
        """
        编码文本为稀疏向量
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[int, float]: 稀疏向量
        """
        if self._bm25_model is None:
            raise ValueError("BM25 模型未训练，请先调用 fit() 方法")
        return self._bm25_model.encode(text)
    
    def is_trained(self) -> bool:
        """
        检查模型是否已训练
        
        Returns:
            bool: 是否已训练
        """
        return self._bm25_model is not None
