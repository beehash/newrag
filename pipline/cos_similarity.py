import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from config import EMBEDDING_CONFIG

from sentence_transformers import SentenceTransformer

class CosineSimilarityCalculator:
    """
    余弦相似度计算类，用于计算查询和文档的相似度
    """
    
    def __init__(self, model_name=None):
        """
        初始化余弦相似度计算器
        
        Args:
            model_name: 用于生成嵌入的模型名称，默认为配置文件中的值
        """
        self.model = SentenceTransformer(model_name or EMBEDDING_CONFIG["model_name"])
    
    def calculate(self, text1, text2):
        """
        计算两个文本之间的余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 余弦相似度得分
        """
        # 生成嵌入
        embedding1 = self.model.encode(text1)
        embedding2 = self.model.encode(text2)
        
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return float(similarity)
    
    def rank_sentences(self, query, sentences, top_k=3):
        """
        对句子列表按与查询的相似度进行排序
        
        Args:
            query: 查询文本
            sentences: 句子列表
            top_k: 返回的句子数量
            
        Returns:
            list: 排序后的句子列表，每个元素包含句子和相似度得分
        """
        # 计算每个句子与查询的相似度
        scored_sentences = []
        for sentence in sentences:
            similarity = self.calculate(query, sentence)
            scored_sentences.append({
                "sentence": sentence,
                "score": similarity
            })
        
        # 按相似度排序
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回前top_k个结果
        return scored_sentences[:top_k]
