import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from config import EMBEDDING_CONFIG

from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name=None):
        """
        初始化Embedding服务
        
        Args:
            model_name: 模型名称，默认为配置文件中的值
        """
        self.model = SentenceTransformer(model_name or EMBEDDING_CONFIG["model_name"])
    
    def embed_text(self, text):
        """
        将单个文本转换为embedding向量
        
        Args:
            text: 要转换的文本
            
        Returns:
            list: embedding向量
        """
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    def embed_batch(self, texts):
        """
        批量将文本转换为embedding向量
        
        Args:
            texts: 要转换的文本列表
            
        Returns:
            list: embedding向量列表
        """
        return self.model.encode(texts).tolist()