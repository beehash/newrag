import os
from sentence_transformers import CrossEncoder

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class Reranker:
    """
    重排序类，使用bge-reranker-large模型
    """
    
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表，每个文档包含title和content
            top_k: 返回结果数量
            
        Returns:
            list: 重排序后的文档列表
        """
        # 处理空文档列表的情况
        if not documents:
            print("文档列表为空，跳过重排序")
            return []
        
        # 构建文本对
        text_pairs = []
        print(f"步骤5开始2")
        for doc in documents:
            # 组合标题和内容作为文档文本
            doc_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            text_pairs.append((query, doc_text))
        print(f"步骤5开始3")
        # 计算相关性分数
        scores = self.model.predict(text_pairs, batch_size=16)
        print(f"步骤5开始3")
        # 为文档添加分数
        for i, doc in enumerate(documents):
            # 打印每个文档的分数
            doc["rerank_score"] = float(scores[i])
            # print(f"处理后的资源: {doc.get('title', '')}  {doc.get('rerank_score', 0)}")
        print(f"步骤5开始4")
        # 按分数排序
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        print(f"步骤5开始5")
        # 返回前top_k个结果
        return documents
