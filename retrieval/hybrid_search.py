from pydoc import doc
from sympy import fu
from ingestion.es_store import ESStore

class HybridSearch:
    """
    混合搜索类，融合Milvus向量检索和Elasticsearch BM25检索
    """
    
    def __init__(self, mv_store):
        """
        初始化混合搜索
        
        Args:
            mv_store: MVStore实例
        """
        self.mv_store = mv_store
        self.es_store = ESStore()
    
    async def search(self, query, query_embedding, limit=10, k=60, entities=None):
        """
        执行混合搜索
        
        Args:
            query: 查询文本
            query_embedding: 查询的密集向量
            limit: 返回结果数量
            k: RRF参数
            entities: 实体词列表，用于Elasticsearch检索的boost
            
        Returns:
            list: 融合后的搜索结果
        """
        try:
            # 1. 执行Milvus向量检索
            print(f"执行Milvus向量检索")
            vector_results = self.mv_store.retrieval(query_embedding, limit=limit)
            print(f"Milvus向量检索完成，结果数量: {len(vector_results)}")
            
            # 2. 执行Elasticsearch BM25检索，支持实体词boost
            print(f"执行Elasticsearch BM25检索")
            print(f"使用实体词boost: {entities}")
            es_results = await self.es_store.retrieval(query, entities=entities, limit=limit)
            print(f"Elasticsearch BM25检索完成，结果数量: {len(es_results)}")
            return  {"vector_results": vector_results, "es_results": es_results}
        except Exception as e:
            print(f"混合搜索失败: {str(e)}")
            # 发生错误时，返回空结果
            return []
    
    def _rrf_fusion(self, all_results, k=60, limit=10):
        """
        使用RRF（Reciprocal Rank Fusion）算法融合搜索结果
        
        Args:
            vector_results: Milvus向量检索结果
            es_results: Elasticsearch BM25检索结果
            k: RRF参数
            limit: 返回结果数量
            
        Returns:
            list: 融合后的搜索结果
        """
        # 构建文档得分映射
        rrf_scores = {}
        # print(f"vector_results: {vector_results}")
        # 处理Milvus向量检索结果
        for rank, result in enumerate(all_results, 1):
            doc_id = result.get("doc_id")
            if doc_id is not None:
                # 确保chunk_id是字符串 
                score = 1.0 / (k + rank)
                key = doc_id
                if key not in rrf_scores:
                    rrf_scores[key] = 0
                rrf_scores[key] += score
            # print(f"处理Milvus结果: {result.get('title', '')} {result.get('similarity', 0)} {rrf_scores[doc_id]}")
        # 按得分排序
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"RRF得分映射: {rrf_scores}")
        # 构建最终结果
        fused_results = []
        fused_info = {}
        for doc_id, score in sorted_chunks:
            # 从原始结果中获取文档信息
            if doc_id in fused_info:
                continue
            chunk_info = None
            # 先从Milvus结果中查找
            for result in all_results:
                if result.get('doc_id') == doc_id:
                    chunk_info = result.copy()
                    break
            # 如果找到了文档信息，更新得分
            if chunk_info and doc_id not in fused_info:
                chunk_info["score"] = score
                fused_info[doc_id] = True
                fused_results.append(chunk_info)
        
        return fused_results
