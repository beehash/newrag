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
    
    async def search(self, query, query_embedding, limit=10, k=60):
        """
        执行混合搜索
        
        Args:
            query: 查询文本
            query_embedding: 查询的密集向量
            limit: 返回结果数量
            k: RRF参数
            
        Returns:
            list: 融合后的搜索结果
        """
        try:
            # 1. 执行Milvus向量检索
            print(f"执行Milvus向量检索，limit: {limit}")
            vector_results = self.mv_store.retrieval(query_embedding, limit=limit)
            print(f"Milvus向量检索完成，结果数量: {len(vector_results)}")
            
            # 2. 执行Elasticsearch BM25检索
            print(f"执行Elasticsearch BM25检索，limit: {limit}")
            es_results = await self.es_store.retrieval(query, limit=limit)
            print(f"Elasticsearch BM25检索完成，结果数量: {len(es_results)}")
            
            # 3. 融合搜索结果（使用RRF算法）
            fused_results = self._rrf_fusion(vector_results, es_results, k=k, limit=limit)
            
            return fused_results
        except Exception as e:
            print(f"混合搜索失败: {str(e)}")
            # 发生错误时，返回空结果
            return []
    
    def _rrf_fusion(self, vector_results, es_results, k=60, limit=10):
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
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.get("chunk_id")
            doc_id = result.get("doc_id")
            if chunk_id is not None and doc_id is not None:
                # 确保chunk_id是字符串
                chunk_id_str = str(chunk_id)
                doc_id_str = str(doc_id)
                score = 1.0 / (k + rank)
                key = f"{doc_id_str}_{chunk_id_str}"
                if key not in rrf_scores:
                    rrf_scores[key] = 0
                rrf_scores[key] += score
        # 处理Elasticsearch BM25检索结果
        for rank, result in enumerate(es_results, 1):
            chunk_id = result.get("chunk_id")
            doc_id = result.get("doc_id")
            if chunk_id is not None and doc_id is not None:
                # 确保chunk_id是字符串
                chunk_id_str = str(chunk_id)
                doc_id_str = str(doc_id)
                score = 1.0 / (k + rank)
                key = f"{doc_id_str}_{chunk_id_str}"
                if key not in rrf_scores:
                    rrf_scores[key] = 0
                rrf_scores[key] += score
        # 按得分排序
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # 构建最终结果
        fused_results = []
        for key_id, score in sorted_chunks:
            # 从右侧分割，只分割一次，避免doc_id或chunk_id中包含"_"的情况
            parts = key_id.rsplit("_", 1)
            if len(parts) == 2:
                doc_id, chunk_id = parts
            else:
                # 如果分割失败，跳过这个结果
                continue

            # 从原始结果中获取文档信息
            chunk_info = None
            # 先从Milvus结果中查找
            for result in vector_results:
                if str(result.get('chunk_id')) == chunk_id and str(result.get('doc_id')) == doc_id:
                    chunk_info = result.copy()
                    break
            # 如果Milvus中没有，从ES结果中查找
            if not chunk_info:
                for result in es_results:
                    if str(result.get('chunk_id')) == chunk_id and str(result.get('doc_id')) == doc_id:
                        chunk_info = result.copy()
                        break
            # 如果找到了文档信息，更新得分
            if chunk_info:
                chunk_info["score"] = score
                fused_results.append(chunk_info)
        
        return fused_results
