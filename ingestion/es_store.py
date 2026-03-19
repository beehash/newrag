"""
Elasticsearch存储模块
用于管理Elasticsearch数据库的连接和操作
"""

from elasticsearch import AsyncElasticsearch
from config import ELASTICSEARCH_CONFIG


class ESStore:
    """
    Elasticsearch存储类
    用于管理Elasticsearch数据库的连接和操作
    """
    
    def __init__(self):
        """
        初始化ESStore
        """
        self.client = None
        self.index_name = ELASTICSEARCH_CONFIG.get("index_name", "rag_chunks")
    
    async def _connect(self):
        """
        连接到Elasticsearch数据库
        """
        try:
            hosts = ELASTICSEARCH_CONFIG.get("hosts", ["http://localhost:9200"])
            username = ELASTICSEARCH_CONFIG.get("username", "")
            password = ELASTICSEARCH_CONFIG.get("password", "")
            
            if username and password:
                # 如果提供了用户名和密码，使用认证连接
                self.client = AsyncElasticsearch(
                    hosts=hosts,
                    basic_auth=(username, password)
                )
            else:
                # 无认证连接
                self.client = AsyncElasticsearch(hosts=hosts)
            
            # 测试连接
            if await self.client.ping():
                print("连接到Elasticsearch数据库成功！")
                # 确保索引存在
                await self._ensure_index_exists()
            else:
                print("连接到Elasticsearch数据库失败！")
        except Exception as e:
            print(f"连接Elasticsearch数据库时出错: {str(e)}")
            self.client = None
    
    async def _ensure_index_exists(self):
        """
        确保索引存在，如果不存在则创建
        """
        try:
            if not await self.client.indices.exists(index=self.index_name):
                # 创建索引
                await self.client.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "chunk_id": {
                                    "type": "keyword"
                                },
                                "doc_id": {
                                    "type": "keyword"
                                },
                                "text": {
                                    "type": "text",
                                    "analyzer": "ik_max_word",
                                    "search_analyzer": "ik_smart"
                                }
                            }
                        }
                    }
                )
                print(f"索引 {self.index_name} 创建成功！")
            else:
                print(f"索引 {self.index_name} 已存在")
        except Exception as e:
            print(f"创建索引时出错: {str(e)}")
    
    async def add(self, data):
        """
        向Elasticsearch添加数据
        
        Args:
            data: 要添加的数据，可以是单个字典或字典列表
                单个数据格式: {
                    "chunk_id": "chunk_id",
                    "doc_id": "doc_id",
                    "text": "text"
                }
        """
        try:
            if not self.client:
                await self._connect()
                if not self.client:
                    print("客户端未初始化")
                    return False
            
            if isinstance(data, list):
                # 批量添加
                for item in data:
                    chunk_id = item.get("chunk_id")
                    if chunk_id:
                        await self.client.index(
                            index=self.index_name,
                            id=chunk_id,
                            body=item
                        )
                print(f"批量添加 {len(data)} 条数据成功")
            else:
                # 单个添加
                chunk_id = data.get("chunk_id")
                if chunk_id:
                    await self.client.index(
                        index=self.index_name,
                        id=chunk_id,
                        body=data
                    )
                    print(f"添加数据成功，chunk_id: {chunk_id}")
                else:
                    print("chunk_id不能为空")
                    return False
            
            return True
        except Exception as e:
            print(f"添加数据时出错: {str(e)}")
            return False
    
    async def delete(self, chunk_id):
        """
        删除指定chunk_id的数据
        
        Args:
            chunk_id: 要删除的数据的chunk_id
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if not self.client:
                await self._connect()
                if not self.client:
                    print("客户端未初始化")
                    return False
            
            await self.client.delete(
                index=self.index_name,
                id=chunk_id
            )
            print(f"删除数据成功，chunk_id: {chunk_id}")
            return True
        except Exception as e:
            print(f"删除数据时出错: {str(e)}")
            return False
    
    async def delete_by_doc_id(self, doc_id):
        """
        根据doc_id删除所有相关数据
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if not self.client:
                await self._connect()
                if not self.client:
                    print("客户端未初始化")
                    return False
            
            # 使用Delete By Query API删除所有匹配doc_id的文档
            await self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {
                            "doc_id": doc_id
                        }
                    }
                }
            )
            print(f"删除doc_id为 {doc_id} 的所有数据成功")
            return True
        except Exception as e:
            print(f"删除数据时出错: {str(e)}")
            return False
    
    async def retrieval(self, query, limit=10):
        """
        使用match模式检索数据
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            list: 检索结果列表
        """
        try:
            if not self.client:
                await self._connect()
                if not self.client:
                    print("客户端未初始化")
                    return []
            
            # 使用match查询
            search_result = await self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "text": query
                        }
                    },
                    "size": limit,
                    "_source": ["chunk_id", "doc_id", "text"]
                }
            )
            
            # 处理检索结果
            results = []
            for hit in search_result.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                score = hit.get("_score", 0)
                results.append({
                    "chunk_id": source.get("chunk_id"),
                    "doc_id": source.get("doc_id"),
                    "text": source.get("text"),
                    "score": score
                })
            
            print(f"检索完成，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            print(f"检索数据时出错: {str(e)}")
            return []
    
    async def close(self):
        """
        关闭Elasticsearch连接
        """
        try:
            if self.client:
                await self.client.close()
                print("Elasticsearch连接已关闭")
        except Exception as e:
            print(f"关闭连接时出错: {str(e)}")


# 测试代码
import asyncio

async def test_es_store():
    es_store = ESStore()
    await es_store._connect()
    
    # 测试添加数据
    test_data = {
        "chunk_id": "test_chunk_1",
        "doc_id": "test_doc_1",
        "text": "这是测试内容，Elasticsearch是一个分布式搜索引擎"
    }
    await es_store.add(test_data)
    
    # 测试检索
    results = await es_store.retrieve("Elasticsearch")
    print("检索结果:", results)
    
    # 测试删除
    await es_store.delete("test_chunk_1")
    
    # 关闭连接
    await es_store.close()

if __name__ == "__main__":
    asyncio.run(test_es_store())
