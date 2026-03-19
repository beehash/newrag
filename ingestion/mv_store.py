import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from config import MILVUS_CONFIG

from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection
from ingestion.embedding import EmbeddingService

class MVStore:
    def __init__(self, connection_name=None, uri=None, token=None, db_name=None, collection_name=None):
        """
        初始化MVStore对象
        
        Args:
            connection_name: 连接名称，默认为配置文件中的值
            uri: 向量数据库URI，默认为配置文件中的值
            token: 访问令牌，默认为配置文件中的值
            db_name: 数据库名称，默认为配置文件中的值
            collection_name: 集合名称，默认为配置文件中的值
        """
        self.connection_name = connection_name or MILVUS_CONFIG["connection_name"]
        self.uri = uri or f"tcp://{MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}"
        self.token = token or MILVUS_CONFIG["token"]
        self.db_name = db_name or MILVUS_CONFIG["db_name"]
        self.collection_name = collection_name or MILVUS_CONFIG["collection_name"]
        self.embedding_model = None
        self.client = None
        
        # 连接到Milvus数据库
        self._connect()
    
    def _connect(self):
        """
        连接到Milvus数据库，支持重连
        """
        print(f"连接到Milvus数据库: {self.uri}")
        try:
            self.client = MilvusClient(
                uri=self.uri,
                db_name=self.db_name
            )
            print("Milvus数据库连接成功！")
            
            # 如果集合不存在，创建新集合
            if not self.client.has_collection(collection_name=self.collection_name):
                self._create_collection()
            else:
                print(f"集合 {self.collection_name} 已存在")

            # 创建集合后需要加载集合
            print(f"加载集合 {self.collection_name}...")
            self.client.load_collection(collection_name=self.collection_name)
            print(f"集合 {self.collection_name} 加载完成")
        except Exception as e:
            print(f"Milvus数据库连接失败: {str(e)}")
            self.client = None
    
    def _create_collection(self):
        """
        创建集合
        """
        try:
            # 建立连接
            host = MILVUS_CONFIG.get("host", "127.0.0.1")
            port = MILVUS_CONFIG.get("port", "19530")
            connections.connect(host=host, port=port)
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # 使用VARCHAR替代TEXT
                FieldSchema(name="create_at", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._get_embedding_dimension())
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, description="Document collection")
            
            # 创建集合
            collection = Collection(name=self.collection_name, schema=schema)
            
            # 创建索引 - 与用户提供的示例保持一致
            index_params = {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "nprobe": 16}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            
            print(f"集合 {self.collection_name} 创建成功！")
            
        except Exception as e:
            print(f"创建集合失败: {str(e)}")
    
    def _check_and_reconnect(self):
        """
        检查连接状态，如果断开则重新连接
        """
        if self.client is None:
            print("检测到客户端未初始化，尝试重新连接...")
            self._connect()
            return
        
        # 尝试执行一个简单的操作来检查连接是否有效
        try:
            # 使用 has_collection 检查连接状态
            self.client.has_collection(collection_name=self.collection_name)
        except Exception as e:
            print(f"检测到连接已断开: {str(e)}，尝试重新连接...")
            self._connect()
    
    def _get_embedding_dimension(self):
        """
        获取embedding向量的维度
        
        Returns:
            int: 向量维度，bge-m3模型为1024维
        """
        return 1024
    
    def getEmbeddingModel(self):
        """
        获取embedding模块，方便此模块中的向量化操作
        
        Returns:
            EmbeddingService: embedding服务实例
        """
        if self.embedding_model is None:
            self.embedding_model = EmbeddingService()
        return self.embedding_model
    
    def get_document(self, doc_id):
        """
        根据doc_id查询文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            dict: 文档信息，包含{filename, filetype, title, create_at, doc_id, content}
        """
        # 检查连接状态，必要时重新连接
        self._check_and_reconnect()
        
        if self.client is None:
            print("客户端未初始化")
            return None
        
        # 从向量数据库中查询所有匹配的doc_id
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f"doc_id == '{doc_id}'",
            output_fields=["doc_id", "filename", "type", "text", "title", "create_at", "chunk_id"]
        )
        
        if not results:
            return None
        
        # 根据chunk_id排序
        results.sort(key=lambda x: x.get("chunk_id", 0))
        
        # 拼接text
        content = "".join([result["text"] for result in results])
        
        # 返回文档信息
        return {
            "filename": results[0]["filename"],
            "filetype": results[0]["type"],
            "title": results[0]["title"],
            "create_at": results[0]["create_at"],
            "doc_id": doc_id,
            "content": content
        }
    
    def getAllDocuments(self):
        """
        获取所有文档
        
        Returns:
            list: 文档列表，每个元素为{filename, filetype, title, create_at, doc_id, content}
        """
        # 检查连接状态，必要时重新连接
        self._check_and_reconnect()
        
        if self.client is None:
            print("客户端未初始化")
            return []
        print("开始获取所有文档...")
        # 从向量数据库中查询所有数据
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=None,
                output_fields=["doc_id", "filename", "type", "text", "title", "create_at", "chunk_id"],
                limit=1000
            )
            print(f"查询结果数量: {len(results)}")
            # 根据doc_id分组
            doc_groups = {}
            for result in results:
                doc_id = result["doc_id"]
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(result)
            print(f"分组后文档数量: {len(doc_groups)}")
            # 处理每个文档组
            documents = []
            for doc_id, group in doc_groups.items():
                # 根据chunk_id排序
                group.sort(key=lambda x: x.get("chunk_id", 0))
                
                # 拼接text
                content = "".join([item["text"] for item in group])
                
                # 添加到文档列表
                documents.append({
                    "filename": group[0]["filename"],
                    "filetype": group[0]["type"],
                    "title": group[0]["title"],
                    "create_at": group[0]["create_at"],
                    "doc_id": doc_id,
                    "content": content
                })
            print(f"最终文档数量: {len(documents)}")
            return documents
        except Exception as e:
            print(f"获取所有文档失败: {str(e)}")
            return []
    
    def deleteDocument(self, doc_id):
        """
        根据doc_id删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 删除是否成功
        """
        # 检查连接状态，必要时重新连接
        self._check_and_reconnect()
        
        if self.client is None:
            print("客户端未初始化")
            return False
        
        # 从向量数据库中删除所有匹配的doc_id
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=f"doc_id == '{doc_id}'"
            )
            # 等待一小段时间，确保删除操作完成
            import time
            time.sleep(0.5)
            # 检查删除是否成功
            return True
        except Exception as e:
            print(f"删除文档失败: {str(e)}")
            return False
    
    def insert(self, data):
        """
        插入数据到向量数据库
        
        Args:
            data: 要插入的数据列表，每个元素包含text字段
            
        Returns:
            list: 插入的实体ID列表
        """
        # 检查连接状态，必要时重新连接
        self._check_and_reconnect()
        
        if self.client is None:
            print("客户端未初始化")
            return []
        
        try:
            print(f"开始插入数据，共 {len(data)} 条")
            
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            print(f"插入成功，结果: {result}")
            # 等待一小段时间，确保数据已写入
            import time
            time.sleep(0.5)
            return result
        except Exception as e:
            print(f"插入数据失败: {str(e)}")
            return []
    
    def retireval(self, query_embedding, limit=10):
        """
        搜索相似向量
        
        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            
        Returns:
            list: 搜索结果
        """
        # 检查连接状态，必要时重新连接
        self._check_and_reconnect()
        
        if self.client is None:
            print("客户端未初始化")
            return []
        
        # 搜索参数 - 与用户提供的示例保持一致
        search_params = {
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="vector",  # 指定使用密集向量字段
            limit=limit,
            output_fields=["doc_id", "chunk_id", "filename", "title", "text", "create_at"],
            search_params=search_params
        )
        
        # 处理搜索结果，添加score字段
        docs = []
        for hit in results[0]:
            docs.append({
                "doc_id": hit.entity.get('doc_id'),
                "chunk_id": hit.entity.get('chunk_id'),
                "filename": hit.entity.get('filename'),
                "title": hit.entity.get('title'),
                "text": hit.entity.get('text'),
                "create_at": hit.entity.get('create_at'),
                "score": hit.distance  # 余弦相似度
            })
        
        return docs