"""
配置文件，管理项目中的所有配置信息
"""

# Milvus向量数据库配置
MILVUS_CONFIG = {
    "connection_name": "rag-milvus",
    "host": "127.0.0.1",  # Milvus服务主机
    "port": "19530",  # Milvus服务端口
    "token": "",  # Milvus认证令牌（如果不需要认证，留空即可）
    "db_name": "default",  # 数据库名称
    "collection_name": "zw_docs"  # 集合名称
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-m3"  # 用于生成文本嵌入的模型
}

# 重排序模型配置
RERANK_CONFIG = {
    "model_name": "BAAI/bge-reranker-large"  # 用于重排序的模型
}

# 大模型配置
LLM_CONFIG = {
    "model_name": "qwen3-235b-a22b-instruct-2507",  # 使用的大模型名称
    "api_key": "sk-54a9eaf0c047451d888a2387a1cd25d9",  # 大模型API密钥
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 大模型API基础URL
}

# 智谱AI配置（用于查询改写和意图识别）
ZHIPU_CONFIG = {
    "api_key": "",  # 智谱AI API密钥
    "base_url": "https://open.bigmodel.cn/api/paas/v4",  # 智谱AI API基础URL
    "rewrite_model": "glm-3-turbo",  # 用于查询改写的模型（ChatGLM3-6B开源模型）
    "intent_model": "glm-3-turbo"  # 用于意图识别的模型（ChatGLM3-6B开源模型）
}

# 服务器配置
SERVER_CONFIG = {
    "host": "127.0.0.1",  # 服务器主机
    "port": 8083,  # 服务器端口
    "upload_dir": "uploads"  # 文件上传目录
}

# 搜索配置
SEARCH_CONFIG = {
    "top_k": 5,  # 默认返回结果数量
    "hybrid_k": 60  # RRF融合参数
}

# 文件类型配置
FILE_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md"
}

# Elasticsearch配置
ELASTICSEARCH_CONFIG = {
    "hosts": ["http://localhost:9200"],  # Elasticsearch服务地址
    "index_name": "rag_chunks",  # 索引名称
    "username": "",  # 用户名（如果不需要认证，留空即可）
    "password": ""  # 密码（如果不需要认证，留空即可）
}
