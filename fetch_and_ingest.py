#!/usr/bin/env python3
"""
从外部接口获取文件并入库的脚本
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
import requests

# 添加项目根目录到Python路径
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from ingestion.chunker import chunk_text
from ingestion.embedding import EmbeddingService
from ingestion.mv_store import MVStore
from ingestion.es_store import ESStore
from config import MILVUS_CONFIG, FILE_TYPES

# 外部API配置
EXTERNAL_API_BASE = "http://122.51.249.138:5678/api"
FILES_ENDPOINT = f"{EXTERNAL_API_BASE}/files"

# 初始化服务
embedding_service = None
mv_store = None
es_store = None

def init_services():
    """
    初始化所有服务
    """
    global embedding_service, mv_store, es_store
    
    print("正在初始化服务...")
    
    # 初始化Embedding服务
    try:
        embedding_service = EmbeddingService()
        print("Embedding服务初始化成功！")
    except Exception as e:
        print(f"Embedding服务初始化失败: {str(e)}")
        return False
    
    # 初始化Milvus存储
    try:
        mv_store = MVStore()
        print("Milvus存储初始化成功！")
    except Exception as e:
        print(f"Milvus存储初始化失败: {str(e)}")
        return False
    
    # 初始化Elasticsearch存储
    try:
        es_store = ESStore()
        print("Elasticsearch存储初始化成功！")
    except Exception as e:
        print(f"Elasticsearch存储初始化失败: {str(e)}")
        return False
    
    return True

def get_file_list():
    """
    获取文件列表
    """
    print(f"正在获取文件列表: {FILES_ENDPOINT}")
    try:
        response = requests.get(FILES_ENDPOINT)
        response.raise_for_status()
        data = response.json()

        print(f"原始响应数据: {data.get('success', False)}")
        data = data.get("files", [])
        # 提取文件ID列表
        file_ids = []
        if isinstance(data, list):
            for item in data:
                if "_id" in item and "$oid" in item["_id"]:
                    file_ids.append(item["_id"]["$oid"])
        
        print(f"成功获取 {len(file_ids)} 个文件ID")
        return file_ids
    except Exception as e:
        print(f"获取文件列表失败: {str(e)}")
        return []

def get_file_content(file_id):
    """
    获取单个文件内容
    """
    endpoint = f"{FILES_ENDPOINT}/{file_id}"
    print(f"正在获取文件内容: {endpoint}")
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        print(f"原始响应数据: {data.get('success', False)}")
        
        return data.get("file", "")
    except Exception as e:
        print(f"获取文件内容失败: {str(e)}")
        return None

async def process_file(file_info):
    """
    处理单个文件
    """
    try:
        # 构建数据结构
        doc_id = str(uuid.uuid4())
        title = file_info.get("title", "")
        content = file_info.get("text", "")
        filename = f"{title}.md" if title else "untitled.md"
        file_type = "md"
        
        print(f"处理文件: {title}")
        print(f"文件内容长度: {len(content)}")
        
        # 对文本进行切块
        chunks = chunk_text(content)
        print(f"文本切块完成，共 {len(chunks)} 个块")
        
        # 为每个块生成embedding
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            print(f"正在处理第 {i+1}/{len(chunks)} 个块")
            embedding = embedding_service.embed_text(chunk)
            chunk_embeddings.append({
                "chunk": chunk,
                "embedding": embedding
            })
        print("所有块的embedding生成完成")
        
        # 创建对象列表为数据入库做准备
        documents_for_db = []
        for i, item in enumerate(chunk_embeddings):
            documents_for_db.append({
                "filename": filename,
                "doc_id": doc_id,
                "text": item["chunk"],
                "vector": item["embedding"],
                "chunk_id": i + 1,
                "type": file_type,
                "title": title,
                "create_at": datetime.now().isoformat()
            })
        
        # 尝试插入数据到Milvus数据库
        if mv_store is not None:
            try:
                mv_store.insert(documents_for_db)
                print(f"文件 {filename} 成功插入到Milvus数据库")
            except Exception as e:
                print(f"插入Milvus数据库失败: {str(e)}")
                return False
        else:
            print("Milvus数据库连接未初始化，跳过数据库插入")
            return False
        
        # 尝试插入数据到Elasticsearch数据库
        if es_store is not None:
            try:
                # 准备Elasticsearch数据
                es_documents = []
                for doc in documents_for_db:
                    es_documents.append({
                        "chunk_id": doc["chunk_id"],
                        "doc_id": doc["doc_id"],
                        "text": doc["text"],
                        "filename": doc["filename"],
                        "title": doc["title"],
                        "create_at": doc["create_at"],
                        "type": doc["type"]
                    })
                
                # 异步插入数据
                await es_store.add(es_documents)
                print(f"文件 {filename} 成功插入到Elasticsearch数据库，共 {len(es_documents)} 条记录")
                return True
            except Exception as e:
                print(f"插入Elasticsearch数据库失败: {str(e)}")
                # 如果Elasticsearch插入失败，删除已插入到Milvus的数据
                if mv_store is not None:
                    print(f"回滚: 删除Milvus中的数据，doc_id: {doc_id}")
                    try:
                        mv_store.deleteDocument(doc_id)
                        print("已成功回滚Milvus中的数据")
                    except Exception as rollback_error:
                        print(f"回滚Milvus数据失败: {str(rollback_error)}")
                return False
        else:
            print("Elasticsearch数据库连接未初始化，跳过数据库插入")
            return False
            
    except Exception as e:
        print(f"处理文件失败: {str(e)}")
        return False

async def main():
    """
    主函数
    """
    # 初始化服务
    if not init_services():
        print("服务初始化失败，退出脚本")
        return
    
    # 获取文件列表
    file_ids = get_file_list()
    if not file_ids:
        print("没有获取到文件列表，退出脚本")
        return
    
    # 处理每个文件
    success_count = 0
    failure_count = 0
    
    for file_id in file_ids:
        file_info = get_file_content(file_id)
        if file_info:
            success = await process_file(file_info)
            if success:
                success_count += 1
            else:
                failure_count += 1
        else:
            failure_count += 1
        
        # 添加一个小延迟，避免请求过于频繁
        time.sleep(0.5)
    
    # 输出统计信息
    print(f"\n处理完成！")
    print(f"成功处理: {success_count} 个文件")
    print(f"失败处理: {failure_count} 个文件")

if __name__ == "__main__":
    asyncio.run(main())