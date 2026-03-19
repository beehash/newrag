import uuid
from datetime import datetime
from ingestion.mv_store import MVStore

# 阿里云向量数据库配置
MILVUS_COLLECTION = "documents"
MILVUS_URI = "your_milvus_uri"
MILVUS_TOKEN = "your_milvus_token"
MILVUS_DB_NAME = "your_milvus_db_name"


class Document:
    def __init__(self, filename, filetype, content, title=None):
        """
        初始化Document对象

        Args:
            filename: 文件名
            filetype: 文件类型
            content: 文件内容
            title: 文档标题，默认为None
        """
        self.filename = filename
        self.filetype = filetype
        self.doc_id = str(uuid.uuid4())
        self.content = content
        self.title = title or filename
        self.create_at = datetime.now().isoformat()

        # 初始化阿里云向量数据库
        self.vector_store = MVStore(
            connection_name="default",
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            db_name=MILVUS_DB_NAME,
            collection_name=MILVUS_COLLECTION
        )

    def get_document(self, doc_id):
        """
        根据doc_id获取文档

        Args:
            doc_id: 文档ID

        Returns:
            Document对象或None
        """
        # 从向量数据库中获取文档
        doc_data = self.vector_store.get_document(doc_id)
        if doc_data:
            # 创建并返回Document对象
            doc = Document(
                filename=doc_data["filename"],
                filetype=doc_data["filetype"],
                content=doc_data["content"],
                title=doc_data["title"]
            )
            doc.doc_id = doc_data["doc_id"]
            doc.create_at = doc_data["create_at"]
            return doc
        return None

    def delete_document(self, doc_id):
        """
        删除文档

        Args:
            doc_id: 文档ID

        Returns:
            bool: 删除是否成功
        """
        # 从向量数据库中删除文档
        return self.vector_store.deleteDocument(doc_id)

    def list_documents(self):
        """
        列出所有文档

        Returns:
            list: 文档列表
        """
        # 从向量数据库中获取所有文档
        return self.vector_store.getAllDocuments()
