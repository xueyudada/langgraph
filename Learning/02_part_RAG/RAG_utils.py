import time
import uuid
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class CustomZhipuEmbeddings(Embeddings):
    """
    自研智谱 AI 向量模型适配器 (离线安全级)
    
    解决痛点：
      直接使用 langchain_openai 或是原生 embeddings 在部分封闭/受限网络环境下，
      由于需要在线下载 cl100k_base 编码器分词包，会高概率导致 SSL 或超时报错。
      本类直接通过 requests 请求智谱开放 API 接口，不产生任何 tiktoken 本地外网依赖。
    """
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.url = f"{self.base_url}/embeddings"
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 智谱接口限制单次请求 input 数组长度不能超过 64，进行分批请求
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "model": self.model,
                "input": batch
            }
            try:
                response = requests.post(self.url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                # 排序以防多文本请求响应乱序
                embeddings_sorted = sorted(data["data"], key=lambda x: x["index"])
                batch_embeddings = [item["embedding"] for item in embeddings_sorted]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                err_msg = ""
                if 'response' in locals() and response is not None:
                    try:
                        err_msg = f" 响应内容: {response.text}"
                    except:
                        pass
                raise RuntimeError(f"❌ 智谱 Embedding 接口请求失败: {e}{err_msg}")
                
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def create_production_document(
    content: str,
    source_path: Path,
    chunk_index: int = 0,
    total_chunks: int = 1,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Document:
    """
    生产级统一元数据 Document 工厂函数
    
    100% 对齐 A1_加载器.py 中的元数据规范，确保入库文档检索一致性。
    """
    # 1. 生成内容 SHA256 哈希作为内容指纹
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 2. 生成基于 DNS 命名空间和路径的唯一 ID，确保防重
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_path.resolve()}_{chunk_index}"))

    # 3. 按照标准字段构造元数据字典
    metadata: dict = {
        # --- 基础溯源信息 ---
        "source": str(source_path.resolve()),            # 绝对路径，方便定位原始文件
        "filename": source_path.name,                    # 文件名
        "file_type": source_path.suffix.replace(".", ""), # 原始物理文件类型 (pdf, docx)

        # --- 数据处理追踪 ---
        "doc_id": doc_id,                                # 唯一标识符
        "content_hash": content_hash,                    # 内容指纹
        "chunk_index": chunk_index,                      # 当前是第几个片段
        "total_chunks": total_chunks,                    # 总共有多少片段
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), # 处理时间

        # --- 增强检索信息 ---
        "content_length": len(content),                  # 字符长度
        "language": "zh",                                # 语言标识

        # --- 业务/权限属性 ---
        "owner": "research_dept",                        # 数据归属部门
        "access_level": "internal",                      # 权限级别
    }

    # 4. 合并自定义元数据（如父块 ID、标题等级等）
    if additional_info:
        metadata.update(additional_info)

    return Document(
        page_content=content,
        metadata=metadata,
    )
