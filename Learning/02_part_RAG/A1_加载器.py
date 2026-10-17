import hashlib
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

"""
ParsedMarkdownLoader (基于解析后 Markdown 的统一加载器)

在生产环境中，我们将文档物理格式解析（Docling/VLM）与数据加载独立开来。
本加载器负责读取由 `A1_解析器.py` 生成的 Markdown 文件，
并自动根据文件名推断并重建出原始文件的相关元数据（如原始后缀名、原始文件路径等），
这对于在向量库中做多文档类型联合过滤（Hybrid Filtering）至关重要。
"""


def create_production_document(
    content: str,
    source_path: Path,
    chunk_index: int = 0,
    total_chunks: int = 1,
    additional_info: dict | None = None,
) -> Document:
    """
    创建一个生产级别的 Document 对象。

    统一的 Document 工厂函数，保证所有入库文档的元数据结构一致，
    这对于向量库的 Hybrid Filtering、去重和权限管控至关重要。
    """

    # 1. 生成内容哈希：用于去重和判断内容是否更新
    #    即使文件名改了，只要内容不变，hash 就不变
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 2. 生成唯一的 Document ID
    #    结合文件路径和分片索引，确保在向量数据库中是唯一的
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_path}_{chunk_index}"))

    # 3. 构造丰富元数据
    metadata: dict = {
        # --- 基础溯源信息 ---
        "source": str(source_path.resolve()),            # 绝对路径，方便定位原始文件
        "filename": source_path.name,                    # 文件名
        "file_type": source_path.suffix.replace(".", ""), # 文件类型 (pdf, docx)

        # --- 数据处理追踪 ---
        "doc_id": doc_id,                                # 唯一标识符
        "content_hash": content_hash,                    # 内容指纹
        "chunk_index": chunk_index,                      # 当前是第几个片段
        "total_chunks": total_chunks,                    # 总共有多少片段
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), # 处理时间

        # --- 增强检索信息 ---
        "content_length": len(content),                  # 字符长度
        "language": "zh",                                # 语言标识

        # --- 业务/权限属性 (生产环境必备) ---
        "owner": "research_dept",                        # 数据归属部门
        "access_level": "internal",                      # 权限级别 (公开/内部/机密)
    }

    # 4. 合并额外自定义信息 (如：大模型生成的摘要、关键词)
    if additional_info:
        metadata.update(additional_info)

    return Document(
        page_content=content,
        metadata=metadata,
    )


class ParsedMarkdownLoader(BaseLoader):
    """
    解析后 Markdown 加载器
    读取通过布局解析生成的 Markdown，重构回其原始文件的信息。
    """
    def __init__(self, md_path: str, encoding: str = "utf-8"):
        self.md_path = Path(md_path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        if not self.md_path.exists():
            raise FileNotFoundError(f"未找到指定的 Markdown 文件: {self.md_path}")

        # 1. 读取 Markdown 内容
        with open(self.md_path, "r", encoding=self.encoding) as f:
            content = f.read()

        # 2. 从解析后的文件名反推原始文件名和后缀
        # A1_解析器的命名规则为：{原文件名}_{原后缀}.md
        # 示例：text_pdf.md -> 原文件 text.pdf，原后缀 .pdf
        md_name = self.md_path.name
        pattern = r"^(.*)_(pdf|docx|xlsx|xls|csv|txt|doc|md)\.md$"
        match = re.match(pattern, md_name, re.IGNORECASE)

        if match:
            base_name = match.group(1)
            original_ext = f".{match.group(2).lower()}"
            original_filename = f"{base_name}{original_ext}"
            # 拼接出假定的原始文件物理路径
            original_path = self.md_path.parent / original_filename
        else:
            # 兜底方案
            original_ext = ".md"
            original_path = self.md_path

        # 3. 通过生产级工厂函数创建 Document，保证元数据规范统一
        yield create_production_document(
            content=content,
            source_path=original_path,
            additional_info={
                "parsed_path": str(self.md_path.resolve()),  # 解析后的 md 文件路径
                "original_suffix": original_ext,             # 原始文件后缀 (例如: .pdf)
            },
        )


def main():
    # 查找本地是否有已经解析好的 md 文件
    current_dir = Path(__file__).parent
    md_files = list(current_dir.glob("*_*.md"))

    if not md_files:
        print("[提示] 目录下暂无由解析器生成的文件 (如 text_pdf.md)。")
        print("现在将为您创建一个临时测试用例：'mock_sample_docx.md' ...")
        # 创建一个模拟的 md 文件来测试加载
        test_md_path = current_dir / "mock_sample_docx.md"
        with open(test_md_path, "w", encoding="utf-8") as f:
            f.write("# 统一测试内容\n这是一个模拟从 Word 转换为 Markdown 的文档内容。")
        md_files = [test_md_path]
    else:
        print(f"检测到目录下存在已解析的文档：{[f.name for f in md_files]}")

    print("\n=== 开始加载测试 ===")
    for md_path in md_files:
        print("-" * 50)
        print(f"正在加载 Markdown 文件: {md_path.name}")
        
        # 实例化自定义加载器
        loader = ParsedMarkdownLoader(str(md_path))
        docs = loader.load()
        
        for idx, doc in enumerate(docs):
            print(f"\n[Document {idx+1}]")
            print(f"元数据 (Metadata) 详情:")
            for k, v in doc.metadata.items():
                print(f"  ├── {k}: {v}")
            print(f"文档内容 (前150字):\n{doc.page_content[:150]}")
            
    # 如果创建了临时文件，提示清理或保留
    temp_mock = current_dir / "mock_sample_docx.md"
    if temp_mock.exists() and "mock_sample_docx.md" in [f.name for f in md_files]:
        try:
            os.remove(temp_mock)
            print("\n已成功清理临时自测文件 'mock_sample_docx.md'。")
        except Exception:
            pass

if __name__ == "__main__":
    main()
