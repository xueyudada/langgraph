import re
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class ParseMDLoader(BaseLoader):
    """ 解析MD文件加载器"""

    def __init__(self, md_path: str,encoding: str = 'utf-8'):
        self.md_path = Path(md_path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        if not self.md_path.exists():
            raise FileNotFoundError(f"Not found {self.md_path}")
        # 1.读取MD内容
        with open(self.md_path,"r",encoding=self.encoding) as f:
            content = f.read()

        # 2. 从解析的文件名中反推原始文件和后缀
        md_name = self.md_path.name
        pattern = r"^(.*)_(pdf|docx|xlsx|xls|csv|txt|doc|md)\.md$"
        match = re.match(pattern,md_name,re.IGNORECASE)

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

        # #3.通过生产级工厂函数创建Document，保证数据规范统一
        # yield create_production_document(
        #     content = content,
        #     source_path = original_path,
        #     additional_info={
        #         ""
        #     }
        # )