import os
import sys
import tempfile
import csv
from typing import Iterator, Dict, Type
import pandas as pd
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

"""
LangChain BaseLoader 统一协议接口实现

在企业级 RAG 开发中，通常需要实现自定义的 Loader 以对接特定的业务系统或数据格式。
本示例通过继承 LangChain 的 `BaseLoader` 统一基类协议，实现了一套高可扩展的文件加载系统。

核心设计：
1. `BaseLoader` 统一协议：子类只需实现 `lazy_load` 迭代器方法，`load` 方法会自动继承并以内存友好的方式运行。
2. 格式专用 Loader：Txt、Csv 和 Excel 分别实现各自的 `lazy_load` 逻辑。
3. `UnifiedDocumentLoader` 路由分发器：继承 `BaseLoader`，对外作为统一接口，内部根据后缀名路由到对应的特定加载器。
"""

class CustomTxtLoader(BaseLoader):
    """自定义文本加载器，读取纯文本或 Markdown"""
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, "r", encoding=self.encoding) as f:
            content = f.read()
        yield Document(
            page_content=content,
            metadata={"source": self.file_path, "type": "txt"}
        )


class CustomCsvLoader(BaseLoader):
    """自定义 CSV 加载器，按行惰性加载"""
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                # 将每一行键值对拼接为文本表达
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                yield Document(
                    page_content=content,
                    metadata={"source": self.file_path, "row": row_idx, "type": "csv"}
                )


class CustomExcelLoader(BaseLoader):
    """自定义 Excel 加载器，使用 pandas 解析并将工作表转为 Markdown 表格"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        # 生产环境实践：使用 with 确保文件句柄在使用完后被及时释放，防止 Windows 独占锁导致无法清理
        with pd.ExcelFile(self.file_path) as xls:
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                # 转换整表为 Markdown 表格，有助于大模型理解其排版和对齐语义
                markdown_table = df.to_markdown(index=False)
                yield Document(
                    page_content=f"### Sheet: {sheet_name}\n\n{markdown_table}",
                    metadata={
                        "source": self.file_path,
                        "sheet_name": sheet_name,
                        "format": "excel_markdown",
                        "type": "excel"
                    }
                )


class CustomPdfLoader(BaseLoader):
    """自定义 PDF 加载器，基于 pypdf 按页惰性加载"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        try:
            import pypdf
        except ImportError:
            raise ImportError("未检测到 pypdf 库，请先安装：`pip install pypdf`")
            
        with open(self.file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                yield Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path,
                        "page": page_idx + 1,
                        "total_pages": len(reader.pages),
                        "type": "pdf"
                    }
                )


class CustomDocxLoader(BaseLoader):
    """自定义 Word (.docx) 加载器，基于 python-docx 提取段落文本"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        try:
            import docx
        except ImportError:
            raise ImportError("未检测到 python-docx 库，请先安装：`pip install python-docx`")
            
        doc = docx.Document(self.file_path)
        paragraphs_text = [p.text for p in doc.paragraphs if p.text.strip()]
        content = "\n".join(paragraphs_text)
        
        yield Document(
            page_content=content,
            metadata={"source": self.file_path, "type": "docx"}
        )


class UnifiedDocumentLoader(BaseLoader):
    """
    统一文档加载器（路由分发器模式）
    继承自 `BaseLoader` 统一协议，对外暴露一致的 `load` 和 `lazy_load` 方法。
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        # 维护后缀名与对应 Loader 类的映射注册表
        self._loaders_map: Dict[str, Type[BaseLoader]] = {
            ".txt": CustomTxtLoader,
            ".md": CustomTxtLoader,
            ".csv": CustomCsvLoader,
            ".xlsx": CustomExcelLoader,
            ".xls": CustomExcelLoader,
            ".pdf": CustomPdfLoader,
            ".docx": CustomDocxLoader
        }

    def lazy_load(self) -> Iterator[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"未找到该文件: {self.file_path}")

        # 获取文件后缀名
        ext = os.path.splitext(self.file_path)[1].lower()
        
        # 获取对应的具体加载器类
        loader_cls = self._loaders_map.get(ext)
        if not loader_cls:
            raise ValueError(f"暂不支持的文件后缀类型: {ext}")
            
        print(f"[UnifiedLoader] 检测到后缀 '{ext}'，分发至: {loader_cls.__name__}")
        
        # 实例化具体的加载器，并委托其进行 lazy_load
        loader_instance = loader_cls(self.file_path)
        yield from loader_instance.lazy_load()


# ==========================================
# 下方为测试辅助代码
# ==========================================

def create_mock_data():
    """生成内置自测数据"""
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = temp_dir.name
    
    txt_path = os.path.join(dir_path, "mock_test.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("这里是统一协议文本加载测试。")
        
    csv_path = os.path.join(dir_path, "mock_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("name,role,status\nAlice,Agent,Active\nBob,RAG-Engine,Building")
        
    xlsx_path = os.path.join(dir_path, "mock_excel.xlsx")
    df = pd.DataFrame({
        "配置项": ["API_KEY", "MODEL_NAME", "TEMPERATURE"],
        "值": ["sk-xxxx", "gpt-4o", 0.7]
    })
    df.to_excel(xlsx_path, index=False, sheet_name="Settings")
    
    return temp_dir, txt_path, csv_path, xlsx_path


def main():
    file_to_load = None
    temp_dir = None
    
    if len(sys.argv) > 1:
        file_to_load = sys.argv[1]
    else:
        print("[提示] 您可以在命令行以 `python A1_DocumentLoader.py <您的文件路径>` 方式运行。")
        user_input = input("请输入要测试加载的文件绝对路径 (直接回车将运行内置自测): ").strip()
        if user_input:
            file_to_load = user_input.strip('"')

    try:
        if file_to_load:
            print(f"\n开始加载用户指定文件: {file_to_load}")
            # 实例化统一加载器
            loader = UnifiedDocumentLoader(file_to_load)
            
            # 使用统一协议的 lazy_load 惰性加载
            for i, doc in enumerate(loader.lazy_load()):
                print(f"\n--- [分片 {i+1}] ---")
                print(f"元数据 (Metadata): {doc.metadata}")
                print(f"正文内容 (前300字):\n{doc.page_content[:300]}")
        else:
            print("\n正在生成并测试内置自测文件 (txt, csv, xlsx)...")
            temp_dir, txt_path, csv_path, xlsx_path = create_mock_data()
            
            for path in [txt_path, csv_path, xlsx_path]:
                print("\n" + "=" * 50)
                loader = UnifiedDocumentLoader(path)
                
                # 使用统一协议的 load 方法 (load 会在内部收集 lazy_load 返回的 Document 列表)
                docs = loader.load()
                print(f"加载成功！共得到 {len(docs)} 个 Document")
                for idx, doc in enumerate(docs):
                    print(f"- Document {idx+1} (Metadata: {doc.metadata}):")
                    print(f"{doc.page_content}")
            print("=" * 50)
            
    except Exception as e:
        print(f"\n[错误] 运行出错: {e}", file=sys.stderr)
    finally:
        if temp_dir:
            temp_dir.cleanup()
            print("\n已成功清理自测产生的临时文件。")

if __name__ == "__main__":
    main()
