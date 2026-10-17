import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 导入共享工具
sys.path.append(str(Path(__file__).parent))
from RAG_utils import CustomZhipuEmbeddings, create_production_document

"""
RAG 切分策略 2: 文档结构感知切分 (Document Structure-aware Splitting)

解决痛点：
  - 传统的分块器（如字符切分）在切分时完全无视文章的标题大纲结构，极易在标题和段落之间拦腰斩断，
    导致切碎的小块不仅在字面上支离破碎，还会完全丢失所处的文档主题层级（例如它是在介绍概念还是在贴代码？）。
    
原理：
  1. 首先利用 `MarkdownHeaderTextSplitter` 解析文档的大纲树，根据各级标题（#, ##, ###）将文本天然分割。
  2. 它会自动将当前文本块所归属的所有上层标题层级（例如 Header 1: 基础, Header 2: 安装）写入元数据（Metadata）。
  3. 进一步使用字符切分器对超长块进行细分，但同时继承上级标题元数据。
  4. 这样在检索时，我们可以直接利用向量库的 Metadata Filtering 进行多级过滤（如只在“第3章”的范围里检索）。
"""

def main():
    load_dotenv()

    print("==============================================")
    print("      RAG 切分策略 2 - 文档结构感知切分 (A3_2) ")
    print("==============================================")

    # 1. 获取输入 Markdown 文件路径
    user_input = input("请输入要切分的 Markdown 文件路径 (默认使用 test_pdf.md): ").strip()
    if not user_input:
        file_path = Path(__file__).parent / "test_pdf.md"
    else:
        file_path = Path(user_input.strip('"'))

    if not file_path.exists():
        print(f"❌ 错误：未找到指定的 Markdown 文件: {file_path.resolve()}")
        sys.exit(1)

    print(f"📄 成功加载物理文件: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        doc_content = f.read()

    # 2. 定义需要切分的标题层级，并定义在元数据中提取的 Key 名
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 3. 执行大纲结构第一次切分
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # 保留正文中的标题文字，方便大模型和向量模型读取
    )
    md_header_docs = markdown_splitter.split_text(doc_content)
    print(f"📊 Markdown 结构树解析完成：初步拆成 {len(md_header_docs)} 个标题语义块。")

    # 4. 二级细化切分（如果某个标题下的字数很多，利用 RecursiveCharacterTextSplitter 继续切分，并继承标题元数据）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        separators=["\n\n", "\n", " ", ""]
    )

    final_documents = []

    for idx, header_doc in enumerate(md_header_docs):
        # 如果当前标题语义块很小，直接作为独立 Document
        if len(header_doc.page_content) <= 400:
            doc = create_production_document(
                content=header_doc.page_content,
                source_path=file_path,
                chunk_index=0,
                total_chunks=1,
                additional_info=header_doc.metadata  # 携带解析出的 Header 1, Header 2 等标题元数据
            )
            final_documents.append(doc)
        else:
            # 否则，进一步切细
            sub_chunks = text_splitter.split_text(header_doc.page_content)
            total_subs = len(sub_chunks)
            for sub_idx, sub_text in enumerate(sub_chunks):
                doc = create_production_document(
                    content=sub_text,
                    source_path=file_path,
                    chunk_index=sub_idx,
                    total_chunks=total_subs,
                    additional_info=header_doc.metadata  # 每一个子切片都会继承父标题信息
                )
                final_documents.append(doc)

    print(f"⚡ 文档结构感知细化切分完成：最终生成 Chunks 数量: {len(final_documents)}")

    # 5. 向量化并持久化至 Chroma 向量库
    chroma_db_dir = Path(__file__).parent / "chroma_db"
    
    if chroma_db_dir.exists():
        try:
            shutil.rmtree(chroma_db_dir)
        except Exception as e:
            print(f"⚠️ 清理历史向量库失败: {e}")

    api_key = os.getenv("EMBEDDINGS_API_KEY")
    base_url = os.getenv("EMBEDDINGS_BASE_URL")
    model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "embedding-3")

    if not api_key:
        print("❌ 错误：环境变量中未检测到 EMBEDDINGS_API_KEY，无法向量化。")
        sys.exit(1)

    embeddings = CustomZhipuEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model_name
    )

    print(f"📦 正在将 {len(final_documents)} 个感知结构块写入 Chroma 向量库...")
    vector_store = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        persist_directory=str(chroma_db_dir.resolve())
    )
    print("💾 向量库持久化成功！")

    # 6. RAG 元数据过滤检索测试 (Metadata Filtering)
    print("\n==============================================")
    print("         RAG 文档大纲过滤检索测试 (Metadata Filter)   ")
    print("==============================================")

    # 打印向量库中存在的二级标题以供选择
    known_headers = set()
    for d in final_documents:
        h2 = d.metadata.get("Header 2")
        if h2:
            known_headers.add(h2)
    
    print("当前文档解析出的二级标题范围：")
    for h2 in known_headers:
        print(f"  - {h2}")

    filter_h2 = input("\n请输入过滤条件(指定二级标题名称，直接回车跳过): ").strip()
    query = input("请输入您想检索的问题 (默认: 'LangGraph'): ").strip()
    if not query:
        query = "LangGraph"

    # 构建元数据过滤器
    search_filter = {"Header 2": filter_h2} if filter_h2 else None
    
    if search_filter:
        print(f"\n🔍 正在检索 (仅在二级标题为 '{filter_h2}' 的范围里查找)...")
    else:
        print(f"\n🔍 正在全局无过滤检索...")

    results = vector_store.similarity_search_with_relevance_scores(
        query, 
        k=2, 
        filter=search_filter
    )

    if not results:
        print("⚠️ 未匹配到符合该过滤条件的文档块。")
    for i, (doc, score) in enumerate(results):
        print(f"\n🎯 [检索命中 {i+1}] (得分: {score:.4f})")
        print(f"   ├─ 所属标题大纲:")
        print(f"   │  H1: {doc.metadata.get('Header 1', '无')}")
        print(f"   │  H2: {doc.metadata.get('Header 2', '无')}")
        print(f"   │  H3: {doc.metadata.get('Header 3', '无')}")
        print(f"   ├─ 内容预览:")
        print(f"   │  {doc.page_content[:200].strip()}...")
        print(f"   └─ 全套元数据: {doc.metadata}")
        print("-" * 50)


if __name__ == "__main__":
    main()
