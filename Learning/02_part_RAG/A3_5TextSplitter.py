import os
import sys
import uuid
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 导入共享工具
sys.path.append(str(Path(__file__).parent))
from RAG_utils import CustomZhipuEmbeddings, create_production_document

# 尝试导入 A2_DocumentCleaner 里的生产级清洗器
try:
    from A2_DocumentCleaner import RAGDocumentCleaner
    HAS_CLEANER = True
except ImportError:
    HAS_CLEANER = False

"""
RAG 切分策略 5: 组合切分策略与写入 Chroma (Combined Strategy & Chroma Ingestion)

本脚本是整个 RAG Ingestion 数据管道的总入口和集大成者：
  - 核心流程：加载原始 Markdown 物理文件 ➡️ 数据清洗 (A2) ➡️ 组合策略切块 (A3_5) ➡️ 向量化与持久化 ➡️ 智能父子检索。
  
组合切分策略：
  1. 【文档大纲结构感知】：先用 `MarkdownHeaderTextSplitter` 解析文档大纲，生成带标题元数据的大块。
  2. 【父子层级切分】：
     - 对于每个标题块，先切出“父块 (Parent Chunks，如 800 字)”，并继承其所属的 H1, H2, H3 标题属性。
     - 再将每个父块细切为“子块 (Child Chunks，如 200 字)”。
     - 子块中绑定 `parent_id`、`parent_content` 且保留大纲标题，满足高精度的 Metadata Filtering。
  3. 【统一元数据】：所有子块通过工厂函数严格对齐 A1_加载器.py 字典格式，并持久化存入 Chroma 中。
"""

def main():
    load_dotenv()

    print("==============================================")
    print("      RAG 切分策略 5 - 组合切块与向量持久化   ")
    print("==============================================")

    # 1. 交互式获取需要处理的物理 Markdown 文件路径
    user_input = input("请输入待处理的 Markdown 物理文件路径 (默认使用 test_pdf.md): ").strip()
    if not user_input:
        file_path = Path(__file__).parent / "test_pdf.md"
    else:
        file_path = Path(user_input.strip('"'))

    if not file_path.exists():
        print(f"❌ 错误：找不到指定的物理文件: {file_path.resolve()}")
        sys.exit(1)

    print(f"📄 成功加载物理文件: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. 步骤一：调用 A2 清洗器进行深度数据清洗（去噪、标准化、内容词纠错）
    cleaned_text = raw_text
    if HAS_CLEANER:
        print("🧼 正在检测到生产级清洗器，启动数据去噪与标准化流程...")
        cleaner = RAGDocumentCleaner()
        cleaned_text = cleaner.clean_single_text(raw_text)
        print(f"✨ 数据清洗完成！文本杂质剔除，字数缩减: {len(raw_text)} -> {len(cleaned_text)}")
    else:
        print("⚠️ 未找到 A2_DocumentCleaner 清洗器，跳过清洗直接切块。")

    # 3. 步骤二：采用文档大纲结构进行一级切分 (提取 Header 层级)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    structure_blocks = markdown_splitter.split_text(cleaned_text)
    print(f"🌲 文档大纲解析完毕，初步拆解为 {len(structure_blocks)} 个标题结构快。")

    # 4. 步骤三：结合父子块结构对每个大纲块进行二级切分
    # 实例化父块分块器
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    # 实例化子块分块器
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )

    all_child_documents = []

    for block in structure_blocks:
        # 提取当前大纲块自带的标题元数据
        block_headers = block.metadata
        
        # 将大纲块切成较大的父块
        parent_chunks = parent_splitter.split_text(block.page_content)
        
        for p_idx, p_text in enumerate(parent_chunks):
            # 生成父块的唯一 ID
            parent_id = str(uuid.uuid4())
            
            # 再把每个父块切细为子块
            child_chunks = child_splitter.split_text(p_text)
            total_children = len(child_chunks)
            
            for c_idx, c_text in enumerate(child_chunks):
                # 合并标题信息、父块绑定信息和子块属性
                additional_info = block_headers.copy()
                additional_info.update({
                    "parent_id": parent_id,
                    "parent_content": p_text,
                    "chunk_type": "combined_child"
                })
                
                # 统一构造生产级统一规范 Document
                child_doc = create_production_document(
                    content=c_text,
                    source_path=file_path,
                    chunk_index=c_idx,
                    total_chunks=total_children,
                    additional_info=additional_info
                )
                all_child_documents.append(child_doc)

    print(f"⚡ 组合切分完成：共计提取出 {len(all_child_documents)} 个包含大纲和父子链接的子块 Document。")

    # 5. 持久化存入本地 Chroma 向量库
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

    print(f"📦 正在将组合子块批量写入本地 Chroma 向量库...")
    vector_store = Chroma.from_documents(
        documents=all_child_documents,
        embedding=embeddings,
        persist_directory=str(chroma_db_dir.resolve())
    )
    print("💾 向量库持久化成功！数据落盘无误。")

    # 6. 综合过滤与父子检索效果测试
    print("\n==============================================")
    print("          RAG 组合策略多维过滤与检索测试        ")
    print("==============================================")

    # 提取可用的二级标题
    known_headers = set()
    for d in all_child_documents:
        h2 = d.metadata.get("Header 2")
        if h2:
            known_headers.add(h2)

    print("已缓存的二级标题过滤分类：")
    for h2 in known_headers:
        print(f"  - {h2}")

    filter_h2 = input("\n请指定二级标题过滤器 (直接回车跳过进行全局检索): ").strip()
    query = input("请输入您想检索的问题 (默认: 'State'): ").strip()
    if not query:
        query = "State"

    # 构建过滤词
    search_filter = {"Header 2": filter_h2} if filter_h2 else None
    
    print(f"\n🔍 正在执行组合检索...")
    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=2,
        filter=search_filter
    )

    if not results:
        print("⚠️ 未检索到匹配当前过滤和相似度条件的文档块。")
    for i, (doc, score) in enumerate(results):
        print(f"\n🎯 [检索命中结果 {i+1}] (向量相关性评分: {score:.4f})")
        print(f"   ├─ 所属标题大纲: H1: {doc.metadata.get('Header 1', '无')} -> H2: {doc.metadata.get('Header 2', '无')}")
        print(f"   ├─ 检索定位子块 (Child Chunk) 预览:")
        print(f"   │  {doc.page_content.strip()}")
        print(f"   ├─ 关联父块 (Parent Chunk) 完整语境还原:")
        print(f"   │  {doc.metadata.get('parent_content').strip()[:350]}...")
        print(f"   └─ 100% 对齐元数据字典: {doc.metadata}")
        print("-" * 50)


if __name__ == "__main__":
    main()
