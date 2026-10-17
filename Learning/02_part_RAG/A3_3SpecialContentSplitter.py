import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 导入共享工具
sys.path.append(str(Path(__file__).parent))
from RAG_utils import CustomZhipuEmbeddings, create_production_document

"""
RAG 切分策略 3: 递归字符切分 (Recursive Character Splitting)

解决痛点：
  - 如果只用最简陋的单字符切分器（按字数硬切），会导致一句话或段落在中间位置（如“的”字后面）被斩断，
    极大破坏语义完整性。
    
原理：
  - 使用 `RecursiveCharacterTextSplitter` 递归地尝试按 separators 中的分隔符进行切分，
    优先级为：`\n\n`（段落） ➡️ `\n`（换行） ➡️ 中文常用标点 `。` `！` `？` `，` ➡️ 空格 ➡️ 最后一个个字符。
  - 它会尽可能保证在不超出 chunk_size 的前提下，让切分结果落在最合理的语义停顿点上。
"""

def main():
    load_dotenv()

    print("==============================================")
    print("      RAG 切分策略 3 - 递归字符切分 (A3_3)    ")
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

    # 2. 实例化递归字符分块器
    # 采用中文标点作为优先级分隔符，保障句意连贯性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )

    # 3. 对文本执行切分
    split_texts = text_splitter.split_text(doc_content)
    total_chunks = len(split_texts)
    print(f"📂 递归字符切块拆分完成：共生成 Chunks {total_chunks} 个。")

    # 4. 统一构造生产级 Document (对齐 A1 元数据规范)
    final_documents = []
    for idx, text in enumerate(split_texts):
        doc = create_production_document(
            content=text,
            source_path=file_path,
            chunk_index=idx,
            total_chunks=total_chunks,
            additional_info={
                "chunk_type": "standard_recursive"
            }
        )
        final_documents.append(doc)

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

    print(f"📦 正在将 {len(final_documents)} 个递归字符块写入 Chroma 向量库...")
    vector_store = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        persist_directory=str(chroma_db_dir.resolve())
    )
    print("💾 向量库持久化成功！")

    # 6. 基础 RAG 语义检索测试
    print("\n==============================================")
    print("             RAG 递归分块检索测试             ")
    print("==============================================")
    
    query = input("请输入您想检索的问题 (默认: 'LangGraph'): ").strip()
    if not query:
        query = "LangGraph"

    print(f"\n🔍 正在检索最相似的分块 (top-2)...")
    results = vector_store.similarity_search_with_relevance_scores(query, k=2)

    for i, (doc, score) in enumerate(results):
        print(f"\n🎯 [检索命中 {i+1}] (得分: {score:.4f})")
        print(f"   ├─ 片段索引 (Chunk Index): {doc.metadata.get('chunk_index')} / {doc.metadata.get('total_chunks')}")
        print(f"   ├─ 内容预览:")
        print(f"   │  {doc.page_content[:250].strip()}...")
        print(f"   └─ 全套元数据: {doc.metadata}")
        print("-" * 50)


if __name__ == "__main__":
    main()
