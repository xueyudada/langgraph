import os
import sys
import uuid
import shutil
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 导入共享工具
sys.path.append(str(Path(__file__).parent))
from RAG_utils import CustomZhipuEmbeddings, create_production_document

"""
RAG 切分策略 1: 父子块结构切分 (Parent-Child Splitting)

解决痛点：
  - 块分得太大：检索不精准，容易包含过多无关杂音，且浪费 LLM 的上下文 Token。
  - 块分得太小：检索极精准，但字数过少会导致核心上下文断裂，LLM 无法正确回答。
  
原理：
  在向量库中只存储和检索尺寸较小的“子块 (Child Chunks)”以确保匹配的精准度；
  但当检索命中某个子块时，利用子块元数据中的关联 ID，自动还原并提取其所属的较大“父块 (Parent Chunks)”作为完整上下文提供给大模型。
"""

def split_by_markdown_headers(content: str) -> list[str]:
    """以二级标题（##）或一级标题（#）为边界切分为父块"""
    lines = content.split("\n")
    chunks = []
    current_chunk = []
    for line in lines:
        if line.startswith("## ") or line.startswith("# "):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
        current_chunk.append(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return [c for c in chunks if c]


def split_into_sentence_chunks(text: str, max_size: int = 200) -> list[str]:
    """以标点符号（。？！）为边界分句，并组装成 100~200 字符的子块"""
    # 使用正则按标点符号切分并保留标点
    raw_sentences = re.split(r'(?<=[。！？\n])', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for s in sentences:
        s_len = len(s)
        # 如果当前累积块不为空，且加上这一句超过最大字符限制 200 字符，则结算当前块
        if current_chunk and (current_len + s_len > max_size):
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_len = s_len
        else:
            current_chunk.append(s)
            current_len += s_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


def main():
    load_dotenv()

    print("==============================================")
    print("      RAG 切分策略 1 - 父子块结构切分 (A3_1)  ")
    print("==============================================")

    # 1. 交互式获取待切分的物理 Markdown 文件路径
    user_input = input("请输入要切分的 Markdown 文件路径 (默认使用 test_pdf.md): ").strip()
    if not user_input:
        # 默认寻找同级目录下的 test_pdf.md
        file_path = Path(__file__).parent / "test_pdf.md"
    else:
        file_path = Path(user_input.strip('"'))

    if not file_path.exists():
        print(f"❌ 错误：未找到指定的 Markdown 文件: {file_path.resolve()}")
        sys.exit(1)

    print(f"📄 成功加载物理文件: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        doc_content = f.read()

    # 3. 按照 A1_加载器.py 中的规则，从解析后的 Markdown 文件名还原原始文件名和后缀
    md_name = file_path.name
    pattern = r"^(.*)_(pdf|docx|xlsx|xls|csv|txt|doc|md)\.md$"
    match = re.match(pattern, md_name, re.IGNORECASE)

    if match:
        base_name = match.group(1)
        original_ext = f".{match.group(2).lower()}"
        original_filename = f"{base_name}{original_ext}"
        original_path = file_path.parent / original_filename
    else:
        original_ext = ".md"
        original_path = file_path

    # 执行大纲父块切分与语义分句子块关联逻辑
    parent_texts = split_by_markdown_headers(doc_content)
    print(f"📂 文档切分初步完成：基于二级/一级标题大纲共生成父块 {len(parent_texts)} 个。")

    all_child_documents = []
    
    # 建立父子映射：将每个父块切细为多个子块，并在子块元数据中强绑定父块 ID 和正文
    for p_idx, p_text in enumerate(parent_texts):
        # 为当前父块生成唯一 ID
        parent_id = str(uuid.uuid4())
        
        # 将当前父块文本作为文本源，传入分句切分器进一步碎分
        child_texts = split_into_sentence_chunks(p_text)
        total_children = len(child_texts)
        
        for c_idx, c_text in enumerate(child_texts):
            # 统一结构化元数据：严格对齐 A1_加载器.py 中的元数据工厂
            child_doc = create_production_document(
                content=c_text,
                source_path=original_path,
                chunk_index=c_idx,
                total_chunks=total_children,
                additional_info={
                    "parent_id": parent_id,            # 父块唯一识别 ID
                    "parent_content": p_text,          # 直接把父块文本封装进元数据，方便检索秒回
                    "chunk_type": "child",
                    "parsed_path": str(file_path.resolve()),  # 解析后的 md 文件路径
                    "original_suffix": original_ext,          # 原始文件后缀
                }
            )
            all_child_documents.append(child_doc)

    print(f"⚡ 子块级细化语义切分完成：共产生关联子块 {len(all_child_documents)} 个。")

    # 4. 向量化并持久化至 Chroma 向量库
    chroma_db_dir = Path(__file__).parent / "chroma_db"
    
    # 询问用户是否清空历史数据库，实现增量追加
    reset_choice = input("是否清空历史向量库？(y/n，默认 n，输入 y 则清空重建，输入 n 则增量追加): ").strip().lower()
    if reset_choice == "y" and chroma_db_dir.exists():
        try:
            shutil.rmtree(chroma_db_dir)
            print("🧹 已成功清空历史数据库，准备重建...")
        except Exception as e:
            print(f"❌ 错误：清空历史向量库失败，数据库文件被占用！")
            print(f"   原因: {e}")
            print("👉 请确保已经退出了所有的检索查询进程（如 A3_1FixedSizeSplitter_search.py，在终端输入 'q' 退出），并关闭所有打开了该数据库的 SQLite 查看器或编辑器，然后重试。")
            sys.exit(1)

    # 获取智谱 Embedding (读取环境变量)
    api_key = os.getenv("EMBEDDINGS_API_KEY")
    base_url = os.getenv("EMBEDDINGS_BASE_URL")
    model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "embedding-3")

    if not api_key:
        print("❌ 错误：环境变量中未检测到 EMBEDDINGS_API_KEY，无法向量化。")
        sys.exit(1)

    print(f"🚀 初始化智谱 Embedding 适配器 (模型: {model_name})...")
    embeddings = CustomZhipuEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model_name
    )

    print(f"📦 正在初始化/加载本地 Chroma 向量库...")
    vector_store = Chroma(
        persist_directory=str(chroma_db_dir.resolve()),
        embedding_function=embeddings
    )
    
    print(f"📥 正在将 {len(all_child_documents)} 个子块写入/追加至向量库...")
    vector_store.add_documents(all_child_documents)
    print("💾 向量库写入/持久化成功！")

    # 5. 完成提示
    print("\n==============================================")
    print("✨ 向量化入库已全部完成！")
    print("👉 请运行 A3_1FixedSizeSplitter_search.py 进行独立检索查询测试。")
    print("==============================================")


if __name__ == "__main__":
    main()
