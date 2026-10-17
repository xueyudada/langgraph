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

# 尝试载入 PyTorch 和 Transformers，用于本地真·Late Chunking 计算
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_LOCAL_TRANSFORMERS = True
except ImportError:
    HAS_LOCAL_TRANSFORMERS = False

"""
RAG 切分策略 4: 延迟分块 (Late Chunking)

解决痛点：
  - 传统的切块方式是“先切块，再单独向量化（Embedding）”。当子块切得极细时，它就会孤立，
    导致 Embedding 模型无法在此子块中编码出整篇文章的全局主旨（例如不知道它是关于哪个项目或背景下的技术细分）。
    
原理 (Late Chunking)：
  - 采用“先整体编码，再按边界池化”的现代设计。
  1. 【整体编码】：先将**整篇长文档**输入到支持长上下文的 Embedding 模型中，获取这篇长文档所有 Token 的 Embedding 序列矩阵。
     这样每一个 Token 的向量天然通过 Attention 机制融合了整篇文章的全部上下文信息。
  2. 【按边界池化】：接着按照普通规则（如递归字符）把文档切块，并找到每一个分块在原文中对应的 Token 索引区间。
  3. 【生成向量】：对每个分块对应的 Token Embedding 子序列做 Mean Pooling（均值池化），直接作为该 Chunk 的最终 Embedding。
  4. 【优势】：分块的向量不仅能保留其自身的局部细节，还天然锁定了整篇文章的全局语境。
"""

class LateChunkingProcessor:
    """延迟分块处理器，包含本地 Transformer 真实计算和 API 模拟双重链路"""
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.initialized = False

    def init_local_model(self):
        """初始化本地长上下文 Embedding 模型 (如 nomic-embed 等)"""
        if HAS_LOCAL_TRANSFORMERS and not self.initialized:
            try:
                print(f"📥 正在尝试本地载入长向量模型 (可能需要下载): {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # 开启 trust_remote_code 以支持非标模型架构
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                self.initialized = True
                print("✅ 本地模型载入成功！将执行 Token-level 真实池化 Late Chunking。")
            except Exception as e:
                print(f"⚠️ 本地模型载入失败: {e}，将降级为 API 上下文模拟模式。")

    def process_late_chunking_local(self, text: str, chunks_text: list[str]) -> list[list[float]]:
        """本地执行真正的 Token-level Late Chunking"""
        # 对整篇文档进行 Token 编码
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取最后一层隐藏状态 (Last Hidden State)作为 Token Embeddings
            token_embeddings = outputs.last_hidden_state[0] # 大小: [seq_len, hidden_dim]

        embeddings = []
        
        # 依次定位每个分块在整篇文本 Token 序列中的起止位置
        # 这里用简易的字符到 Token 映射机制来定位边界
        input_ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids]
        
        current_token_idx = 0
        for chunk in chunks_text:
            # 找到当前 chunk 在 token 列表中的匹配区间
            chunk_tokens = self.tokenizer.tokenize(chunk)
            chunk_len = len(chunk_tokens)
            
            start_idx = current_token_idx
            # 粗略定位边界并防止越界
            end_idx = min(start_idx + chunk_len, len(token_embeddings))
            
            # 提取区间内的 token embeddings 并做 Mean Pooling (均值池化)
            chunk_token_embs = token_embeddings[start_idx:end_idx]
            if len(chunk_token_embs) > 0:
                chunk_emb = chunk_token_embs.mean(dim=0).tolist()
            else:
                # 越界兜底
                chunk_emb = token_embeddings.mean(dim=0).tolist()
                
            embeddings.append(chunk_emb)
            current_token_idx = end_idx

        return embeddings

    def process_late_chunking_api_simulated(
        self, 
        text: str, 
        chunks_text: list[str], 
        emb_api: CustomZhipuEmbeddings
    ) -> list[list[float]]:
        """
        API 降级模拟：当本地无显卡/未装 PyTorch 时运行。
        通过提取并强行在每个子块前注入文档前言或全局概要，作为虚拟上下文，从而让 API 也能进行带上下文的向量计算。
        """
        # 提取全局摘要或文档前 80 个字符作为全局前言（Global Preface）
        doc_preface = text.strip()[:80].replace("\n", " ")
        
        simulated_texts = []
        for chunk in chunks_text:
            # 将全局摘要以不可见或显式方式拼装到子块中
            simulated_text = f"[文档主旨: {doc_preface}] {chunk}"
            simulated_texts.append(simulated_text)
            
        # 通过智谱 API 批量获取嵌入向量
        return emb_api.embed_documents(simulated_texts)


def main():
    load_dotenv()

    print("==============================================")
    print("      RAG 切分策略 4 - 延迟分块 Late Chunking (A3_4) ")
    print("==============================================")

    # 1. 物理文件输入
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

    # 2. 对文档进行细粒度切块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    chunks_text = text_splitter.split_text(doc_content)
    total_chunks = len(chunks_text)
    print(f"📂 物理文档分块切分完成：共计子块 {total_chunks} 个。")

    # 3. 初始化 Embedding（优先本地长向量，无则 API 降级模拟）
    api_key = os.getenv("EMBEDDINGS_API_KEY")
    base_url = os.getenv("EMBEDDINGS_BASE_URL")
    model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "embedding-3")

    if not api_key:
        print("❌ 错误：环境变量中未检测到 EMBEDDINGS_API_KEY，无法向量化。")
        sys.exit(1)

    emb_api = CustomZhipuEmbeddings(api_key=api_key, base_url=base_url, model=model_name)
    processor = LateChunkingProcessor()

    # 如果系统配置了 torch 和 transformers，尝试载入本地模型进行真实的 Late Chunking 池化
    # 否则直接走 API-based 上下文增强模拟
    use_local_transformers = False
    if HAS_LOCAL_TRANSFORMERS:
        # 只有在用户本地有 GPU 等情况可以尝试（此处为健壮逻辑）
        # 默认不下载几个G的超大模型，而是由用户按需启用，这里我们做自动判别与询问
        choice = input("检测到本地支持 Transformers，是否下载并载入本地模型执行【真·Late Chunking】(若选 n 则执行 API 语义模拟)？(y/n，默认 n): ").strip().lower()
        if choice == 'y':
            processor.init_local_model()
            if processor.initialized:
                use_local_transformers = True

    # 4. 执行延迟分块计算
    print("⏳ 正在计算包含全局上下文的延迟分块向量...")
    if use_local_transformers:
        calculated_embeddings = processor.process_late_chunking_local(doc_content, chunks_text)
    else:
        print("ℹ️ 执行 API-based 上下文增强模拟（已拼接整篇文档主旨前缀进行二次 Embedding）。")
        calculated_embeddings = processor.process_late_chunking_api_simulated(doc_content, chunks_text, emb_api)

    # 5. 构造统一规范 Document
    documents = []
    for idx, text in enumerate(chunks_text):
        doc = create_production_document(
            content=text,
            source_path=file_path,
            chunk_index=idx,
            total_chunks=total_chunks,
            additional_info={
                "chunk_type": "late_chunking_simulated" if not use_local_transformers else "late_chunking_local",
                "global_preface": doc_content.strip()[:60]
            }
        )
        documents.append(doc)

    # 6. 持久化至 Chroma
    # 注意：Chroma 在添加数据时，可以通过 add_documents 传递我们刚才自主计算并池化出的 embeddings 向量
    chroma_db_dir = Path(__file__).parent / "chroma_db"
    
    if chroma_db_dir.exists():
        try:
            shutil.rmtree(chroma_db_dir)
        except Exception as e:
            print(f"⚠️ 清理历史向量库失败: {e}")

    print(f"📦 正在将已池化的延迟分块写入本地 Chroma 向量库...")
    # 由于是外部计算好的向量，我们先用空的 embedding 初始化 Chroma，再显式调用 add_documents
    vector_store = Chroma(
        persist_directory=str(chroma_db_dir.resolve()),
        embedding_function=emb_api  # 仅做结构定义，实际写入和检索使用我们计算好的向量或查询向量
    )
    
    # 写入带有 Custom Embeddings 的 Documents
    vector_store.add_documents(
        documents=documents,
        embeddings=calculated_embeddings
    )
    print("💾 延迟分块向量数据落盘持久化完成！")

    # 7. RAG 检索验证测试
    print("\n==============================================")
    print("             RAG 延迟分块检索测试             ")
    print("==============================================")
    
    query = input("请输入您想检索的问题 (默认: 'LangGraph'): ").strip()
    if not query:
        query = "LangGraph"

    print(f"\n🔍 正在检索...")
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
