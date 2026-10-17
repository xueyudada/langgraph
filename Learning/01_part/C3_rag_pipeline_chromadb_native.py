import os
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 区别对比说明
"""
================================================================================
💡 langchain-chroma 与原生 chromadb 的主要区别：
================================================================================
1. 抽象层级与数据类型：
   - langchain-chroma：
     * 是 LangChain 生态的一部分，继承自 LangChain 的 VectorStore 基类。
     * 直接接收 LangChain 的 `Document` 对象列表，便于在 LangChain 管道（LCEL）或 Agent 中作为 Retriever 直接传递使用。
   - 原生 chromadb：
     * 是底层的独立向量数据库 SDK。
     * 接收基础的 Python 数据类型（List[str]、List[dict]、List[str] 等），例如 documents、metadatas、ids 和 embeddings 列表必须分开传递。
     * 适合不使用 LangChain、追求轻量级、需要更精细地控制数据库 Collection 或者需要与其他非 LangChain 框架（如 LlamaIndex 等）配合的场景。

2. 向量计算 (Embedding) 机制：
   - langchain-chroma：
     * 要求传入 LangChain 的 `Embeddings` 实例。在增、删、改、查时，包装器自动在后台调用 Embedding 类将文本转化为向量。
   - 原生 chromadb：
     * 更加灵活。你可以选择：
       a) 传入预先计算好的向量列表（embeddings 参数），这意味着你可以用任何库（原生 HuggingFace、PyTorch 等）算好向量直接存入。
       b) 注册 Chroma 原生的 EmbeddingFunction（如 chromadb.utils.embedding_functions 里的各类包装器），让 Chroma 自动计算。
       c) 如果不传，Chroma 可能会在后台使用它默认的文本表示模型。

3. 持久化存储与加载：
   - langchain-chroma：
     * 初始化时通过 `persist_directory` 声明持久化路径，在后台自动打理 SQLite 的读写。
   - 原生 chromadb：
     * 采用 Client - Collection 模型。
     * 运行 `chromadb.PersistentClient(path)` 实例化持久化客户端，所有的 Collections 会被保存在该路径下。
     * 写入和读取 Collection 完全基于该 Client 进行。

4. 接口功能：
   - langchain-chroma：
     * 提供 `.as_retriever()` 方法，完美支持 LCEL 的 `|` 管道运算。
   - 原生 chromadb：
     * 提供 `.query()`、`.get()`、`.peek()`、`.update()`、`.delete()` 等底层数据库 CRUD 接口，支持底层元数据过滤（Metadata Filtering）。
================================================================================
"""

# 1. 准备演示用的文本
DEMO_DOC_CONTENT = """LangChain 是一个功能强大的框架，旨在帮助开发人员构建由大型语言模型 (LLM) 驱动的应用程序。
它提供了诸如 Model I/O、Retrieval (RAG)、Chains 和 Agents 等核心模块。
其中，LCEL (LangChain Expression Language) 是用于声明式链接 LangChain 组件的语言。

LangGraph 是建立在 LangChain 之上的扩展库。
它专门用于构建支持“循环” (Loops) 和“状态管理” (State Management) 的复杂多智能体 (Multi-Agent) 系统。
如果你的 AI 应用需要复杂的决策流或反馈环，LangGraph 是更合适的选择。

Model Context Protocol (MCP) 是近期推出的一项开放标准。
它允许开发者构建标准的协议接口，使 AI 智能体 (Agents) 能够安全、一致地接入各种外部工具和数据源。"""

# 加载环境变量
load_dotenv()
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")
def main():
    print("🚀 === 原生 chromadb 持久化演示与区别说明 === 🚀\n")
    
    # ---- 步骤 1: 文本切分 ----
    print("--- 步骤 1: 切分文本 ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
    # 原生接口不需要 Document 对象，因此我们直接获取切分后的纯字符串列表
    chunks = splitter.split_text(DEMO_DOC_CONTENT)
    print(f"切分成功，共 {len(chunks)} 个 chunks。")
    for i, chunk in enumerate(chunks):
        print(f"  [Chunk {i}]: {repr(chunk)}")
    print()
    
    # ---- 步骤 2: 计算 Embeddings (向量化) ----
    print("--- 步骤 2: 手动使用 Embedding 模型生成向量 ---")
    # 我们这里依然借用 OpenAIEmbeddings 计算，但原生 chromadb 支持你传入直接算好的 list[float]
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=EMBEDDINGS_API_KEY,
        openai_api_base=EMBEDDINGS_BASE_URL,
        model=EMBEDDINGS_MODEL_NAME  # 如果需要可在此指定嵌入模型
    )
    # 手动计算所有 chunk 的向量
    embeddings_list = embeddings_model.embed_documents(chunks)
    print(f"成功为所有 chunk 计算了 Embeddings。向量维度: {len(embeddings_list[0])}\n")
    
    # ---- 步骤 3: 初始化原生 chromadb 持久化客户端 ----
    print("--- 步骤 3: 初始化原生 chromadb 持久化客户端 ---")
    persist_path = "./chroma_db_native"
    client = chromadb.PersistentClient(path=persist_path)
    print(f"持久化客户端初始化成功，存储路径: {persist_path}\n")
    
    # ---- 步骤 4: 获取或创建 Collection 并存入数据 ----
    print("--- 步骤 4: 创建/获取 Collection 并存入数据 ---")
    # 注意：原生 chromadb 一切以 Collection 为中心
    collection = client.get_or_create_collection(name="demo_collection")
    
    # 如果 collection 里已经有数据了，我们可以先清空以便重复运行演示
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection 已经存在并且包含 {existing_count} 条数据，正在清空以便重新演示...")
        # 获取所有已有的 id 并删除
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            
    # 原生添加接口：必须手动传入 ids。
    ids = [f"id_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "demo_doc", "index": i} for i in range(len(chunks))]
    
    # 将文本、向量、元数据、ID 写入 Collection
    # 写入后客户端会自动执行持久化写入底层 sqlite 和 parquet 存储
    collection.add(
        documents=chunks,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )
    print(f"数据成功存入原生 Collection，当前总数: {collection.count()}\n")
    
    # ---- 步骤 5: 执行检索 (Query) ----
    print("--- 步骤 5: 原生检索 (Query) ---")
    question = "请问 LangGraph 有什么用？它的核心特点是什么？"
    print(f"问题: '{question}'")
    
    # 原生查询需要先把问题的文本算成向量
    query_embeddings = embeddings_model.embed_query(question)
    
    # 查询相似度最高的 2 条
    results = collection.query(
        query_embeddings=[query_embeddings], # 可以一次传入多个查询向量
        n_results=2
    )
    
    # 打印召回结果
    print("\n🔍 [原生 chromadb 检索召回的文本片段与得分]:")
    # results 是一个字典，包含 documents, distances, metadatas, ids 等 key
    # 它们的结构都是嵌套的 List，因为支持批量查询。由于我们只传了一个 query，我们取 [0]
    matched_docs = results["documents"][0]
    matched_distances = results["distances"][0] # 距离越小越相似 (通常是 L2 距离或余弦距离的变体)
    matched_metadatas = results["metadatas"][0]
    
    for i in range(len(matched_docs)):
        print(f"  召回片段 {i+1}:")
        print(f"    内容: {matched_docs[i]}")
        print(f"    元数据: {matched_metadatas[i]}")
        print(f"    距离/得分: {matched_distances[i]}")

if __name__ == "__main__":
    main()
