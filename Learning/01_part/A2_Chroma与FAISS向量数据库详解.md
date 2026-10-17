# Chroma & FAISS 向量数据库详解

> 适用栈：LangChain 1.x + `langchain-openai` / `langchain-community`
> 前置：理解 Embedding 概念（文本 → 向量），了解 RAG 基本流程
> 目标：能独立安装、使用、对比 Chroma 和 FAISS，在 RAG 链路中做出合理选型

---

## 目录

- [一、为什么需要向量数据库](#一为什么需要向量数据库)
- [二、FAISS — Facebook AI Similarity Search](#二faiss--facebook-ai-similarity-search)
- [三、Chroma — AI-Native 向量数据库](#三chroma--ai-native-向量数据库)
- [四、FAISS vs Chroma 对比总览](#四faiss-vs-chroma-对比总览)
- [五、安装与环境配置](#五安装与环境配置)
- [六、FAISS 实战教学](#六faiss-实战教学)
- [七、Chroma 实战教学](#七chroma-实战教学)
- [八、进阶话题](#八进阶话题)
- [九、选型决策指南](#九选型决策指南)

---

## 一、为什么需要向量数据库

### 1.1 核心问题

LLM 本身无法感知训练截止日期之后的新知识，也无法访问私域数据。RAG（Retrieval-Augmented Generation）的解决思路是：

```
用户提问 → 将问题向量化 → 在向量数据库中检索最相关的文档片段 → 将片段+问题一起发给 LLM → 生成答案
```

其中**向量数据库**承担了"语义搜索引擎"的角色。

### 1.2 向量数据库的核心能力

```
┌──────────────────────────────────────────────────────┐
│                    向量数据库                          │
│                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ 存储向量  │ ←→ │ 相似度检索    │ ←→ │ 元数据过滤   │ │
│  │ (Insert)  │    │ (ANN Search)  │    │ (Metadata)  │ │
│  └──────────┘    └──────────────┘    └─────────────┘ │
│                                                      │
│  关键指标：精度(Recall) / 速度(QPS) / 内存占用 / 持久化  │
└──────────────────────────────────────────────────────┘
```

- **ANN (Approximate Nearest Neighbor)**：近似最近邻搜索，在速度和精度之间取平衡
- **相似度度量**：余弦相似度（Cosine）、欧氏距离（L2）、内积（IP）
- **Top-K 检索**：返回最相似的 K 个结果

---

## 二、FAISS — Facebook AI Similarity Search

### 2.1 什么是 FAISS

FAISS 是 Meta（原 Facebook）开源的高效向量相似度搜索库，专攻**大规模向量检索**。

| 维度 | 说明 |
|---|---|
| 开发者 | Meta（Facebook AI Research） |
| 定位 | **向量索引库**（不是完整的数据库） |
| 语言 | C++ 核心 + Python 绑定 |
| GitHub | <https://github.com/facebookresearch/faiss> |
| 核心优势 | 极致的检索速度和内存效率，支持 GPU 加速 |
| 适用场景 | 百万到十亿级向量检索，离线/嵌入式场景优先 |

### 2.2 核心概念

```
FAISS 的核心抽象：Index（索引）

Index  = 存储结构 + 检索算法 + (可选)压缩/量化

┌────────────────────────────────────────────────────┐
│                 FAISS Index 类型                    │
│                                                    │
│  精确搜索 IndexFlatL2 / IndexFlatIP                │
│  ├─ 暴力计算，100% 精度，数据量大时太慢              │
│                                                    │
│  倒排索引 IndexIVFFlat                              │
│  ├─ 先对向量做 K-Means 聚类                         │
│  ├─ 搜索时只在最近的几个聚类中心内找                  │
│  └─ 需要训练阶段                                   │
│                                                    │
│  乘积量化 IndexIVFPQ                                │
│  ├─ 在 IVFFlat 基础上对向量做有损压缩                │
│  ├─ 大幅减少内存占用，代价是少量精度损失              │
│  └─ 十亿级向量的标配                               │
│                                                    │
│  HNSW IndexHNSWFlat                                 │
│  ├─ 基于图结构，搜索质量高，无需训练                  │
│  └─ 内存占用较大，构建速度较慢                       │
└────────────────────────────────────────────────────┘
```

### 2.3 FAISS 不是"数据库"

**关键理解**：FAISS 是一个索引库，而不是数据库。它：

- ✅ 可以做：存向量、搜向量、GPU 加速、压缩
- ❌ 不能做（原生）：持久化到磁盘、元数据过滤、增删改、分布式、并发安全
- 🔧 解决方案：通过 `IndexIDMap` 映射回原始数据，或配合 LangChain 的 `FAISS` wrapper 使用

---

## 三、Chroma — AI-Native 向量数据库

### 3.1 什么是 Chroma

Chroma 是为 AI 应用设计的**开源嵌入式向量数据库**，提供完整的数据库体验。

| 维度 | 说明 |
|---|---|
| 开发者 | Chroma 团队 |
| 定位 | **完整的向量数据库**（类似传统数据库的体验） |
| 语言 | Python / TypeScript 客户端 + Rust 后端 |
| GitHub | <https://github.com/chroma-core/chroma> |
| 核心优势 | 开箱即用、元数据过滤强、持久化内置、API 友好 |
| 适用场景 | 原型到中等规模生产、需要复杂元数据过滤的场景 |

### 3.2 核心概念

```
Chroma 的数据模型：

Collection（集合）
├── Document 1
│   ├── id: "doc_1"
│   ├── embedding: [0.1, -0.3, ...]   ← 向量
│   ├── document: "原始文本..."         ← 原文
│   └── metadata: {"source": "web"}   ← 元数据（可做过滤）
├── Document 2
│   ├── id: "doc_2"
│   └── ...
└── ...

┌──────────────────────────────────────────────────────┐
│               Chroma 核心特性                         │
│                                                      │
│  ✅ 自动持久化到磁盘（无需手动 save）                   │
│  ✅ 富元数据过滤（where={"source": "pdf"}）             │
│  ✅ 内置 Embedding 函数（也可自定义）                   │
│  ✅ Client/Server 架构（支持远程部署）                  │
│  ✅ 文档自动分块和向量化                               │
│  ✅ 多种相似度度量（cosine / l2 / ip）                  │
└──────────────────────────────────────────────────────┘
```

### 3.3 Chroma 的两种运行模式

| 模式 | 说明 | 适合场景 |
|---|---|---|
| **内存模式 (Ephemeral)** | 数据存内存，进程结束即消失 | 快速原型、测试 |
| **持久化模式 (Persistent)** | 数据存磁盘，重启不丢失 | 实际项目、生产 |

---

## 四、FAISS vs Chroma 对比总览

```
┌────────────────┬──────────────────────┬──────────────────────┐
│      维度       │        FAISS         │       Chroma          │
├────────────────┼──────────────────────┼──────────────────────┤
│ 本质            │ 向量索引库            │ 完整向量数据库         │
│ 持久化          │ 需手动 save/load      │ 自动持久化            │
│ 元数据过滤       │ 不原生支持            │ 原生支持（where 语法） │
│ 增删改          │ 受限（需重建索引）     │ 原生支持              │
│ GPU 加速        │ ✅ 强项               │ ❌ 不支持             │
│ 检索速度        │ 极快（尤其 GPU）       │ 快（CPU）             │
│ 大数据量(>10M)  │ ✅ 优势               │ ⚠️ 吃力               │
│ 部署复杂度       │ 低（只是一个库）       │ 中（Client/Server）    │
│ 内存占用        │ 低（有量化压缩）       │ 中（无量化）           │
│ API 友好度      │ 底层 C++ 风格         │ 高级 Pythonic 风格     │
│ 生态集成        │ 学术/研究界标配         │ LangChain/LlamaIndex  │
│ 典型数据量       │ 1M ~ 1B+             │ 1K ~ 1M              │
└────────────────┴──────────────────────┴──────────────────────┘
```

**一句话总结**：
- **FAISS**：当你需要极快、极大规模地检索向量，并且愿意手动管理外围功能时用
- **Chroma**：当你需要一个开箱即用的完整数据库，有元数据过滤需求，规模在百万级以内时用

---

## 五、安装与环境配置

### 5.1 FAISS 安装

```bash
# CPU 版本（最常用，兼容性最好）
pip install faiss-cpu

# GPU 版本（需要 CUDA 环境）
pip install faiss-gpu

# 如果与 LangChain 集成使用，同时安装
pip install langchain-community
```

**Windows 用户注意**：
- `faiss-cpu` 在 Windows 上可直接 pip 安装
- `faiss-gpu` 在 Windows 上配置较复杂，建议用 CPU 版或 WSL2 + Linux

### 5.2 Chroma 安装

```bash
# 基础安装（含内存模式和持久化模式）
pip install chromadb

# 如果与 LangChain 集成使用
pip install langchain-chroma
# 或通过 langchain-community（旧版本）
pip install langchain-community
```

### 5.3 版本兼容性速查

```bash
# 推荐版本组合
pip install chromadb>=0.4.0
pip install faiss-cpu>=1.7.4
pip install langchain-core>=0.3.0
pip install langchain-community>=0.3.0
pip install langchain-chroma>=0.1.0
```

---

## 六、FAISS 实战教学

### 6.1 原生 FAISS 使用

```python
"""FAISS 原生 API 完整演示"""
import numpy as np
import faiss

# ============================================
# Step 1: 准备数据
# ============================================
# 假设我们已有 1000 个 128 维的向量
dimension = 128
num_vectors = 1000
np.random.seed(42)
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# ============================================
# Step 2: 创建索引 — Flat（暴力检索，100% 精度）
# ============================================
index = faiss.IndexFlatL2(dimension)  # L2 欧氏距离
print(f"索引类型: {type(index).__name__}")
print(f"索引维度: {index.d}")

# 添加向量
index.add(vectors)
print(f"索引中向量数: {index.ntotal}")

# ============================================
# Step 3: 检索 — 找出 Top-K 最相似的向量
# ============================================
query = np.random.random((5, dimension)).astype('float32')  # 5 个查询
k = 3  # 返回 Top-3

distances, indices = index.search(query, k)
print(f"\n检索结果:")
print(f"  - 距离矩阵 shape: {distances.shape}")   # (5, 3)
print(f"  - 索引矩阵 shape: {indices.shape}")      # (5, 3)
print(f"  - 第 0 个查询的最近 3 个邻居索引: {indices[0]}")
print(f"  - 对应距离: {distances[0]}")

# ============================================
# Step 4: 持久化 — 保存和加载
# ============================================
faiss.write_index(index, "my_index.faiss")
print("\n索引已保存到 my_index.faiss")

loaded_index = faiss.read_index("my_index.faiss")
print(f"从磁盘加载索引，包含 {loaded_index.ntotal} 个向量")
```

### 6.2 FAISS + IVFFlat（倒排索引，大规模场景）

```python
"""FAISS 倒排索引 — 适合百万级以上向量"""
import numpy as np
import faiss

dimension = 128
nlist = 100  # 聚类中心数量（越大 → 精度越高，速度越慢）

# ============================================
# Step 1: 创建量化器 + IVFFlat 索引
# ============================================
quantizer = faiss.IndexFlatL2(dimension)  # 底层使用精确搜索
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# ============================================
# Step 2: 训练 — IVFFlat 必须先训练！
# ============================================
# 用 10000 条代表性数据训练 K-Means 聚类
train_vectors = np.random.random((10000, dimension)).astype('float32')
index.train(train_vectors)
print("训练完成")

# ============================================
# Step 3: 添加向量
# ============================================
add_vectors = np.random.random((50000, dimension)).astype('float32')
index.add(add_vectors)
print(f"已添加 {index.ntotal} 个向量")

# ============================================
# Step 4: 检索 — 注意 nprobe 参数
# ============================================
index.nprobe = 10  # 搜索时探测的聚类中心数（越大 → 精度越高，速度越慢）
query = np.random.random((3, dimension)).astype('float32')
distances, indices = index.search(query, k=5)
print(f"检索完成，结果 shape: {indices.shape}")
```

### 6.3 FAISS + HNSW（图索引，高精度场景）

```python
"""FAISS HNSW 索引 — 无需训练，检索质量高"""
import numpy as np
import faiss

dimension = 128

# 创建 HNSW 索引
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M（每个节点的连接数）
# M 越大 → 精度越高，内存越大（推荐 16-64）

vectors = np.random.random((20000, dimension)).astype('float32')
index.add(vectors)
print(f"HNSW 索引包含 {index.ntotal} 个向量")

# 检索
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)
print(f"最近邻索引: {indices[0]}")
print(f"对应距离: {distances[0]}")

# 注意：HNSW 不能用 write_index 保存，需要序列化
# 可使用 pickle 或 faiss.serialize_index() (部分版本支持)
```

### 6.4 FAISS 与 LangChain 集成

```python
"""FAISS + LangChain 完整 RAG 检索 — 对应你的 C3 示例"""
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

# 1. 加载与切分文档
loader = TextLoader("your_document.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = splitter.split_documents(docs)
print(f"文档切分为 {len(splits)} 个 chunks")

# 2. 创建 Embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    model="text-embedding-3-small"
)

# 3. 从文档构建 FAISS 向量库
vectorstore = FAISS.from_documents(splits, embeddings)
print("FAISS 向量库构建完成")

# 4. 转换为检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",     # 相似度检索
    search_kwargs={"k": 4}        # 返回 Top-4
)

# 5. 直接使用 similarity_search
question = "LangGraph 是什么？"
results = vectorstore.similarity_search(question, k=3)
for i, doc in enumerate(results):
    print(f"\n--- 结果 {i+1} ---")
    print(f"内容: {doc.page_content[:100]}...")
    print(f"元数据: {doc.metadata}")

# 6. 带分数的检索
results_with_scores = vectorstore.similarity_search_with_score(question, k=3)
for doc, score in results_with_scores:
    print(f"分数(L2距离,越小越相似): {score:.4f} | 内容: {doc.page_content[:80]}...")

# 7. 持久化 FAISS 索引
vectorstore.save_local("faiss_index_dir")
print("FAISS 索引已保存")

# 8. 从磁盘加载
loaded_vectorstore = FAISS.load_local(
    "faiss_index_dir",
    embeddings,
    allow_dangerous_deserialization=True  # ⚠️ 仅加载自己保存的索引时使用
)
print("FAISS 索引已从磁盘加载")

# 9. MMR 检索（最大边际相关性 — 兼顾相关性和多样性）
mmr_results = vectorstore.max_marginal_relevance_search(
    question, k=3, fetch_k=10, lambda_mult=0.5
)
# lambda_mult: 0=最大多样性, 1=最大相关性
print(f"MMR 检索返回 {len(mmr_results)} 个结果")
```

### 6.5 FAISS 带 ID 映射（管理外部 ID）

```python
"""FAISS 使用 IndexIDMap 管理外部 ID"""
import numpy as np
import faiss

dimension = 128

# 底层索引 + ID 映射包装
base_index = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(base_index)

# 准备向量和自定义 ID
vectors = np.random.random((100, dimension)).astype('float32')
custom_ids = np.arange(1000, 1100).astype('int64')  # 自定义 ID 范围

index.add_with_ids(vectors, custom_ids)

# 检索时返回的就是自定义 ID
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=3)
print(f"返回的自定义 ID: {indices[0]}")  # 在 1000-1099 范围内
```

---

## 七、Chroma 实战教学

### 7.1 Chroma 原生 API 使用

```python
"""Chroma 原生 API 完整演示"""
import chromadb
from chromadb.config import Settings

# ============================================
# 模式 1: 内存模式（Ephemeral）— 进程结束数据消失
# ============================================
client = chromadb.Client()
# 或显式指定: chromadb.EphemeralClient()

collection = client.create_collection(
    name="demo_memory",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# ============================================
# 模式 2: 持久化模式（Persistent）— 数据存磁盘
# ============================================
persistent_client = chromadb.PersistentClient(path="./chroma_data")

collection = persistent_client.get_or_create_collection(
    name="my_knowledge_base",
    metadata={"hnsw:space": "cosine"}  # l2 / ip / cosine
)

# ============================================
# 添加文档
# ============================================
collection.add(
    documents=[
        "LangChain 是一个用于开发 LLM 应用的框架。",
        "LangGraph 用于构建带循环的多智能体系统。",
        "MCP 是模型上下文协议，让代理接入外部工具。",
        "RAG 是检索增强生成，结合了信息检索和文本生成。",
    ],
    metadatas=[
        {"source": "docs", "topic": "langchain"},
        {"source": "docs", "topic": "langgraph"},
        {"source": "docs", "topic": "mcp"},
        {"source": "docs", "topic": "rag"},
    ],
    ids=["doc_1", "doc_2", "doc_3", "doc_4"],
    # 如果不传 embeddings，Chroma 会使用默认的 all-MiniLM-L6-v2 自动向量化
)
print(f"集合中有 {collection.count()} 个文档")

# ============================================
# 查询 — 按文本语义检索
# ============================================
results = collection.query(
    query_texts=["什么是多智能体系统？"],
    n_results=2,              # 返回 Top-2
    # where={"topic": "langgraph"},  # 元数据过滤（可选）
)
print(f"\n查询结果:")
for i, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0])
):
    print(f"  {i+1}. [{distance:.4f}] {doc}")

# ============================================
# 元数据过滤查询
# ============================================
results = collection.query(
    query_texts=["工具和协议"],
    n_results=3,
    where={"topic": {"$in": ["mcp", "rag"]}},  # 只在指定 topic 中搜索
)
print(f"\n过滤查询结果: {results['documents'][0]}")

# ============================================
# 增删改操作
# ============================================
# 更新
collection.update(
    ids=["doc_1"],
    documents=["LangChain 是一个强大的 LLM 应用开发框架。"],
    metadatas=[{"source": "docs", "topic": "langchain", "version": "v2"}],
)

# 删除单条
collection.delete(ids=["doc_4"])

# 按元数据条件批量删除
# collection.delete(where={"topic": "rag"})

print(f"操作后集合中有 {collection.count()} 个文档")

# ============================================
# 获取文档
# ============================================
doc = collection.get(ids=["doc_1"])
print(f"\n获取 doc_1: {doc['documents'][0]}")

# 获取全部（谨慎使用，数据量大时很慢）
all_docs = collection.get()
print(f"全部文档数: {len(all_docs['ids'])}")
```

### 7.2 Chroma + 自定义 Embedding

```python
"""Chroma 配合自定义 Embedding 函数"""
import chromadb
from chromadb.utils import embedding_functions

# ============================================
# 选项 1: 使用 OpenAI Embedding
# ============================================
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    api_base="your-api-base",   # 兼容端点时使用
    model_name="text-embedding-3-small"
)

# ============================================
# 选项 2: 使用 Sentence Transformers（本地运行）
# ============================================
sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # 本地小模型，无需 API
)

# ============================================
# 选项 3: 自定义 Embedding 函数
# ============================================
class MyEmbeddingFunction:
    """自定义 embedding 函数包装器"""
    def __init__(self, embed_func):
        self.embed_func = embed_func

    def __call__(self, texts):
        # texts: List[str]
        # 返回: List[List[float]]
        return self.embed_func(texts)

# 创建带自定义 embedding 函数的集合
client = chromadb.PersistentClient(path="./chroma_custom_db")

collection = client.create_collection(
    name="custom_embedding_demo",
    embedding_function=openai_ef,  # 传入你的 embedding 函数
    metadata={"hnsw:space": "cosine"}
)

# 此后添加文档时不再需要手动传 embedding
# Chroma 会自动调用 embedding_function 来向量化
collection.add(
    documents=["这是需要向量化的文本"],
    ids=["doc_custom_1"]
)
```

### 7.3 Chroma 与 LangChain 集成

```python
"""Chroma + LangChain 完整 RAG 检索"""
from langchain_chroma import Chroma
# 或旧版：from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

# 1. 加载与切分文档
loader = TextLoader("your_document.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = splitter.split_documents(docs)

# 2. 创建 Embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    model="text-embedding-3-small"
)

# 3. 从文档构建 Chroma 向量库（持久化到磁盘）
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",  # 持久化路径
    collection_name="my_docs"
)
print(f"Chroma 向量库已创建，持久化目录: ./chroma_langchain_db/")

# 4. 下次使用时直接加载（不再需要重新构建）
vectorstore = Chroma(
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
    collection_name="my_docs"
)

# 5. 转换为检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        # "filter": {"source": "pdf"}  # 元数据过滤
    }
)

# 6. 相似度检索
question = "LangGraph 的核心功能是什么？"
results = vectorstore.similarity_search(question, k=3)
for i, doc in enumerate(results):
    print(f"\n结果 {i+1}: {doc.page_content[:100]}...")
    print(f"  元数据: {doc.metadata}")

# 7. 带分数的检索
results = vectorstore.similarity_search_with_score(question, k=3)
for doc, score in results:
    print(f"[{score:.4f}] {doc.page_content[:80]}...")

# 8. MMR 检索（多样性 + 相关性）
mmr_results = vectorstore.max_marginal_relevance_search(
    question, k=3, fetch_k=15, lambda_mult=0.6
)
print(f"\nMMR 检索返回 {len(mmr_results)} 个结果")

# 9. 元数据过滤检索
# Chroma 在 LangChain 中的过滤语法：
filtered_results = vectorstore.similarity_search(
    question,
    k=3,
    filter={"source": {"$eq": "pdf"}}  # Chroma where 语法
)
```

### 7.4 Chroma Client/Server 模式

```python
"""Chroma 的 Client/Server 模式 — 适合团队共享"""
# ═══════════════════════════════════════════
# 启动 Chroma 服务端（在终端中执行）：
# chroma run --path /data/chroma_db --host 0.0.0.0 --port 8000
# ═══════════════════════════════════════════

import chromadb

# 客户端连接远程 Chroma 服务
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    # ssl=True,         # 生产环境使用 HTTPS
    # headers={"X-Api-Key": "your-api-key"}  # 认证
)

# 检查服务端状态
heartbeat = client.heartbeat()
print(f"Chroma 服务心跳: {heartbeat}")

# 之后的使用方式完全一样
collection = client.get_or_create_collection(name="shared_kb")
collection.add(
    documents=["团队共享的知识文档"],
    ids=["shared_1"]
)
```

---

## 八、进阶话题

### 8.1 相似度度量对比

```
┌──────────────┬──────────────────────┬──────────────────────┐
│    度量       │      含义             │    何时使用           │
├──────────────┼──────────────────────┼──────────────────────┤
│ Cosine       │ 夹角余弦，值域[-1, 1]  │ 文本 Embedding 首选  │
│              │ 1 = 完全相同方向       │ 只关心"方向"不关心"长度"│
├──────────────┼──────────────────────┼──────────────────────┤
│ L2 (欧氏距离) │ 直线距离，越小越相似   │ 向量长度有意义时     		│
│              │                      │ 图像特征、位置数据    	│
├──────────────┼──────────────────────┼──────────────────────┤
│ IP (内积)    │ 点积，越大越相似       │ 未归一化向量的相似度 		 │
│              │ 等价于 Cosine（归一化后）│ 推荐系统             │
└──────────────┴──────────────────────┴──────────────────────┘
```

> **实践建议**：OpenAI 的 Embedding 已经做了 L2 归一化，因此用 Cosine 和 IP 是等价的。L2 也能用但数值范围不同。

### 8.2 chunk_size 调优

这是 RAG 系统中**最影响效果的参数之一**：

```python
# 小 chunk → 精度高但可能丢失上下文
splitter_small = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

# 中 chunk → 平衡之选（大多数场景的起点）
splitter_mid = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 大 chunk → 上下文完整但可能引入噪声
splitter_large = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
```

**经验法则**：
- 问答型 RAG：chunk_size 300-800
- 摘要型 RAG：chunk_size 1000-2000
- overlap 一般为 chunk_size 的 10%-20%

### 8.3 检索策略：Similarity vs MMR

```python
# Similarity — 返回最相似的 K 个（可能重复/相似）
retriever_sim = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# MMR — 在相关性和多样性之间平衡
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,      # 先从候选池中取 20 个
        "lambda_mult": 0.5  # 0=最大多样性, 1=最大相关性
    }
)

# Similarity Score Threshold — 低于阈值不返回
retriever_thresh = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 4}
)
```

### 8.4 生产环境部署建议

```python
"""
┌─────────────────────────────────────────────────────────┐
│                    生产部署决策树                         │
│                                                         │
│  你的数据量有多大？                                       │
│  ├─ < 100K 文档                                          │
│  │  └─ 需要复杂过滤？→ Chroma                            │
│  │     不需要过滤？→ 两者都行，选你顺手的                   │
│  │                                                       │
│  ├─ 100K ~ 1M 文档                                       │
│  │  └─ 需要持久化+过滤 → Chroma（SSD 要给力）               │
│  │     追求速度+无过滤 → FAISS IVFFlat                   │
│  │                                                       │
│  └─ > 1M 文档                                            │
│     └─ FAISS IVFPQ 或 HNSW + GPU                         │
│        或考虑：Milvus / Qdrant / Weaviate / Pinecone      │
│                                                         │
│  你的部署方式？                                           │
│  ├─ 嵌入式（无服务器）→ FAISS                            │
│  ├─ Client/Server → Chroma / Qdrant / Milvus            │
│  └─ 全托管云服务 → Pinecone / Zilliz Cloud              │
└─────────────────────────────────────────────────────────┘
"""
```

---

## 九、选型决策指南

### 9.1 快速决策表

| 你的场景 | 推荐 | 理由 |
|---|---|---|
| "我在学习 RAG，先跑通流程" | **Chroma** | API 最友好，自动持久化，心智负担最低 |
| "我要做原型/Demo" | **Chroma** | 内存模式零配置启动，不需要额外服务 |
| "我的文档量很大（百万级+）" | **FAISS** | 倒排索引+量化压缩，内存效率极高 |
| "我需要复杂元数据过滤" | **Chroma** | `where` 语法原生支持，表达力强 |
| "我要在生产环境部署" | **Chroma** | Client/Server 模式，持久化可靠 |
| "我跑在嵌入式设备/无服务器上" | **FAISS** | 纯 C++ 库，无外部依赖，体积小 |
| "我需要 GPU 加速检索" | **FAISS** | 这是 FAISS 的核心优势 |
| "我用 LangChain 搭 RAG" | 两者都行 | LangChain 对两者都有良好封装 |

### 9.2 最终建议

```
学习路径推荐:

阶段 1 (你在这里) ──→ 用 Chroma
  理由: 自动持久化 + 元数据过滤，让你专注于理解 RAG 流程而不是管理索引

阶段 2 ──→ 用 FAISS (Flat)
  理由: 理解底层索引机制，为大规模场景做准备

阶段 3 ──→ FAISS (IVFPQ / HNSW) + GPU
  理由: 真正需要处理百万级+数据时

补充考察 ──→ Milvus / Qdrant / Pinecone
  理由: 当 FAISS/Chroma 无法满足你的分布式/高可用需求时
```

---

> 📚 **参考资源**
> - FAISS Wiki: <https://github.com/facebookresearch/faiss/wiki>
> - Chroma 文档: <https://docs.trychroma.com/>
> - LangChain VectorStores: <https://python.langchain.com/docs/integrations/vectorstores/>
> - 你的本地示例: [C3_rag_pipeline_example.py](C3_rag_pipeline_example.py)
