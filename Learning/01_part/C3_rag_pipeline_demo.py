import os
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. 准备演示用的示例文档内容
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
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")
async def main():
    print("🚀 === RAG (检索增强生成) 管道演示 === 🚀\n")
    
    # 检查配置
    if not API_KEY:
        print("⚠️ 警告: 未检测到 OPENAI_API_KEY 环境变量，请确保已在 .env 文件中正确配置。")
        return

    # 创建演示用文本文件
    temp_filename = "demo_rag_source.txt"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(DEMO_DOC_CONTENT)
        
    try:
        # ---- 步骤 1: 文档加载 (Document Loading) ----
        print("--- 步骤 1: 加载文档 ---")
        loader = TextLoader(temp_filename, encoding="utf-8")
        docs = loader.load()
        print(f"成功加载文档。文档字符长度: {len(docs[0].page_content)}\n")
        
        # ---- 步骤 2: 文本切分 (Text Splitting) ----
        print("--- 步骤 2: 文本切分 (Chunking) ---")
        # 较小的 chunk_size 有助于精准定位，overlap 避免信息截断
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=80, 
            chunk_overlap=20
        )
        splits = splitter.split_documents(docs)
        print(f"文档已被切分为 {len(splits)} 个 Chunks。")
        for i, chunk in enumerate(splits):
            print(f"  [Chunk {i}]: {repr(chunk.page_content)}")
        print()
        
        # ---- 步骤 3: 向量化与存储 (Embeddings & Vector Store) ----
        print("--- 步骤 3: 向量化与构建索引 ---")
        embeddings = OpenAIEmbeddings(
            openai_api_key=EMBEDDINGS_API_KEY,
            openai_api_base=EMBEDDINGS_BASE_URL,
            model=EMBEDDINGS_MODEL_NAME # 如果需要可在此指定嵌入模型
        )
        
        persist_directory = "./chroma_db"
        
        # 判断本地是否存在已持久化的数据库，如果存在则直接加载，不存在则创建
        if os.path.exists(persist_directory):
            print(f"检测到本地已存在已持久化的向量库，正在从 '{persist_directory}' 加载...")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print("向量索引 (Chroma) 加载成功。\n")
        else:
            print(f"未检测到本地向量库，正在创建并持久化到 '{persist_directory}'...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print("向量索引 (Chroma) 构建并持久化成功。\n")
        
        # ---- 步骤 4: 创建检索器 (Retriever) ----
        print("--- 步骤 4: 创建检索器 ---")
        # 配置 k=2，表示检索最相似的 2 个 chunk
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print("检索器创建完毕。\n")
        
        # ---- 步骤 5: 组装 RAG 链 (LCEL) ----
        print("--- 步骤 5: 组装 LCEL RAG 链 ---")
        
        # 提示词模版
        template = """你是一个专业的AI助手。请基于以下提供的上下文片段回答用户的问题。
如果你在上下文中找不到答案，就诚实地说你不知道，不要胡乱编造。

【上下文】：
{context}

【问题】：{question}
【回答】："""
        prompt = ChatPromptTemplate.from_template(template)
        
        # 实例化大模型
        llm = ChatOpenAI(
            api_key=API_KEY, 
            base_url=BASE_URL, 
            model=MODEL_NAME,
            temperature=0
        )
        
        # 定义辅助函数：把检索到的文档列表拼接成一段文本
        def format_docs(docs_list):
            formatted = "\n---\n".join(doc.page_content for doc in docs_list)
            return formatted

        # 使用 RunnableParallel 显式捕获中间步骤 (context 和 question)
        # 这样便于在 LCEL 流程中并联执行检索
        rag_chain = (
            RunnableParallel({
                "context": retriever | format_docs, 
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG 链组装完成。\n")
        
        # ---- 步骤 6: 测试运行 ----
        question = "请问 LangGraph 有什么用？它的核心特点是什么？"
        print(f"--- 步骤 6: 提问 ---")
        print(f"问题: '{question}'")
        
        # 演示检索阶段
        retrieved_docs = await retriever.ainvoke(question)
        print("\n🔍 [检索器召回的文本片段]:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  召回片段 {i+1}: {doc.page_content}")
            
        print("\n🤖 [大模型生成回答中...]")
        # 运行 RAG 链
        answer = await rag_chain.ainvoke(question)
        print(f"最终回答:\n{answer}")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("请检查你的 API_KEY、BASE_URL 网络连接以及依赖包是否正确安装。")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print("\n🧹 临时文件已清理。")

if __name__ == "__main__":
    asyncio.run(main())
