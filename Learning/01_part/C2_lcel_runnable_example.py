import os
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME, temperature=0)

# 【模块 3】LCEL 与 Runnable 原语核心用法

# 1. RunnableLambda - 将普通 Python 函数塞进 LCEL 链中
def clean_input(text: str) -> str:
    """清理输入空格，转换为小写"""
    return text.strip().lower()

# 2. 基础 Prompt 和 Parser
classify_prompt = ChatPromptTemplate.from_template("判断这段文本的情感是 'positive', 'negative' 还是 'neutral'，只输出这个英文单词:\n{text}")
positive_prompt = ChatPromptTemplate.from_template("你收到了一个好评，请礼貌地感谢他，并简短回复:\n{text}")
negative_prompt = ChatPromptTemplate.from_template("你收到了一个差评，请诚恳地道歉并提出解决方案:\n{text}")

# 3. 路由判断逻辑
def is_positive(inputs: dict) -> bool:
    return "positive" in inputs["sentiment"].lower()

def is_negative(inputs: dict) -> bool:
    return "negative" in inputs["sentiment"].lower()

# 4. RunnableBranch - 条件分支路由
branch = RunnableBranch(
    (is_positive, positive_prompt | llm | StrOutputParser()),
    (is_negative, negative_prompt | llm | StrOutputParser()),
    # fallback (neutral)
    RunnableLambda(lambda x: "感谢您的中肯反馈，我们会继续努力优化！")
)

async def main():
    print("=== 模块 3: LCEL / Runnable 核心原语实战 ===\n")
    
    # 5. RunnableParallel + Passthrough.assign 的高阶组装 (RAG 中的核心胶水层)
    sentiment_chain = classify_prompt | llm | StrOutputParser()
    
    # master_chain 演示了: 管道(|)、透传与注入(assign)、路由(branch)、重试和配置注入
    master_chain = (
        # 步骤 1: 预处理输入文本
        {"text": RunnableLambda(clean_input)}  #将改函数包装为标准的Runnable对象。以字典的形式在管道中传递
        # 步骤 2: 保留原文本字典，并并行执行情感分析，结果赋给新字段 "sentiment"
        | RunnablePassthrough.assign(sentiment=sentiment_chain)
        # 步骤 3: 路由分支，根据 sentiment 决定走哪个 Prompt
        | branch
    ).with_config(run_name="CustomerServiceChain") \
     .with_retry(stop_after_attempt=3) # 带有重试机制的生产级配置
    
    print("[1] 测试路由 - 好评路线:")
    res_pos = await master_chain.ainvoke("   这个产品太棒了，我很喜欢！   ")
    print(f"-> 回复: {res_pos}")
    
    print("\n[2] 测试路由 - 差评路线:")
    res_neg = await master_chain.ainvoke("质量很差，退钱！根本没法用。")
    print(f"-> 回复: {res_neg}")
    
    # 6. astream_events: 获取链内部运行事件 (模块6：可观测性必备能力)
    print("\n[3] 测试 astream_events (监控链的内部数据流转):")
    # 可以观察到 RunnableLambda、模型调用、分支判断的各个阶段
    async for event in master_chain.astream_events("质量一般般吧，没想象中好", version="v2"):
        kind = event["event"]
        name = event["name"]
        if kind == "on_chain_start":
            print(f" -> [开始执行]: {name}")
        elif kind == "on_chat_model_end":
            print(f" -> [模型返回]: {name}")

if __name__ == "__main__":
    asyncio.run(main())
