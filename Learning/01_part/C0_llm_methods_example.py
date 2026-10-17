import os
import asyncio
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

"""
LangChain 中 LLM (ChatOpenAI 等 Runnable 对象) 的核心方法全景解析。
除了 llm.bind()，这里展示了 LangChain 中最常用和最重要的实例方法。
分为三大类：1. 运行时配置绑定类  2. 执行与流式输出类  3. 容错与高阶功能类
"""

# ==========================================
# 准备工作：定义一个工具和一个 Pydantic 模型
# ==========================================
@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气"""
    return f"{location}今天晴，25度"

class WeatherResponse(BaseModel):
    """结构化的天气响应"""
    location: str = Field(description="城市名称")
    temperature: float = Field(description="温度，数字格式")
    description: str = Field(description="天气描述")


async def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=API_KEY, base_url=BASE_URL)
    messages = [HumanMessage(content="北京天气怎么样？")]

    print("==========================================")
    print(" 1. 运行时配置与功能绑定类 (Binding)")
    print("==========================================")

    # 1.1 bind_tools: 将 Python 函数、Pydantic 类或 Tool 对象绑定给模型，使其具备调用工具的能力
    print("\n[bind_tools] 绑定工具...")
    llm_with_tools = llm.bind_tools([get_weather])
    # 此时模型如果判断需要查天气，返回的 content 为空，而 tool_calls 会包含调用信息
    
    # 1.2 with_structured_output: 强制模型输出符合你定义的 Pydantic 数据结构的 JSON
    print("\n[with_structured_output] 强制结构化输出...")
    try:
        structured_llm = llm.with_structured_output(WeatherResponse)
        # 只要你调用 structured_llm.invoke，它直接返回 WeatherResponse 对象，而不是 AIMessage
    except NotImplementedError:
        pass # 有些旧模型/第三方API不支持原生强结构化，可以使用 bind_tools 替代

    # 1.3 with_config: 动态注入 LangChain 运行时的配置 (如 Tags, Metadata, 回调函数, 递归限制)
    print("\n[with_config] 绑定运行时配置 (Tags/Metadata)...")
    configurable_llm = llm.with_config(
        tags=["weather_agent_test"], 
        metadata={"user_id": "123"},
        run_name="MyCustomLLMCall" 
    )


    print("\n==========================================")
    print(" 2. 执行与流式输出类 (Execution & Streaming)")
    print("==========================================")
    # 所有的执行方法都有 同步(无 'a' 前缀) 和 异步(有 'a' 前缀) 两个版本。推荐使用异步。

    # 2.1 invoke / ainvoke: 最基础的单次调用，输入一句/一组消息，返回一条回复 (AIMessage)
    print("\n[ainvoke] 基础调用...")
    # response = await llm.ainvoke("你好！")

    # 2.2 stream / astream: 流式调用，逐个字/token 返回结果，用于实现打字机效果
    print("\n[astream] 流式输出演示...")
    # async for chunk in llm.astream("请从 1 数到 5"):
    #     print(chunk.content, end="", flush=True)

    # 2.3 batch / abatch: 批量并发调用。传入一个列表，它会自动并发处理所有请求
    print("\n[abatch] 批量并发调用...")
    # responses = await llm.abatch([
    #     "说一个关于苹果的冷知识", 
    #     "说一个关于香蕉的冷知识"
    # ])

    # 2.4 astream_events: 极为强大的事件流接口！常用于复杂的 Agent 场景。
    # 它可以把链条 (Chain/Agent) 中每个组件(llm, tool, parser) 启动、结束、流式输出的过程，全部抛出事件。
    print("\n[astream_events] 监听底层事件流...")
    # async for event in llm.astream_events("你好", version="v2"):
    #     if event["event"] == "on_chat_model_stream":
    #         print(event["data"]["chunk"].content, end="")


    print("\n==========================================")
    print(" 3. 容错与高阶组件类 (Resilience & Advanced)")
    print("==========================================")

    # 3.1 with_fallbacks: 提供备用模型！如果当前模型(如 GPT-4o)因为网络或限流报错，自动回退到备用模型(如 GPT-3.5)
    print("\n[with_fallbacks] 配置降级备用模型...")
    backup_llm = ChatOpenAI(model="gpt-3.5-turbo", max_retries=1)
    robust_llm = llm.with_fallbacks(fallbacks=[backup_llm])
    
    # 3.2 with_retry: 为单个组件配置更细粒度的重试策略(重试次数、延迟、特定异常拦截)
    # 注意：ChatOpenAI 初始化就有 max_retries，但 with_retry 可以给任意 Runnable(包括 Prompt/Parser) 增加重试
    print("\n[with_retry] 配置重试策略...")
    retryable_llm = llm.with_retry(
        stop_after_attempt=3, 
        wait_exponential_jitter=True # 指数退避抖动
    )

    # 3.3 __or__ (就是 | 符号): Runnable 核心特性。将 llm 与 prompt、parser 组合成链
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    print("\n[ | 操作符] 组合链 (Chain)...")
    prompt = ChatPromptTemplate.from_template("把这段话翻译成英文：{text}")
    chain = prompt | llm | StrOutputParser()
    # await chain.ainvoke({"text": " LangChain 太棒了！"})

if __name__ == "__main__":
    asyncio.run(main())
