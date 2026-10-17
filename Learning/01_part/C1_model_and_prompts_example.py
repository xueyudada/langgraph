import os
import asyncio
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field

# LangChain 核心导入
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 加载 .env 文件中的配置
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# 【模块 6】LangSmith 追踪机制 (取消注释即可开启)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# 1. 结构化输出定义 (PydanticModel)
class UserProfile(BaseModel):
    """用户信息结构化输出模型"""
    name: str = Field(description="用户的姓名")
    age: Optional[int] = Field(description="用户的年龄，如果未提及则为空")
    interests: List[str] = Field(description="用户的兴趣爱好列表")

async def main():
    print("=== 模块 1 & 2: 模型接入与 Prompt 与结构化输出 ===\n")

    # 【模块 1】模型接入主入口
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=0.7
    )

    # 也可以使用 init_chat_model 工厂函数 (适合多模型动态切换)
    # from langchain.chat_models import init_chat_model
    # llm_factory = init_chat_model("openai:" + MODEL_NAME, base_url=BASE_URL, api_key=API_KEY)

    # 1. Messages 体系与同步调用 (invoke)
    # messages = [
    #     SystemMessage(content="你是一个专业的文本信息提取助手。"),
    #     HumanMessage(content="我叫张三，今年28岁，喜欢打篮球和看科幻小说。")
    # ]
    # print("[1] 测试基础 Invoke:")
    # response = await llm.ainvoke(messages)
    # print(f"-> 原始 AIMessage content: {response.content}")
    # 注意: 如果是多模态或工具调用，数据可能在 response.content_blocks 中

    # 2. Bind 绑定运行期参数
    bound_llm = llm.bind(temperature=0.1)

    # 【模块 2】PromptTemplate 与 MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个名叫 {bot_name} 的助手,请以 JSON 格式回答。"),
        # MessagesPlaceholder 用于动态插入多轮聊天历史记录
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])
    
    # 3. 结构化输出 (Agent 底层真实逻辑：使用 bind_tools)
    # 既然 with_structured_output 因为强制塞入 response_format 被服务器拒绝，
    # 我们直接演示 Agent 的底层原理：把 Pydantic 类当作 Tool 绑定给大模型！
    print("\n[2] 测试结构化输出 (模拟 Agent 底层 bind_tools 逻辑):")
    
    # 这一步就是 create_react_agent 底层在做的事情
    llm_with_tools = llm.bind_tools([UserProfile])
    
    # 我们直接把 Prompt 和带有 Tool 的模型连起来
    chain_structured = prompt | llm_with_tools
    
    # 执行调用
    ai_msg = await chain_structured.ainvoke({
        "bot_name": "信息收集者",
        "chat_history": [],
        "user_input": "我叫李华，平时喜欢听音乐和跑步。"
    })
    
    # 大模型不会返回普通文本，而是会发起一个 tool_call，参数正好完美符合 UserProfile！
    print(f"-> 原始返回的消息对象: {ai_msg.content}")
    print(f"-> 触发的工具调用 (tool_calls): {ai_msg.tool_calls}")
    
    if ai_msg.tool_calls:
        # 提取出工具调用里的参数，这就是我们要的结构化数据！
        extracted_data = ai_msg.tool_calls[0]["args"]
        print(f"-> 完美提取出的结构化字典: {extracted_data}")
    
    # 4. StrOutputParser 与流式输出 (astream)
    print("\n[3] 测试 StrOutputParser 与 流式输出 (astream):")
    chain_str = prompt | bound_llm | StrOutputParser()
    
    async for chunk in chain_str.astream({
        "bot_name": "小助手",
        "chat_history": [SystemMessage(content="之前提过我是程序员")],
        "user_input": "结合我之前的职业，简短推荐我学哪门新语言？"
    }):
        # 流式打印前端逐字输出
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
